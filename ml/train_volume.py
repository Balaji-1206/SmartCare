import json
import sys
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure root is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features, compute_residuals_intervals

DATA_PATH = Path("data/raw/data10yrs.csv")
ART_DIR = Path("ml/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "dow",
    "month",
    "is_weekend",
    "temperature",
    "rainfall",
    "humidity",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_mean_28",
    "roll_std_28",
]

def build_volume_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, "total_patients", lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])
    
    # Drop rows without full lag context
    lag_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=lag_cols + ["total_patients"]).reset_index(drop=True)
    
    X = d[FEATURE_COLS].fillna(0)
    y = d["total_patients"].astype(float)
    return X, y

def train_volume_model():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_volume_dataset(df)
    
    # Temporal train/test split (last 365 days as test)
    split_idx = len(X) - 365
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Volume Model on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    print(f"Test MAE: {mae:.2f} visits | Test R2: {r2:.3f}")
    
    # Fit full model for production
    full_model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    full_model.fit(X, y)
    full_preds = full_model.predict(X)
    
    intervals = compute_residuals_intervals(y.values, full_preds)
    print(f"Residual intervals: {intervals}")
    
    # Export artifacts
    joblib.dump(full_model, ART_DIR / "volume_model.pkl")
    with open(ART_DIR / "volume_features.json", "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_COLS}, f, indent=2)
    with open(ART_DIR / "volume_intervals.json", "w", encoding="utf-8") as f:
        json.dump(intervals, f, indent=2)
        
    print("[OK] Volume model artifacts saved successfully.")

if __name__ == "__main__":
    train_volume_model()
