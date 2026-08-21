import json
import sys
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Ensure root is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features, compute_residuals_intervals

DATA_PATH = Path("data/raw/data10yrs.csv")
DEMAND_ART_DIR = Path("ml/artifacts/demand")

ITEMS = ["paracetamol", "ors_packets", "malaria_kits", "antibiotics"]

FEATURE_COLS = [
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
    "dow",
    "month",
    "temperature",
    "rainfall",
    "humidity",
]

def build_demand_dataset(df: pd.DataFrame, item_code: str) -> Tuple[pd.DataFrame, pd.Series]:
    col_name = f"{item_code}_used"
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in CSV.")
        
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, col_name, lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])
    
    lag_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=lag_cols + [col_name]).reset_index(drop=True)
    
    X = d[FEATURE_COLS].fillna(0)
    y = d[col_name].astype(float)
    return X, y

def train_demand_models():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    for item in ITEMS:
        print(f"\n--- Training Demand Model for '{item}' ---")
        X, y = build_demand_dataset(df, item)
        
        split_idx = len(X) - 365
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        
        preds_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds_test)
        print(f"[{item}] Test MAE: {mae:.2f} units")
        
        # Fit full model
        full_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        full_model.fit(X, y)
        full_preds = full_model.predict(X)
        intervals = compute_residuals_intervals(y.values, full_preds)
        
        item_dir = DEMAND_ART_DIR / item
        item_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_model, item_dir / "model.pkl")
        with open(item_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_COLS}, f, indent=2)
        with open(item_dir / "intervals.json", "w", encoding="utf-8") as f:
            json.dump(intervals, f, indent=2)
            
        print(f"[OK] Saved artifacts for '{item}' in {item_dir}")

if __name__ == "__main__":
    train_demand_models()
