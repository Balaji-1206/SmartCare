import json
import sys

# Guard against environments where pyarrow C-extension is blocked by system security policies
if "pyarrow" not in sys.modules:
    try:
        import pyarrow  # noqa: F401
    except (ImportError, Exception):
        sys.modules["pyarrow"] = None

from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure root is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import (
    add_calendar_features,
    add_lag_and_rolling_features,
    add_weather_features,
    compute_residuals_intervals,
)

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
    d = add_lag_and_rolling_features(
        d, "total_patients", lags=[1, 7, 14, 28], roll_windows=[7, 14, 28]
    )
    
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
    
    # Temporal train/test split (last 365 days as hold-out test set)
    split_idx = len(X) - 365
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n=======================================================")
    print(f"  Training Volume Models on {len(X_train)} samples")
    print(f"  Evaluating on {len(X_test)} unseen test samples")
    print(f"=======================================================")
    
    # 1. Point prediction model (conditional expectation)
    print("\n[1/3] Training Point Forecast Model...")
    point_model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
    )
    point_model.fit(X_train, y_train)
    preds_test = point_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    print(f"      -> Test MAE: {mae:.2f} visits | Test R2: {r2:.3f}")
    
    # 2. Lower quantile model (alpha=0.10: 10th percentile floor)
    print("\n[2/3] Training Quantile Model (alpha=0.10, Lower Bound Floor)...")
    q10_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    q10_model.fit(X_train, y_train)
    p10_test = q10_model.predict(X_test)
    
    # 3. Upper quantile model (alpha=0.90: 90th percentile surge ceiling)
    print("\n[3/3] Training Quantile Model (alpha=0.90, Surge Capacity Ceiling)...")
    q90_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    q90_model.fit(X_train, y_train)
    p90_test = q90_model.predict(X_test)
    
    # Quantile Calibration Evaluation
    coverage = float(np.mean((y_test.values >= p10_test) & (y_test.values <= p90_test)))
    widths = p90_test - p10_test
    mean_width = float(np.mean(widths))
    min_width = float(np.min(widths))
    max_width = float(np.max(widths))
    
    print("\n----------------- Quantile Interval Calibration -----------------")
    print(f"  Target Coverage:       80.0%")
    print(f"  Empirical Coverage:    {coverage * 100:.1f}%")
    print(f"  Mean Interval Width:   {mean_width:.1f} visits")
    print(f"  Min Width (calm days): {min_width:.1f} visits")
    print(f"  Max Width (surge days):{max_width:.1f} visits")
    print("-----------------------------------------------------------------")
    
    # Fit full models on entire dataset for production serving
    print("\nFitting full production models on entire dataset (3,654 records)...")
    full_point = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
    )
    full_point.fit(X, y)
    
    full_q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    full_q10.fit(X, y)
    
    full_q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    full_q90.fit(X, y)
    
    full_preds = full_point.predict(X)
    intervals_fallback = compute_residuals_intervals(y.values, full_preds)
    intervals_payload = {
        **intervals_fallback,
        "coverage_80": round(coverage, 3),
        "mean_interval_width": round(mean_width, 1),
        "min_interval_width": round(min_width, 1),
        "max_interval_width": round(max_width, 1),
        "model_type": "multi_quantile_gradient_boosting",
    }
    
    # Export artifacts
    joblib.dump(full_point, ART_DIR / "volume_model.pkl")
    joblib.dump(
        {"q10": full_q10, "q50": full_point, "q90": full_q90},
        ART_DIR / "volume_quantiles.pkl"
    )
    with open(ART_DIR / "volume_features.json", "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_COLS}, f, indent=2)
    with open(ART_DIR / "volume_intervals.json", "w", encoding="utf-8") as f:
        json.dump(intervals_payload, f, indent=2)
        
    print("\n[OK] Volume model artifacts and dynamic quantiles saved successfully:")
    print(f"     - {ART_DIR / 'volume_model.pkl'} (Point Model)")
    print(f"     - {ART_DIR / 'volume_quantiles.pkl'} (Quantile Bundle: q10, q50, q90)")
    print(f"     - {ART_DIR / 'volume_intervals.json'} (Calibration Metadata)")

if __name__ == "__main__":
    train_volume_model()
