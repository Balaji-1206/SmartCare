import json
import sys

# Guard against environments where pyarrow C-extension is blocked by system security policies
if "pyarrow" not in sys.modules:
    try:
        import pyarrow  # noqa: F401
    except (ImportError, Exception):
        sys.modules["pyarrow"] = None

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

try:
    import mlflow
    HAS_MLFLOW = True
except (ImportError, Exception):
    HAS_MLFLOW = False

# Ensure root is in pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.utils import (
    add_calendar_features,
    add_lag_and_rolling_features,
    add_weather_features,
    calculate_wape,
    compute_residuals_intervals,
)

DATA_PATH = ROOT_DIR / "data" / "raw" / "data10yrs.csv"
ART_DIR = ROOT_DIR / "ml" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DB_URI = f"sqlite:///{ROOT_DIR.as_posix()}/mlflow.db"

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

def _create_regressor(loss: str = "regression", alpha: float = 0.5, n_estimators: int = 100, max_depth: int = 3):
    if HAS_LIGHTGBM:
        obj = "quantile" if loss == "quantile" else "regression"
        leaves = 15 if max_depth >= 4 else 7
        return lgb.LGBMRegressor(
            objective=obj,
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=leaves,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    else:
        loss_param = "quantile" if loss == "quantile" else "squared_error"
        return GradientBoostingRegressor(
            loss=loss_param,
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            random_state=42,
        )

def train_volume_model():
    engine_name = "LightGBM" if HAS_LIGHTGBM else "Scikit-Learn GradientBoosting"
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_volume_dataset(df)
    
    print(f"\n=======================================================")
    print(f"  Training Volume Models using [{engine_name}]")
    print(f"  Total Samples: {len(X)} records | Features: {len(FEATURE_COLS)}")
    print(f"=======================================================")

    # 1. 5-Fold Expanding Window TimeSeriesSplit Cross-Validation
    print("\n--- [MLOps] 5-Fold Expanding Window TimeSeriesSplit ---")
    tscv = TimeSeriesSplit(n_splits=5, test_size=180)
    cv_maes = []
    cv_wapes = []
    cv_r2s = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_fold_tr, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_tr, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_model = _create_regressor(loss="regression", n_estimators=150, max_depth=4)
        fold_model.fit(X_fold_tr, y_fold_tr)
        val_preds = fold_model.predict(X_fold_val)
        
        f_mae = mean_absolute_error(y_fold_val, val_preds)
        f_wape = calculate_wape(y_fold_val.values, val_preds)
        f_r2 = r2_score(y_fold_val, val_preds)
        
        cv_maes.append(f_mae)
        cv_wapes.append(f_wape)
        cv_r2s.append(f_r2)
        print(f"  Fold {fold}/5: Train={len(train_idx)}, Val={len(val_idx)} | MAE: {f_mae:.2f} | WAPE: {f_wape * 100:.1f}% | R2: {f_r2:.3f}")
        
    mean_cv_mae = float(np.mean(cv_maes))
    std_cv_mae = float(np.std(cv_maes))
    mean_cv_wape = float(np.mean(cv_wapes))
    mean_cv_r2 = float(np.mean(cv_r2s))
    print(f"  -> 5-Fold CV Mean MAE: {mean_cv_mae:.2f} ± {std_cv_mae:.2f} visits | Mean WAPE: {mean_cv_wape * 100:.1f}% | Mean R2: {mean_cv_r2:.3f}")

    # 2. Holdout Test Set Evaluation (last 365 days)
    split_idx = len(X) - 365
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n--- Holdout Test Set Evaluation (Last 365 Days) ---")
    point_model = _create_regressor(loss="regression", n_estimators=150, max_depth=4)
    point_model.fit(X_train, y_train)
    preds_test = point_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    test_wape = calculate_wape(y_test.values, preds_test)
    test_r2 = r2_score(y_test, preds_test)
    print(f"  Point Model -> Test MAE: {test_mae:.2f} visits | WAPE: {test_wape * 100:.1f}% | R2: {test_r2:.3f}")
    
    # Quantile Models (alpha=0.10, 0.90)
    q10_model = _create_regressor(loss="quantile", alpha=0.10, n_estimators=100, max_depth=3)
    q10_model.fit(X_train, y_train)
    p10_test = q10_model.predict(X_test)
    
    q90_model = _create_regressor(loss="quantile", alpha=0.90, n_estimators=100, max_depth=3)
    q90_model.fit(X_train, y_train)
    p90_test = q90_model.predict(X_test)
    
    # Quantile Calibration Evaluation
    coverage = float(np.mean((y_test.values >= p10_test) & (y_test.values <= p90_test)))
    widths = p90_test - p10_test
    mean_width = float(np.mean(widths))
    min_width = float(np.min(widths))
    max_width = float(np.max(widths))
    
    print(f"\n----------------- {engine_name} Calibration -----------------")
    print(f"  Target Coverage:       80.0%")
    print(f"  Empirical Coverage:    {coverage * 100:.1f}%")
    print(f"  Mean Interval Width:   {mean_width:.1f} visits")
    print(f"  Min Width (calm days): {min_width:.1f} visits")
    print(f"  Max Width (surge days):{max_width:.1f} visits")
    print("-----------------------------------------------------------------")
    
    # 3. Fit full models on entire dataset for production serving
    print(f"\nFitting full production {engine_name} models on entire dataset ({len(X)} records)...")
    full_point = _create_regressor(loss="regression", n_estimators=150, max_depth=4)
    full_point.fit(X, y)
    
    full_q10 = _create_regressor(loss="quantile", alpha=0.10, n_estimators=100, max_depth=3)
    full_q10.fit(X, y)
    
    full_q90 = _create_regressor(loss="quantile", alpha=0.90, n_estimators=100, max_depth=3)
    full_q90.fit(X, y)
    
    full_preds = full_point.predict(X)
    intervals_fallback = compute_residuals_intervals(y.values, full_preds)
    intervals_payload = {
        **intervals_fallback,
        "coverage_80": round(coverage, 3),
        "mean_interval_width": round(mean_width, 1),
        "min_interval_width": round(min_width, 1),
        "max_interval_width": round(max_width, 1),
        "engine": "lightgbm" if HAS_LIGHTGBM else "gradient_boosting",
        "model_type": "multi_quantile_lightgbm" if HAS_LIGHTGBM else "multi_quantile_gbr",
        "cv_mean_mae": round(mean_cv_mae, 2),
        "cv_mean_wape": round(mean_cv_wape, 3),
        "cv_mean_r2": round(mean_cv_r2, 3),
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
        
    print("\n[OK] Volume model artifacts and dynamic quantiles saved successfully.")

    # 4. MLflow Experiment Logging
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_DB_URI)
            mlflow.set_experiment("SmartCare_Volume_Forecasting")
            with mlflow.start_run(run_name="volume_lightgbm_multiquantile"):
                mlflow.log_params({
                    "engine": engine_name,
                    "n_estimators_point": 150,
                    "max_depth_point": 4,
                    "num_leaves_point": 15,
                    "n_estimators_quant": 100,
                    "max_depth_quant": 3,
                    "learning_rate": 0.05,
                    "quantiles": "0.10,0.50,0.90",
                    "cv_splits": 5,
                    "cv_test_size": 180,
                    "features_count": len(FEATURE_COLS),
                })
                for k in range(5):
                    mlflow.log_metric(f"cv_fold_{k+1}_mae", round(cv_maes[k], 3))
                    mlflow.log_metric(f"cv_fold_{k+1}_wape", round(cv_wapes[k], 4))
                    mlflow.log_metric(f"cv_fold_{k+1}_r2", round(cv_r2s[k], 4))
                mlflow.log_metric("cv_mean_mae", round(mean_cv_mae, 3))
                mlflow.log_metric("cv_std_mae", round(std_cv_mae, 3))
                mlflow.log_metric("cv_mean_wape", round(mean_cv_wape, 4))
                mlflow.log_metric("cv_mean_r2", round(mean_cv_r2, 4))
                mlflow.log_metric("test_mae", round(test_mae, 3))
                mlflow.log_metric("test_wape", round(test_wape, 4))
                mlflow.log_metric("test_r2", round(test_r2, 4))
                mlflow.log_metric("coverage_80", round(coverage, 3))
                mlflow.log_metric("mean_interval_width", round(mean_width, 2))
                mlflow.log_metric("min_interval_width", round(min_width, 2))
                mlflow.log_metric("max_interval_width", round(max_width, 2))
                mlflow.log_artifact(str(ART_DIR / "volume_intervals.json"))
                mlflow.log_artifact(str(ART_DIR / "volume_features.json"))
                mlflow.set_tag("pipeline", "patient_volume")
                mlflow.set_tag("model_type", "lightgbm_multiquantile")
            print(f"[MLOps] Successfully logged volume run to MLflow experiment 'SmartCare_Volume_Forecasting' ({MLFLOW_DB_URI})")
        except Exception as e:
            print(f"[MLOps Warning] MLflow tracking encountered error: {e}")

if __name__ == "__main__":
    train_volume_model()
