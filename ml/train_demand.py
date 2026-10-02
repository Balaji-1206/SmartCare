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
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
    add_clinical_cross_features,
    add_lag_and_rolling_features,
    add_weather_features,
    calculate_wape,
    compute_residuals_intervals,
)

DATA_PATH = ROOT_DIR / "data" / "raw" / "data10yrs.csv"
DEMAND_ART_DIR = ROOT_DIR / "ml" / "artifacts" / "demand"
MLFLOW_DB_URI = f"sqlite:///{ROOT_DIR.as_posix()}/mlflow.db"

ITEMS = ["paracetamol", "ors_packets", "malaria_kits", "antibiotics"]

CLINICAL_SYMPTOMS = ["total_patients", "fever_cases", "cough_cases", "diarrhea_cases", "vomiting_cases"]
CLINICAL_FEATURE_COLS = [
    f"{sym}_{suffix}"
    for sym in CLINICAL_SYMPTOMS
    for suffix in ["lag_1", "lag_7", "roll_mean_7"]
]

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
] + CLINICAL_FEATURE_COLS

def build_demand_dataset(df: pd.DataFrame, item_code: str) -> Tuple[pd.DataFrame, pd.Series]:
    col_name = f"{item_code}_used"
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in CSV.")
        
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, col_name, lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])
    d = add_clinical_cross_features(d, clinical_cols=CLINICAL_SYMPTOMS)
    
    lag_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=lag_cols + [col_name]).reset_index(drop=True)
    
    X = d[FEATURE_COLS].fillna(0)
    y = d[col_name].astype(float)
    return X, y

def _create_demand_model():
    if HAS_LIGHTGBM:
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    else:
        return GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

def train_demand_models():
    engine_name = "LightGBM" if HAS_LIGHTGBM else "Scikit-Learn GradientBoosting"
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Training Pharmaceutical Demand Models using [{engine_name}]...")
    
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_DB_URI)
            mlflow.set_experiment("SmartCare_Demand_Forecasting")
        except Exception as e:
            print(f"[MLOps Warning] Failed to initialize MLflow: {e}")
            
    for item in ITEMS:
        print(f"\n=======================================================")
        print(f"  Training Demand Model for '{item}'")
        print(f"=======================================================")
        X, y = build_demand_dataset(df, item)
        
        # 1. 5-Fold Expanding Window TimeSeriesSplit
        print(f"--- [MLOps] 5-Fold Expanding Window TimeSeriesSplit for '{item}' ---")
        tscv = TimeSeriesSplit(n_splits=5, test_size=180)
        cv_maes = []
        cv_wapes = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_fold_tr, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_tr, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = _create_demand_model()
            fold_model.fit(X_fold_tr, y_fold_tr)
            val_preds = fold_model.predict(X_fold_val)
            
            f_mae = mean_absolute_error(y_fold_val, val_preds)
            f_wape = calculate_wape(y_fold_val.values, val_preds)
            cv_maes.append(f_mae)
            cv_wapes.append(f_wape)
            print(f"  Fold {fold}/5: Train={len(train_idx)}, Val={len(val_idx)} | MAE: {f_mae:.2f} units | WAPE: {f_wape * 100:.1f}%")
            
        mean_cv_mae = float(np.mean(cv_maes))
        std_cv_mae = float(np.std(cv_maes))
        mean_cv_wape = float(np.mean(cv_wapes))
        print(f"  -> 5-Fold CV Mean MAE: {mean_cv_mae:.2f} ± {std_cv_mae:.2f} units | Mean WAPE: {mean_cv_wape * 100:.1f}%")
        
        # 2. Holdout Test Set Evaluation (last 365 days)
        split_idx = len(X) - 365
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = _create_demand_model()
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, preds_test)
        test_wape = calculate_wape(y_test.values, preds_test)
        print(f"  Holdout Test -> MAE: {test_mae:.2f} units | WAPE: {test_wape * 100:.1f}%")
        
        # 3. Fit full model on entire dataset
        full_model = _create_demand_model()
        full_model.fit(X, y)
        full_preds = full_model.predict(X)
        intervals = compute_residuals_intervals(y.values, full_preds)
        intervals["engine"] = "lightgbm" if HAS_LIGHTGBM else "gradient_boosting"
        intervals["cv_mean_mae"] = round(mean_cv_mae, 2)
        intervals["cv_mean_wape"] = round(mean_cv_wape, 3)
        intervals["test_mae"] = round(test_mae, 2)
        intervals["test_wape"] = round(test_wape, 3)
        
        item_dir = DEMAND_ART_DIR / item
        item_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_model, item_dir / "model.pkl")
        with open(item_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_COLS}, f, indent=2)
        with open(item_dir / "intervals.json", "w", encoding="utf-8") as f:
            json.dump(intervals, f, indent=2)
            
        print(f"[OK] Saved {engine_name} artifacts for '{item}' in {item_dir}")
        
        # 4. MLflow Experiment Logging
        if HAS_MLFLOW:
            try:
                with mlflow.start_run(run_name=f"demand_{item}_lightgbm"):
                    mlflow.log_params({
                        "item": item,
                        "engine": engine_name,
                        "clinical_cross_features": True,
                        "features_count": len(FEATURE_COLS),
                        "n_estimators": 100,
                        "max_depth": 3,
                        "learning_rate": 0.05,
                        "cv_splits": 5,
                        "cv_test_size": 180,
                    })
                    for k in range(5):
                        mlflow.log_metric(f"cv_fold_{k+1}_mae", round(cv_maes[k], 3))
                        mlflow.log_metric(f"cv_fold_{k+1}_wape", round(cv_wapes[k], 4))
                    mlflow.log_metric("cv_mean_mae", round(mean_cv_mae, 3))
                    mlflow.log_metric("cv_std_mae", round(std_cv_mae, 3))
                    mlflow.log_metric("cv_mean_wape", round(mean_cv_wape, 4))
                    mlflow.log_metric("test_mae", round(test_mae, 3))
                    mlflow.log_metric("test_wape", round(test_wape, 4))
                    mlflow.log_artifact(str(item_dir / "features.json"))
                    mlflow.log_artifact(str(item_dir / "intervals.json"))
                    mlflow.set_tag("pipeline", "medicine_demand")
                    mlflow.set_tag("item", item)
                print(f"[MLOps] Logged '{item}' run to MLflow experiment 'SmartCare_Demand_Forecasting'")
            except Exception as e:
                print(f"[MLOps Warning] Failed to log MLflow run for '{item}': {e}")

if __name__ == "__main__":
    train_demand_models()

