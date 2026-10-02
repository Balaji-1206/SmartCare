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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import mlflow
    HAS_MLFLOW = True
except (ImportError, Exception):
    HAS_MLFLOW = False

# Ensure root is in pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features

DATA_PATH = ROOT_DIR / "data" / "raw" / "data10yrs.csv"
SYNDROMES_ART_DIR = ROOT_DIR / "ml" / "artifacts" / "syndromes"
MLFLOW_DB_URI = f"sqlite:///{ROOT_DIR.as_posix()}/mlflow.db"

SYNDROMES = ["animal_bite", "cough", "diarrhea", "fever", "skin_rash", "vomiting"]

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
    "tp_lag_1",
    "tp_lag_7",
    "tp_mean_7",
    "dow",
    "month",
    "is_weekend",
    "temperature",
    "rainfall",
    "humidity",
]

def build_syndrome_dataset(df: pd.DataFrame, syn_code: str) -> Tuple[pd.DataFrame, pd.Series, float]:
    col_name = f"{syn_code}_cases"
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in CSV.")
        
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    
    # Target counts and lags
    d = add_lag_and_rolling_features(d, col_name, lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])
    
    # Total patient volume lags
    if "total_patients" in d.columns:
        tp = pd.to_numeric(d["total_patients"], errors="coerce")
        d["tp_lag_1"] = tp.shift(1)
        d["tp_lag_7"] = tp.shift(7)
        d["tp_mean_7"] = tp.rolling(7).mean()
    else:
        d["tp_lag_1"] = 0
        d["tp_lag_7"] = 0
        d["tp_mean_7"] = 0
        
    lag_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=lag_cols + [col_name]).reset_index(drop=True)
    
    # Define outbreak target: higher than 75th percentile / threshold
    counts = pd.to_numeric(d[col_name], errors="coerce").fillna(0)
    p75 = float(np.percentile(counts, 75))
    threshold = max(1.0, p75)
    y = (counts >= threshold).astype(int)
    
    # Ensure both classes exist
    if y.nunique() < 2:
        threshold = max(1.0, float(np.median(counts)))
        y = (counts >= threshold).astype(int)
        
    X = d[FEATURE_COLS].fillna(0)
    return X, y, threshold

def _create_classifier():
    if HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

def train_syndrome_models():
    engine_name = "LightGBM" if HAS_LIGHTGBM else "Scikit-Learn GradientBoosting"
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Training Syndromic Outbreak Classifiers using [{engine_name}]...")
    
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_DB_URI)
            mlflow.set_experiment("SmartCare_Syndrome_Surveillance")
        except Exception as e:
            print(f"[MLOps Warning] Failed to initialize MLflow: {e}")
            
    for syn in SYNDROMES:
        print(f"\n=======================================================")
        print(f"  Training Outbreak Model for '{syn}'")
        print(f"=======================================================")
        X, y, threshold = build_syndrome_dataset(df, syn)
        
        # 1. 5-Fold Expanding Window TimeSeriesSplit
        print(f"--- [MLOps] 5-Fold Expanding Window TimeSeriesSplit for '{syn}' ---")
        tscv = TimeSeriesSplit(n_splits=5, test_size=180)
        cv_aucs = []
        cv_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_fold_tr, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_tr, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = _create_classifier()
            fold_model.fit(X_fold_tr, y_fold_tr)
            val_probs = fold_model.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            try:
                f_auc = roc_auc_score(y_fold_val, val_probs) if len(np.unique(y_fold_val)) > 1 else 0.5
            except Exception:
                f_auc = 0.5
            f_acc = accuracy_score(y_fold_val, val_preds)
            
            cv_aucs.append(f_auc)
            cv_accs.append(f_acc)
            print(f"  Fold {fold}/5: Train={len(train_idx)}, Val={len(val_idx)} | ROC-AUC: {f_auc:.3f} | Accuracy: {f_acc * 100:.1f}%")
            
        mean_cv_auc = float(np.mean(cv_aucs))
        std_cv_auc = float(np.std(cv_aucs))
        mean_cv_acc = float(np.mean(cv_accs))
        print(f"  -> 5-Fold CV Mean ROC-AUC: {mean_cv_auc:.3f} ± {std_cv_auc:.3f} | Mean Accuracy: {mean_cv_acc * 100:.1f}%")
        
        # 2. Holdout Test Set Evaluation (last 365 days)
        split_idx = len(X) - 365
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = _create_classifier()
        model.fit(X_train, y_train)
        probs_test = model.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= 0.5).astype(int)
        try:
            test_auc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else 0.5
        except Exception:
            test_auc = 0.5
        test_acc = accuracy_score(y_test, preds_test)
        print(f"  Holdout Test -> Threshold: >= {threshold:.1f} cases | ROC-AUC: {test_auc:.3f} | Accuracy: {test_acc * 100:.1f}%")
        
        # 3. Fit full model on entire dataset
        full_model = _create_classifier()
        full_model.fit(X, y)
        
        syn_dir = SYNDROMES_ART_DIR / syn
        syn_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_model, syn_dir / "model.pkl")
        with open(syn_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_COLS}, f, indent=2)
        with open(syn_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "syndrome": syn,
                "threshold": threshold,
                "engine": "lightgbm" if HAS_LIGHTGBM else "gradient_boosting",
                "test_roc_auc": round(float(test_auc), 3),
                "test_accuracy": round(float(test_acc), 3),
                "cv_mean_roc_auc": round(mean_cv_auc, 3),
                "cv_mean_accuracy": round(mean_cv_acc, 3),
            }, f, indent=2)
            
        print(f"[OK] Saved {engine_name} artifacts for '{syn}' in {syn_dir}")
        
        # 4. MLflow Experiment Logging
        if HAS_MLFLOW:
            try:
                with mlflow.start_run(run_name=f"syndrome_{syn}_lightgbm"):
                    mlflow.log_params({
                        "syndrome": syn,
                        "engine": engine_name,
                        "threshold": threshold,
                        "features_count": len(FEATURE_COLS),
                        "n_estimators": 100,
                        "max_depth": 3,
                        "learning_rate": 0.05,
                        "cv_splits": 5,
                        "cv_test_size": 180,
                    })
                    for k in range(5):
                        mlflow.log_metric(f"cv_fold_{k+1}_auc", round(cv_aucs[k], 4))
                        mlflow.log_metric(f"cv_fold_{k+1}_acc", round(cv_accs[k], 4))
                    mlflow.log_metric("cv_mean_auc", round(mean_cv_auc, 4))
                    mlflow.log_metric("cv_std_auc", round(std_cv_auc, 4))
                    mlflow.log_metric("cv_mean_acc", round(mean_cv_acc, 4))
                    mlflow.log_metric("test_roc_auc", round(test_auc, 4))
                    mlflow.log_metric("test_accuracy", round(test_acc, 4))
                    mlflow.log_artifact(str(syn_dir / "features.json"))
                    mlflow.log_artifact(str(syn_dir / "meta.json"))
                    mlflow.set_tag("pipeline", "syndrome_surveillance")
                    mlflow.set_tag("syndrome", syn)
                print(f"[MLOps] Logged '{syn}' run to MLflow experiment 'SmartCare_Syndrome_Surveillance'")
            except Exception as e:
                print(f"[MLOps Warning] Failed to log MLflow run for '{syn}': {e}")

if __name__ == "__main__":
    train_syndrome_models()

