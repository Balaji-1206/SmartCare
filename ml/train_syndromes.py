import json
import sys
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Ensure root is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features

DATA_PATH = Path("data/raw/data10yrs.csv")
SYNDROMES_ART_DIR = Path("ml/artifacts/syndromes")

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

def train_syndrome_models():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    for syn in SYNDROMES:
        print(f"\n--- Training Syndrome Model for '{syn}' ---")
        X, y, threshold = build_syndrome_dataset(df, syn)
        
        split_idx = len(X) - 365
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        
        probs_test = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs_test)
        except Exception:
            auc = 0.5
        print(f"[{syn}] Outbreak Threshold: >= {threshold:.1f} cases | Test ROC-AUC: {auc:.3f}")
        
        # Fit full model
        full_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        full_model.fit(X, y)
        
        syn_dir = SYNDROMES_ART_DIR / syn
        syn_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_model, syn_dir / "model.pkl")
        with open(syn_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_COLS}, f, indent=2)
        with open(syn_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold, "task": "outbreak_probability"}, f, indent=2)
            
        print(f"[OK] Saved artifacts for '{syn}' in {syn_dir}")

if __name__ == "__main__":
    train_syndrome_models()
