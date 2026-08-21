from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import json

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features

ART_ROOT = Path("ml/artifacts/syndromes")

def list_available_syndromes() -> List[str]:
    if not ART_ROOT.exists():
        return []
    return sorted([p.name for p in ART_ROOT.iterdir() if p.is_dir() and (p / "model.pkl").exists()])

def _build_features_for_syn(hist_df: pd.DataFrame, syn_col: str) -> pd.DataFrame:
    """
    Builds temporal lag and rolling features for syndrome classification without same-day target leakage.
    """
    d = hist_df.sort_values("date").copy()
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"])
        
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, syn_col, lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])

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
    d = d.dropna(subset=lag_cols).reset_index(drop=True)

    X = d.drop(columns=["date"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def _load_syn_artifacts(syn_code: str):
    base = ART_ROOT / syn_code
    model = joblib.load(base / "model.pkl")
    feats = json.load(open(base / "features.json", "r", encoding="utf-8"))
    meta  = json.load(open(base / "meta.json", "r", encoding="utf-8"))
    feat_list = feats.get("features", [])
    threshold = float(meta.get("threshold", 1.0))
    return model, feat_list, threshold

def predict_one_syn(hist_df: pd.DataFrame, syn_code: str) -> Dict:
    """
    Returns: {"syndrome": syn_code, "prob": float}
    """
    csv_col = f"{syn_code}_cases"
    if csv_col not in hist_df.columns:
        raise FileNotFoundError(f"Column '{csv_col}' not found in history data.")
        
    model, feat_list, thr = _load_syn_artifacts(syn_code)
    X_all = _build_features_for_syn(hist_df, csv_col)

    for col in feat_list:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feat_list]
    
    if X_all.empty:
        raise ValueError(f"Not enough history to predict '{syn_code}'.")

    x = X_all.iloc[[-1]]
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[:, 1][0])
    else:
        prob = float(model.predict(x)[0])
        
    return {"syndrome": syn_code, "prob": round(prob, 3)}
