# smartcare/api/services/syndromes.py
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib, json

ART_ROOT = Path("ml/artifacts/syndromes")

def list_available_syndromes() -> List[str]:
    if not ART_ROOT.exists():
        return []
    return sorted([p.name for p in ART_ROOT.iterdir() if p.is_dir()])

def _build_features_for_syn(hist_df: pd.DataFrame, syn_col: str, threshold: int = 1) -> pd.DataFrame:
    d = pd.DataFrame(index=hist_df.index)
    # numeric counts
    y_count = pd.to_numeric(hist_df[syn_col], errors="coerce").fillna(0)
    d["y_count"] = y_count

    # lags/rollings
    for lag in [1,7,14,28]:
        d[f"lag_{lag}"] = d["y_count"].shift(lag)
    for w in [7,14,28]:
        d[f"roll_mean_{w}"] = d["y_count"].rolling(w).mean()
        d[f"roll_std_{w}"]  = d["y_count"].rolling(w).std()

    # context
    d["date"] = hist_df["date"].values
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)

    if "total_patients" in hist_df.columns:
        tp = pd.to_numeric(hist_df["total_patients"], errors="coerce")
        d["tp_lag_1"]  = tp.shift(1)
        d["tp_lag_7"]  = tp.shift(7)
        d["tp_mean_7"] = tp.rolling(7).mean()

    for col in ["temperature","rainfall","humidity"]:
        if col in hist_df.columns:
            d[col] = pd.to_numeric(hist_df[col], errors="coerce")

    need = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=need)

    X = d.drop(columns=["date"], errors="ignore").replace([np.inf,-np.inf], np.nan).fillna(0)
    return X

def _load_syn_artifacts(syn_code: str):
    base = ART_ROOT / syn_code
    model = joblib.load(base / "model.pkl")
    feats = json.load(open(base / "features.json"))
    meta  = json.load(open(base / "meta.json"))
    feat_list = feats.get("features", [])
    threshold = float(meta.get("threshold", 0.5))
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
    # align to training feature order
    for col in feat_list:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feat_list]
    if X_all.empty:
        raise ValueError(f"Not enough history to predict '{syn_code}'.")

    x = X_all.iloc[[-1]]
    prob = float(model.predict_proba(x)[:,1][0])
    return {"syndrome": syn_code, "prob": prob}
