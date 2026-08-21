from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import json

from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features

ART_ROOT = Path("ml/artifacts/demand")

def list_available_items() -> List[str]:
    """Return item_code folder names found under ml/artifacts/demand/"""
    if not ART_ROOT.exists():
        return []
    return sorted([p.name for p in ART_ROOT.iterdir() if p.is_dir() and (p / "model.pkl").exists()])

def _build_features_for_item(hist_df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """
    Build leak-free temporal features for a single medicine demand series.
    """
    if item_col not in hist_df.columns:
        raise FileNotFoundError(f"Column '{item_col}' not found in history dataframe.")

    d = hist_df.sort_values("date").copy()
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"])
        
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, item_col, lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])

    lag_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=lag_cols).reset_index(drop=True)

    X = d.drop(columns=["date"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def _load_item_artifacts(item_code: str):
    base = ART_ROOT / item_code
    model_path = base / "model.pkl"
    feats_path = base / "features.json"
    intr_path  = base / "intervals.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for item '{item_code}' at {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"features.json not found for item '{item_code}'")
    if not intr_path.exists():
        raise FileNotFoundError(f"intervals.json not found for item '{item_code}'")

    model = joblib.load(model_path)
    features = json.load(open(feats_path, "r", encoding="utf-8")).get("features", [])
    intervals = json.load(open(intr_path, "r", encoding="utf-8"))
    return model, features, intervals

def predict_one_item(hist_df: pd.DataFrame, item_code: str) -> Dict:
    """
    Predict demand for a single item_code.
    Returns dict {item_code, yhat, p10, p90}.
    """
    csv_col = f"{item_code}_used"
    X_all = _build_features_for_item(hist_df, csv_col)
    model, feat_list, intervals = _load_item_artifacts(item_code)

    for col in feat_list:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feat_list]

    if X_all.empty:
        raise ValueError(f"Not enough history to predict for '{item_code}'.")

    x = X_all.iloc[[-1]]
    yhat = max(0.0, float(model.predict(x)[0]))

    p10 = max(0.0, yhat + float(intervals.get("residual_p10", -1.0)))
    p90 = max(0.0, yhat + float(intervals.get("residual_p90", 1.0)))

    return {"item_code": item_code, "yhat": yhat, "p10": p10, "p90": p90}
