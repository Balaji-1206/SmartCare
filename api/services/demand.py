from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib, json

ART_ROOT = Path("ml/artifacts/demand")

def list_available_items() -> List[str]:
    """Return item_code folder names found under ml/artifacts/demand/"""
    if not ART_ROOT.exists():
        return []
    return sorted([p.name for p in ART_ROOT.iterdir() if p.is_dir()])

def _build_features_for_item(hist_df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """
    Build the same kind of features you trained in Colab for a single demand series.
    hist_df must have a datetime 'date' column and optional weather columns.
    item_col is the CSV column name like 'paracetamol_used'.
    """
    if item_col not in hist_df.columns:
        raise FileNotFoundError(f"Column '{item_col}' not found in history CSV.")

    d = pd.DataFrame({"y": pd.to_numeric(hist_df[item_col], errors="coerce")})
    # Lags
    for lag in [1, 7, 14, 28]:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    # Rolling stats
    for w in [7, 14, 28]:
        d[f"roll_mean_{w}"] = d["y"].rolling(w).mean()
        d[f"roll_std_{w}"]  = d["y"].rolling(w).std()

    # Calendar + optional weather copied from hist_df
    d["date"] = hist_df["date"].values
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    for col in ["temperature", "rainfall", "humidity"]:
        if col in hist_df.columns:
            d[col] = pd.to_numeric(hist_df[col], errors="coerce")

    # Keep only rows that have full lag/rolling context
    need = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]
    d = d.dropna(subset=need)

    # Final X matrix
    X = d.drop(columns=["y", "date"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def _load_item_artifacts(item_code: str):
    """
    Load artifacts for an item_code folder:
    ml/artifacts/demand/<item_code>/{model.pkl, features.json, intervals.json}
    """
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
    features = json.load(open(feats_path)).get("features", [])
    intervals = json.load(open(intr_path))
    return model, features, intervals

def predict_one_item(hist_df: pd.DataFrame, item_code: str) -> Dict:
    """
    Predict demand for a single item_code.
    - hist_df: DataFrame with 'date' and '<item_code>_used' column.
    - item_code: folder name under artifacts (e.g., 'paracetamol', 'ors_packets').
    Returns dict {item_code, yhat, p10, p90}.
    """
    # Build features from history
    csv_col = f"{item_code}_used"
    X_all = _build_features_for_item(hist_df, csv_col)

    # Load artifacts and align feature columns in the same order as training
    model, feat_list, intervals = _load_item_artifacts(item_code)

    # Add any missing columns as 0, then reorder
    for col in feat_list:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feat_list]

    if X_all.empty:
        raise ValueError(f"Not enough history to predict for '{item_code}'.")

    x = X_all.iloc[[-1]]
    yhat = float(model.predict(x)[0])

    p10 = yhat + float(intervals.get("residual_p10", -1.0))
    p90 = yhat + float(intervals.get("residual_p90",  1.0))

    return {"item_code": item_code, "yhat": yhat, "p10": p10, "p90": p90}
