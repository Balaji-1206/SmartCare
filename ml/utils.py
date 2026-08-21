import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Adds day-of-week, month, is_weekend features based on datetime date column."""
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col])
    d["dow"] = d[date_col].dt.dayofweek
    d["month"] = d[date_col].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    return d

def add_lag_and_rolling_features(
    df: pd.DataFrame,
    col: str,
    lags: List[int] = [1, 7, 14, 28],
    roll_windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """Adds temporal lags and rolling statistics for a target column."""
    d = df.copy()
    series = pd.to_numeric(d[col], errors="coerce")
    for lag in lags:
        d[f"lag_{lag}"] = series.shift(lag)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = series.rolling(w).mean()
        d[f"roll_std_{w}"] = series.rolling(w).std()
    return d

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures weather features are numeric and cleaned."""
    d = df.copy()
    for col in ["temperature", "rainfall", "humidity"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d

def compute_residuals_intervals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes empirical 10th and 90th percentile prediction intervals from residuals."""
    residuals = y_true - y_pred
    p10 = float(np.percentile(residuals, 10))
    p90 = float(np.percentile(residuals, 90))
    return {"residual_p10": p10, "residual_p90": p90}
