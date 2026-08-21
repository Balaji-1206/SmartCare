import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.utils import (
    add_calendar_features,
    add_lag_and_rolling_features,
    add_weather_features,
    compute_residuals_intervals,
)
from api.services.demand import list_available_items, predict_one_item
from api.services.syndromes import list_available_syndromes, predict_one_syn

ART_DIR = Path("ml/artifacts")
DATA_CSV = Path("data/raw/data10yrs.csv")

def test_add_calendar_features():
    df = pd.DataFrame({"date": ["2026-08-21", "2026-08-22", "2026-08-23"]})
    res = add_calendar_features(df, "date")
    assert "dow" in res.columns
    assert "month" in res.columns
    assert "is_weekend" in res.columns
    # 2026-08-21 is Friday (dow=4, is_weekend=0)
    assert res.loc[0, "dow"] == 4
    assert res.loc[0, "is_weekend"] == 0
    # 2026-08-22 is Saturday (dow=5, is_weekend=1)
    assert res.loc[1, "is_weekend"] == 1

def test_add_lag_and_rolling_features():
    df = pd.DataFrame({"val": list(range(35))})
    res = add_lag_and_rolling_features(df, "val", lags=[1, 7], roll_windows=[7])
    assert "lag_1" in res.columns
    assert "lag_7" in res.columns
    assert "roll_mean_7" in res.columns
    assert "roll_std_7" in res.columns
    assert res.loc[10, "lag_1"] == 9
    assert res.loc[10, "lag_7"] == 3

def test_volume_model_artifacts_exist_and_leak_free():
    vol_model_path = ART_DIR / "volume_model.pkl"
    vol_feats_path = ART_DIR / "volume_features.json"
    vol_intv_path = ART_DIR / "volume_intervals.json"
    
    assert vol_model_path.exists(), "volume_model.pkl should exist"
    assert vol_feats_path.exists(), "volume_features.json should exist"
    assert vol_intv_path.exists(), "volume_intervals.json should exist"
    
    with open(vol_feats_path, "r") as f:
        feats_data = json.load(f)
    features = feats_data["features"]
    
    # Assert leak-free: no same-day demographic sums or same-day case counts
    leaked_cols = ["male_patients", "female_patients", "children_patients", "adult_patients", "elderly_patients", "fever_cases", "cough_cases"]
    for col in leaked_cols:
        assert col not in features, f"Leaked column '{col}' must not be in volume model features"

def test_syndromes_artifacts_leak_free():
    syns = list_available_syndromes()
    assert len(syns) >= 6, "Should have 6 syndromes available"
    
    for syn in syns:
        feats_path = ART_DIR / "syndromes" / syn / "features.json"
        assert feats_path.exists()
        with open(feats_path, "r") as f:
            data = json.load(f)
        features = data["features"]
        assert "y_count" not in features, f"Syndrome '{syn}' features must not include 'y_count' target leakage"

def test_demand_inference():
    df = pd.read_csv(DATA_CSV)
    items = list_available_items()
    assert len(items) >= 4, "Should have 4 demand items"
    
    for item in items:
        res = predict_one_item(df, item)
        assert "yhat" in res
        assert "p10" in res
        assert "p90" in res
        assert res["yhat"] >= 0, f"Predicted demand for {item} must be >= 0"
        assert res["p90"] >= res["p10"], "p90 interval must be >= p10"

def test_syndromes_inference():
    df = pd.read_csv(DATA_CSV)
    syns = list_available_syndromes()
    for syn in syns:
        res = predict_one_syn(df, syn)
        assert "syndrome" in res
        assert "prob" in res
        assert 0.0 <= res["prob"] <= 1.0, f"Probability for {syn} must be between 0 and 1"
