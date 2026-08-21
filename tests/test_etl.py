import pandas as pd
import pytest
from pathlib import Path

DATA_CSV = Path("data/raw/data10yrs.csv")

def test_raw_csv_exists_and_valid():
    assert DATA_CSV.exists(), "data/raw/data10yrs.csv must exist"
    df = pd.read_csv(DATA_CSV)
    assert len(df) >= 3000, "Dataset should have at least 3000 daily records (~10 years)"
    
    required_cols = [
        "date",
        "day_of_week",
        "total_patients",
        "temperature",
        "rainfall",
        "humidity",
        "paracetamol_used",
        "fever_cases"
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from data10yrs.csv"

def test_date_integrity():
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing or df.sort_values("date")["date"].is_monotonic_increasing
    assert not df["date"].duplicated().any(), "Dates should not contain duplicates"

def test_non_negative_values():
    df = pd.read_csv(DATA_CSV)
    numeric_cols = ["total_patients", "paracetamol_used", "antibiotics_used", "fever_cases", "cough_cases"]
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (series >= 0).all(), f"Values in '{col}' must be non-negative"
