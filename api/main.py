from __future__ import annotations
import os
import json
import threading
import datetime as dt
from pathlib import Path
from typing import List, Optional, Dict, Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# =========================
# Timezone (IST for clinic)
# =========================
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
def _today_local_str() -> str:
    return dt.datetime.now(tz=LOCAL_TZ).date().strftime("%Y-%m-%d")

# =========================
# Robust .env loading
# =========================
from dotenv import load_dotenv
def _find_env_near_main() -> str | None:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        candidate = p / ".env"
        if candidate.exists():
            return str(candidate)
    return None

_ENV_PATH = _find_env_near_main()
load_dotenv(_ENV_PATH or None, override=True)

# Optional live weather requests
try:
    import requests
except Exception:
    requests = None

# =========================
# Services & ML Utilities
# =========================
from ml.utils import add_calendar_features, add_lag_and_rolling_features, add_weather_features
from .services.demand import predict_one_item, list_available_items
from .services.syndromes import list_available_syndromes, predict_one_syn

# =========================
# Paths & Artifacts
# =========================
ART_DIR = Path("ml/artifacts")
VOL_MODEL_PATH = ART_DIR / "volume_model.pkl"
VOL_FEATS_PATH = ART_DIR / "volume_features.json"
VOL_INTV_PATH  = ART_DIR / "volume_intervals.json"

DATA_CSV = Path("data/raw/data10yrs.csv")
WEATHER_OVERRIDES_JSON = Path("data/raw/weather_overrides.json") 
NURSE_LOG_JSON = Path("data/raw/nurse_log.json")                 
INVENTORY_JSON = Path("data/raw/inventory.json")

if not VOL_MODEL_PATH.exists() or not VOL_INTV_PATH.exists():
    raise RuntimeError("Volume artifacts missing. Run 'python -m ml.train_volume' first.")

if not DATA_CSV.exists():
    raise RuntimeError("data/raw/data10yrs.csv not found.")

vol_model = joblib.load(VOL_MODEL_PATH)
vol_intervals = json.load(open(VOL_INTV_PATH, "r", encoding="utf-8"))
vol_feat_list = json.load(open(VOL_FEATS_PATH, "r", encoding="utf-8")).get("features", []) if VOL_FEATS_PATH.exists() else None

# =========================
# Thread-safe JSON helper
# =========================
_file_lock = threading.Lock()

def _safe_load_json(path: Path) -> dict:
    if path.exists():
        with _file_lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}

def _safe_save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# =========================
# In-Memory History Caching & Dynamic Merge
# =========================
_cached_hist_base: Optional[pd.DataFrame] = None

def _load_hist_base(force_reload: bool = False) -> pd.DataFrame:
    global _cached_hist_base
    if _cached_hist_base is None or force_reload:
        df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        _cached_hist_base = df
    return _cached_hist_base.copy()

def _load_weather_overrides() -> dict:
    return _safe_load_json(WEATHER_OVERRIDES_JSON)

def _save_weather_override(date_str: str, temperature: float | None, rainfall: float | None, humidity: float | None):
    data = _load_weather_overrides()
    data[date_str] = {"temperature": temperature, "rainfall": rainfall, "humidity": humidity}
    _safe_save_json(WEATHER_OVERRIDES_JSON, data)

def _hist_with_weather_and_logs() -> pd.DataFrame:
    """
    Returns history dataframe merged with weather overrides and any recent nurse logs.
    """
    df = _load_hist_base()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # 1. Apply weather overrides
    overrides = _load_weather_overrides()
    if overrides:
        dfo = pd.DataFrame([
            {
                "date": pd.to_datetime(k).normalize(),
                "temperature": v.get("temperature"),
                "rainfall": v.get("rainfall"),
                "humidity": v.get("humidity")
            }
            for k, v in overrides.items()
        ]).sort_values("date")
        df = df.merge(dfo, on="date", how="left", suffixes=("", "_ovr"))
        for col in ["temperature", "rainfall", "humidity"]:
            ocol = f"{col}_ovr"
            if ocol in df.columns:
                df[col] = np.where(df[ocol].notna(), df[ocol], df[col])
                df.drop(columns=[ocol], inplace=True)

    # 2. Apply nurse logs (dynamic recent visits and symptom counts)
    nurse_logs = _load_nurse_log()
    if nurse_logs:
        log_rows = []
        for d_str, entry in nurse_logs.items():
            try:
                entry_date = pd.to_datetime(d_str).normalize()
            except Exception:
                continue
            # Calculate total symptom cases logged by nurse
            fever = entry.get("fever") or 0
            cough = entry.get("cough") or 0
            diarrhea = entry.get("diarrhea") or 0
            vomiting = entry.get("vomiting") or 0
            cold = entry.get("cold") or 0
            total_cases = fever + cough + diarrhea + vomiting + cold
            log_rows.append({
                "date": entry_date,
                "fever_cases": fever,
                "cough_cases": cough,
                "diarrhea_cases": diarrhea,
                "vomiting_cases": vomiting,
                "total_patients": total_cases if total_cases > 0 else np.nan,
            })
        if log_rows:
            dfl = pd.DataFrame(log_rows).sort_values("date")
            # Combine or append dates beyond CSV
            max_csv_date = df["date"].max()
            new_dates_df = dfl[dfl["date"] > max_csv_date]
            if not new_dates_df.empty:
                df = pd.concat([df, new_dates_df], ignore_index=True)

    return df.sort_values("date").reset_index(drop=True)

# =========================
# Feature builder for Volume
# =========================
def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"])
    d = add_calendar_features(d, "date")
    d = add_weather_features(d)
    d = add_lag_and_rolling_features(d, "total_patients", lags=[1, 7, 14, 28], roll_windows=[7, 14, 28])
    return d

def prep_X_from_features(feat_df: pd.DataFrame, training_features: Optional[List[str]] = None) -> pd.DataFrame:
    X = feat_df.copy()
    X = X.drop(columns=[c for c in ["date", "total_patients"] if c in X.columns], errors="ignore")
    # Align
    if training_features:
        for col in training_features:
            if col not in X.columns:
                X[col] = 0
        X = X[training_features]
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# Pydantic models
# =========================
class VolumeReq(BaseModel):
    facility_id: str = "C001"

class VolumeRes(BaseModel):
    predicted_visits: float
    p10: float | None = None
    p90: float | None = None
    model_version: str = "v1.0.0"
    for_date: str

class DemandReq(BaseModel):
    items: Optional[List[str]] = None

class DemandResItem(BaseModel):
    item_code: str
    yhat: float
    p10: float | None = None
    p90: float | None = None

class SyndromesReq(BaseModel):
    top_n: int = 3
    syndromes: Optional[List[str]] = None

class SyndromeResItem(BaseModel):
    syndrome: str
    prob: float
    rank: int

class WeatherUpsertReq(BaseModel):
    date: str  # "YYYY-MM-DD"
    temperature: Optional[float] = None
    rainfall: Optional[float] = None
    humidity: Optional[float] = None

class WeatherFetchReq(BaseModel):
    date: Optional[str] = None 
    lat: float
    lon: float
    units: str = "metric"
    provider: str = "openweather"

class NurseLogReq(BaseModel):
    date: Optional[str] = None             
    fever: Optional[int] = None
    cough: Optional[int] = None
    diarrhea: Optional[int] = None
    vomiting: Optional[int] = None
    cold: Optional[int] = None
    others: Optional[int] = None
    notes: Optional[str] = None
    by: Optional[str] = None

class InventoryUpsertReq(BaseModel):
    item_code: str
    name: Optional[str] = None
    on_hand: Optional[int] = None
    reorder_point: Optional[int] = None

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI(title="SmartCare API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _clean_num(x: float | None) -> float | None:
    if x is None:
        return None
    return round(max(0.0, float(x)), 2)

def compute_status_level(yhat: float) -> str:
    df = _hist_with_weather_and_logs()
    last90 = df.tail(90)["total_patients"].dropna().astype(float)
    if len(last90) < 10:
        return "GREEN" if yhat < 50 else ("YELLOW" if yhat < 80 else "RED")
    p60 = np.percentile(last90, 60)
    p85 = np.percentile(last90, 85)
    if yhat <= p60:
        return "GREEN"
    if yhat <= p85:
        return "YELLOW"
    return "RED"

# =========================
# Root & Health
# =========================
@app.get("/")
def root():
    return JSONResponse({
        "app": "SmartCare API",
        "status": "ok",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/predict/volume (POST)",
            "/predict/demand (POST)",
            "/predict/syndromes (POST)",
            "/mobile/today (GET)",
            "/nurse/log (POST)",
            "/nurse/log/{date} (GET)",
            "/inventory (GET)",
            "/inventory/upsert (POST)",
            "/weather/upsert (POST)",
            "/weather/today (GET)",
            "/weather/fetch (POST)",
            "/alerts (GET)",
            "/debug/status-thresholds (GET)"
        ]
    })

# =========================
# Predictions
# =========================
@app.post("/predict/volume", response_model=VolumeRes)
def predict_volume(req: VolumeReq):
    df = _hist_with_weather_and_logs()
    feats = build_volume_features(df)
    need = [c for c in feats.columns if c.startswith("lag_") or c.startswith("roll_")]
    feats_clean = feats.dropna(subset=need)
    if feats_clean.empty:
        raise HTTPException(400, "Not enough history to form features.")
    
    X_all = prep_X_from_features(feats_clean, vol_feat_list)
    x = X_all.iloc[[-1]]

    yhat = max(0.0, float(vol_model.predict(x)[0]))
    p10_res = float(vol_intervals.get("residual_p10", -5.0))
    p90_res = float(vol_intervals.get("residual_p90",  5.0))
    p10 = max(0.0, yhat + p10_res)
    p90 = max(0.0, yhat + p90_res)

    return VolumeRes(
        predicted_visits=_clean_num(yhat) or 0.0,
        p10=_clean_num(p10),
        p90=_clean_num(p90),
        model_version="v1.0.0",
        for_date=_today_local_str(),
    )

@app.post("/predict/demand", response_model=List[DemandResItem])
def predict_demand(req: DemandReq):
    items = req.items or list_available_items()
    if not items:
        raise HTTPException(404, "No demand artifacts found under ml/artifacts/demand/.")
    df = _hist_with_weather_and_logs()
    out: List[DemandResItem] = []
    for item in items:
        try:
            pred = predict_one_item(df, item)
            out.append(DemandResItem(
                item_code=pred["item_code"],
                yhat=_clean_num(pred["yhat"]) or 0.0,
                p10=_clean_num(pred["p10"]),
                p90=_clean_num(pred["p90"]),
            ))
        except FileNotFoundError:
            continue
        except Exception as e:
            raise HTTPException(500, f"Error predicting item '{item}': {e}")
    if not out:
        raise HTTPException(404, "No demand predictions produced.")
    return out

@app.post("/predict/syndromes", response_model=List[SyndromeResItem])
def predict_syndromes(req: SyndromesReq):
    df = _hist_with_weather_and_logs()
    syns = req.syndromes or list_available_syndromes()
    if not syns:
        raise HTTPException(404, "No syndrome artifacts found under ml/artifacts/syndromes/.")
    out = []
    for s in syns:
        try:
            out.append(predict_one_syn(df, s))
        except Exception:
            continue
    if not out:
        raise HTTPException(404, "No syndrome predictions produced.")
    out = sorted(out, key=lambda x: x["prob"], reverse=True)[: max(1, req.top_n)]
    return [SyndromeResItem(syndrome=o["syndrome"], prob=round(float(o["prob"]), 3), rank=i+1) for i, o in enumerate(out)]

# =========================
# Nurse Log (IST calendar)
# =========================
def _load_nurse_log() -> dict:
    return _safe_load_json(NURSE_LOG_JSON)

def _save_nurse_log_entry(date_str: str, payload: dict, merge: bool = True):
    data = _load_nurse_log()
    existing = data.get(date_str, {}) if merge else {}

    for k in ["fever", "cough", "diarrhea", "vomiting", "cold", "others"]:
        v_old = existing.get(k, 0)
        v_new = payload.get(k)
        if v_new is not None:
            if isinstance(v_old, (int, float)) and isinstance(v_new, (int, float)):
                existing[k] = int(v_old) + int(v_new)
            else:
                existing[k] = int(v_new)
        elif k not in existing:
            existing[k] = 0

    if payload.get("notes") is not None:
        existing["notes"] = payload["notes"]
    if payload.get("by") is not None:
        existing["by"] = payload["by"]

    existing["date"] = date_str
    data[date_str] = existing
    _safe_save_json(NURSE_LOG_JSON, data)

@app.post("/nurse/log")
def nurse_log(req: NurseLogReq):
    if req.date:
        try:
            date_norm = pd.to_datetime(req.date).date().strftime("%Y-%m-%d")
        except Exception:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        date_norm = _today_local_str()
        
    payload = req.model_dump()
    payload.pop("date", None)
    _save_nurse_log_entry(date_norm, payload, merge=True)
    return {"ok": True, "saved": _load_nurse_log().get(date_norm, {})}

@app.get("/nurse/log/history")
def nurse_log_history_shortcut(days: int = 7):
    """Return last N days of nurse logs. Alias for /nurse/log/history endpoint defined later."""
    import datetime as dt
    nl = _load_nurse_log()
    today = dt.date.today()
    result = []
    for i in range(days - 1, -1, -1):
        d = today - dt.timedelta(days=i)
        d_str = d.strftime("%Y-%m-%d")
        entry = nl.get(d_str, {})
        result.append({
            "date": d_str,
            "fever": entry.get("fever", 0) or 0,
            "cough": entry.get("cough", 0) or 0,
            "diarrhea": entry.get("diarrhea", 0) or 0,
            "vomiting": entry.get("vomiting", 0) or 0,
            "cold": entry.get("cold", 0) or 0,
            "others": entry.get("others", 0) or 0,
            "notes": entry.get("notes", ""),
            "by": entry.get("by", ""),
            "has_entry": bool(entry),
        })
    return {"days": days, "history": result}

@app.get("/nurse/log/{date}")
def nurse_log_get(date: str):
    try:
        date_norm = pd.to_datetime(date).date().strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    return {"date": date_norm, "log": _load_nurse_log().get(date_norm, {})}


# =========================
# Inventory
# =========================
def _load_inventory() -> dict:
    return _safe_load_json(INVENTORY_JSON) or {
        "paracetamol":  {"name":"Paracetamol 500mg", "on_hand": 200, "reorder_point": 150},
        "ors_packets":  {"name":"ORS Sachets",       "on_hand":  45, "reorder_point":  60},
        "malaria_kits": {"name":"Malaria Test Kits", "on_hand":  30, "reorder_point":  35},
        "antibiotics":  {"name":"Antibiotics",       "on_hand":  40, "reorder_point":  30},
    }

def _save_inventory(inv: dict):
    _safe_save_json(INVENTORY_JSON, inv)

@app.get("/inventory")
def get_inventory():
    return _load_inventory()

@app.post("/inventory/upsert")
def upsert_inventory(req: InventoryUpsertReq):
    inv = _load_inventory()
    row = inv.get(req.item_code, {"name": req.name or req.item_code, "on_hand": 0, "reorder_point": 0})
    if req.name is not None: row["name"] = req.name
    if req.on_hand is not None: row["on_hand"] = int(req.on_hand)
    if req.reorder_point is not None: row["reorder_point"] = int(req.reorder_point)
    inv[req.item_code] = row
    _save_inventory(inv)
    return {"ok": True, "item": req.item_code, "data": row}

@app.get("/inventory/enriched")
def get_inventory_enriched():
    """
    Returns inventory enriched with demand-based days_to_stockout and daily_demand per item.
    """
    inv = _load_inventory()
    available = set(list_available_items())
    df = _hist_with_weather_and_logs()
    result = {}
    for code, row in inv.items():
        entry = dict(row)
        entry["days_to_stockout"] = None
        entry["daily_demand"] = None
        if code in available:
            try:
                pred = predict_one_item(df, code)
                yhat = max(0.0, float(pred.get("yhat", 0) or 0))
                p10  = _clean_num(pred.get("p10"))
                p90  = _clean_num(pred.get("p90"))
                if yhat > 0.1:
                    entry["days_to_stockout"] = int(row["on_hand"] / yhat)
                    entry["daily_demand"] = round(yhat, 2)
                    entry["daily_demand_p10"] = p10
                    entry["daily_demand_p90"] = p90
            except Exception:
                pass
        result[code] = entry
    return result

# =========================
# Alerts Engine
# =========================
def compute_critical_alerts(demand_preds: List[DemandResItem], inv: dict) -> List[dict]:
    alerts = []
    for d in demand_preds:
        inv_row = inv.get(d.item_code)
        if not inv_row:
            continue

        need = max(0.0, float(d.yhat or 0))
        high_today = max(0.0, float(d.p90 or need))
        weekly_high = high_today * 7.0

        # Days to stockout: on_hand / daily_demand
        dos = int(inv_row["on_hand"] / need) if need > 0.1 else None

        severity = None
        if inv_row["on_hand"] < inv_row["reorder_point"] * 0.25:
            severity = "HIGH"
        elif inv_row["on_hand"] < inv_row["reorder_point"] * 0.5:
            severity = "MEDIUM"
        elif inv_row["on_hand"] <= inv_row["reorder_point"]:
            severity = "LOW"

        if severity:
            alerts.append({
                "type": "stockout_risk",
                "severity": severity,
                "message": f"{inv_row['name']}: only {inv_row['on_hand']} left (reorder level {inv_row['reorder_point']})",
                "item_code": d.item_code,
                "days_to_stockout": dos,
                "daily_demand": round(need, 2),
            })

        if weekly_high > inv_row["on_hand"]:
            alerts.append({
                "type": "stockout_risk",
                "severity": "HIGH",
                "message": f"{inv_row['name']}: need ~{weekly_high:.0f} this week, only {inv_row['on_hand']} in stock",
                "item_code": d.item_code,
                "days_to_stockout": dos,
                "daily_demand": round(need, 2),
            })
        elif weekly_high > inv_row["reorder_point"]:
            alerts.append({
                "type": "reorder",
                "severity": "MEDIUM",
                "message": f"{inv_row['name']}: need ~{weekly_high:.0f} this week, reorder level is {inv_row['reorder_point']}",
                "item_code": d.item_code,
                "days_to_stockout": dos,
                "daily_demand": round(need, 2),
            })

    alerts.sort(key=lambda a: 0 if a["severity"] == "HIGH" else (1 if a["severity"] == "MEDIUM" else 2))
    return alerts

@app.get("/alerts")
def get_all_alerts():
    items = list_available_items()
    demand_list = predict_demand(DemandReq(items=items))
    inv = _load_inventory()
    alerts = compute_critical_alerts(demand_list, inv)
    return {"alerts": alerts}

@app.get("/alerts/outbreak")
def outbreak_detection():
    """
    Detect syndromic outbreaks by comparing last 7 days vs prior 7 days of nurse logs.
    Triggers alert when:
      - current_week >= 2× prior_week AND current_week >= 3 cases (surge), OR
      - prior_week == 0 AND current_week >= 5 cases (new emergence).
    """
    nl = _load_nurse_log()
    today = dt.datetime.now(tz=LOCAL_TZ).date()
    symptoms = ["fever", "cough", "diarrhea", "vomiting", "cold"]

    last7: dict = {s: 0 for s in symptoms}
    prior7: dict = {s: 0 for s in symptoms}
    days_with_cases: dict = {s: 0 for s in symptoms}
    daily_last7: dict = {s: [] for s in symptoms}

    for i in range(7):
        d = today - dt.timedelta(days=i)
        entry = nl.get(d.strftime("%Y-%m-%d"), {})
        for s in symptoms:
            v = int(entry.get(s, 0) or 0)
            last7[s] += v
            daily_last7[s].append(v)
            if v > 0:
                days_with_cases[s] += 1

    for i in range(7, 14):
        d = today - dt.timedelta(days=i)
        entry = nl.get(d.strftime("%Y-%m-%d"), {})
        for s in symptoms:
            prior7[s] += int(entry.get(s, 0) or 0)

    outbreak_alerts = []
    for s in symptoms:
        curr = last7[s]
        base = prior7[s]
        is_surge = base > 0 and curr >= base * 2.0 and curr >= 3
        is_new   = base == 0 and curr >= 5

        if is_surge or is_new:
            ratio = round(curr / max(1, base), 1) if base > 0 else None
            severity = "HIGH" if (ratio is not None and ratio >= 3.0) or (is_new and curr >= 10) else "MEDIUM"
            outbreak_alerts.append({
                "syndrome": s,
                "label": s.replace("_", " ").title(),
                "current_7d_cases": curr,
                "baseline_7d_cases": base,
                "ratio": ratio,
                "severity": severity,
                "days_with_cases": days_with_cases[s],
                "daily_trend": list(reversed(daily_last7[s])),  # oldest→newest
                "message": (
                    f"{s.title()}: {curr} cases this week vs {base} prior week ({ratio}x surge)"
                    if is_surge else
                    f"{s.title()}: {curr} new cases this week - no prior baseline, monitor closely"
                ),
                "recommendation": (
                    "Immediate investigation and isolation protocol recommended."
                    if severity == "HIGH" else
                    "Increase surveillance frequency and ensure adequate supplies."
                ),
            })

    outbreak_alerts.sort(key=lambda x: (0 if x["severity"] == "HIGH" else 1, -x["current_7d_cases"]))

    return {
        "outbreak_alerts": outbreak_alerts,
        "has_high_outbreak": any(a["severity"] == "HIGH" for a in outbreak_alerts),
        "total_outbreaks": len(outbreak_alerts),
        "checked_at": today.strftime("%Y-%m-%d"),
        "window_days": 7,
    }


# Mobile Aggregator
# =========================
@app.get("/mobile/today")
def mobile_today():
    # 1. Volume
    vol = predict_volume(VolumeReq())

    # 2. Demand
    items = list_available_items()
    demand_list = predict_demand(DemandReq(items=items))

    # 3. Inventory + Alerts
    inv = _load_inventory()
    alerts = compute_critical_alerts(demand_list, inv)
    high_alerts = [a for a in alerts if a["severity"] == "HIGH"]

    # 4. Syndromes
    try:
        syn_top = predict_syndromes(SyndromesReq(top_n=3))
        syn_payload = [s.model_dump() for s in syn_top]
    except Exception:
        syn_payload = []

    # 5. Nurse log for IST today
    nl = _load_nurse_log()
    today_local = _today_local_str()
    nurse_today = nl.get(today_local, {})

    # 6. Delta vs yesterday
    df = _hist_with_weather_and_logs()
    try:
        yday = float(df.iloc[-2]["total_patients"])
        raw_delta = ((vol.predicted_visits - yday) / max(1.0, yday)) * 100
        delta_pct = round(max(-999.0, min(999.0, raw_delta)), 1)
    except Exception:
        delta_pct = 0.0

    # 7. Confidence interval bounds
    p10_res = float(vol_intervals.get("residual_p10", -5.0))
    p90_res = float(vol_intervals.get("residual_p90", 5.0))

    # 8. Last 7 days volumes for sparkline
    try:
        last7 = df.tail(7)["total_patients"].fillna(0).astype(float).tolist()
    except Exception:
        last7 = []

    # 9. Weather context
    try:
        last_row = df.iloc[-1]
        weather_ctx = {
            "temperature": _clean_num(last_row.get("temperature")),
            "rainfall": _clean_num(last_row.get("rainfall")),
            "humidity": _clean_num(last_row.get("humidity")),
        }
    except Exception:
        weather_ctx = {}

    return {
        "expected_patients": vol.predicted_visits,
        "expected_patients_p10": _clean_num(max(0.0, vol.predicted_visits + p10_res)),
        "expected_patients_p90": _clean_num(max(0.0, vol.predicted_visits + p90_res)),
        "delta_vs_yesterday_pct": delta_pct,
        "last7_volumes": last7,
        "weather": weather_ctx,
        "status": {
            "level": compute_status_level(vol.predicted_visits),
            "reason": "Based on percentile thresholds (last 90 days)"
        },
        "top_syndromes": syn_payload,
        "critical_alerts": high_alerts,
        "all_alerts_count": len(alerts),
        "demand_preview": [
            {
                "item_code": d.item_code,
                "yhat": _clean_num(d.yhat),
                "p10": _clean_num(d.p10),
                "p90": _clean_num(d.p90),
            } for d in demand_list
        ][:4],
        "nurse_log_today": nurse_today,
        "for_date": today_local,
    }

# =========================
# Weather endpoints
# =========================
@app.post("/weather/upsert")
def weather_upsert(req: WeatherUpsertReq):
    try:
        date_norm = pd.to_datetime(req.date).date().strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    _save_weather_override(date_norm, req.temperature, req.rainfall, req.humidity)
    return {"ok": True, "date": date_norm, "applied": {"temperature": req.temperature, "rainfall": req.rainfall, "humidity": req.humidity}}

@app.get("/weather/today")
def weather_today():
    df = _hist_with_weather_and_logs()
    if df.empty:
        raise HTTPException(404, "No history.")
    last = df.iloc[-1]
    return {
        "date": str(last["date"].date()),
        "temperature": None if "temperature" not in df.columns else _clean_num(last.get("temperature")),
        "rainfall": None if "rainfall" not in df.columns else _clean_num(last.get("rainfall")),
        "humidity": None if "humidity" not in df.columns else _clean_num(last.get("humidity")),
    }

@app.post("/weather/fetch")
def weather_fetch(req: WeatherFetchReq):
    if requests is None:
        raise HTTPException(500, "requests library is not installed.")
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(500, "Server misconfig: OPENWEATHER_API_KEY is missing")
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        raise HTTPException(400, "Invalid latitude or longitude")

    date_norm = req.date or _today_local_str()
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(
            url,
            params={"lat": req.lat, "lon": req.lon, "appid": api_key, "units": req.units},
            timeout=10,
        )
    except Exception as e:
        raise HTTPException(502, f"Weather provider network error: {e}")

    if r.status_code >= 400:
        raise HTTPException(502, f"Weather provider error ({r.status_code}): {r.text[:200]}")

    data = r.json()
    main = data.get("main", {}) if isinstance(data, dict) else {}
    temp = main.get("temp")
    humid = main.get("humidity")

    rain = None
    rain_obj = data.get("rain") if isinstance(data, dict) else None
    if isinstance(rain_obj, dict):
        rain = rain_obj.get("1h") or rain_obj.get("3h")

    _save_weather_override(date_norm, temp, rain, humid)
    return {"ok": True, "date": date_norm, "source": "openweather", "applied": {"temperature": temp, "rainfall": rain, "humidity": humid}}

# =========================
# Debug & Telemetry
# =========================
@app.get("/debug/status-thresholds")
def debug_status_thresholds():
    df = _hist_with_weather_and_logs()
    last90 = df.tail(90)["total_patients"].dropna().astype(float)
    if len(last90) < 10:
        return {"mode": "fallback", "green_lt": 50, "yellow_lt": 80}
    p60 = float(np.percentile(last90, 60))
    p85 = float(np.percentile(last90, 85))
    return {"mode": "percentile", "p60_green_max": round(p60, 2), "p85_yellow_max": round(p85, 2)}

@app.get("/debug/env")
def debug_env():
    return {"OPENWEATHER_API_KEY_present": bool(os.getenv("OPENWEATHER_API_KEY"))}

@app.get("/debug/where")
def debug_where():
    return {
        "cwd": os.getcwd(),
        "env_path": _ENV_PATH,
        "cached_rows": len(_cached_hist_base) if _cached_hist_base is not None else 0,
    }

@app.get("/debug/nurse-log")
def debug_nurse_log():
    """Return raw nurse log JSON for inspection."""
    return _load_nurse_log()

# =========================
# Stats & Analytics
# =========================
@app.get("/stats/summary")
def stats_summary():
    """
    Return 7-day and 30-day patient volume statistics for dashboard analytics.
    """
    df = _hist_with_weather_and_logs()
    if df.empty:
        raise HTTPException(404, "No history data available.")

    tp = df["total_patients"].dropna().astype(float)

    def _stats(series: pd.Series) -> dict:
        return {
            "mean": round(float(series.mean()), 1) if len(series) > 0 else None,
            "max": round(float(series.max()), 1) if len(series) > 0 else None,
            "min": round(float(series.min()), 1) if len(series) > 0 else None,
            "std": round(float(series.std()), 1) if len(series) > 0 else None,
        }

    last7 = df.tail(7)["total_patients"].dropna().astype(float)
    last30 = df.tail(30)["total_patients"].dropna().astype(float)

    # Day-of-week breakdown (last 90 days)
    df90 = df.tail(90).copy()
    if "date" in df90.columns:
        df90["weekday"] = pd.to_datetime(df90["date"]).dt.day_name()
        dow = df90.groupby("weekday")["total_patients"].mean().dropna()
        dow_dict = {k: round(float(v), 1) for k, v in dow.items()}
    else:
        dow_dict = {}

    # Trend: compare last 7 vs prior 7
    prior7 = df.iloc[-14:-7]["total_patients"].dropna().astype(float) if len(df) >= 14 else pd.Series(dtype=float)
    if len(last7) > 0 and len(prior7) > 0:
        trend_pct = round(((last7.mean() - prior7.mean()) / max(1.0, prior7.mean())) * 100, 1)
    else:
        trend_pct = None

    return {
        "last7": {
            "dates": df.tail(7)["date"].dt.strftime("%Y-%m-%d").tolist() if "date" in df.columns else [],
            "volumes": df.tail(7)["total_patients"].fillna(0).astype(float).tolist(),
            **_stats(last7),
        },
        "last30": _stats(last30),
        "day_of_week_avg": dow_dict,
        "trend_7d_pct": trend_pct,
        "total_records": int(len(tp)),
    }

@app.get("/nurse/log/history")
def nurse_log_history(days: int = 7):
    """Return last N days of nurse logs as an ordered list."""
    nl = _load_nurse_log()
    today = dt.date.today()
    result = []
    for i in range(days - 1, -1, -1):
        d = today - dt.timedelta(days=i)
        d_str = d.strftime("%Y-%m-%d")
        entry = nl.get(d_str, {})
        result.append({
            "date": d_str,
            "fever": entry.get("fever", 0) or 0,
            "cough": entry.get("cough", 0) or 0,
            "diarrhea": entry.get("diarrhea", 0) or 0,
            "vomiting": entry.get("vomiting", 0) or 0,
            "cold": entry.get("cold", 0) or 0,
            "others": entry.get("others", 0) or 0,
            "notes": entry.get("notes", ""),
            "by": entry.get("by", ""),
            "has_entry": bool(entry),
        })
    return {"days": days, "history": result}

