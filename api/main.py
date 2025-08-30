# api/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json, os, datetime as dt
from zoneinfo import ZoneInfo

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
print("Loaded .env from:", _ENV_PATH)

# =========================
# Optional live requests
# =========================
try:
    import requests
except Exception:
    requests = None

# =========================
# Services (your ML modules)
# =========================
from .services.demand import predict_one_item, list_available_items
from .services.syndromes import list_available_syndromes, predict_one_syn

# =========================
# Paths & artifacts
# =========================
ART_DIR = Path("ml/artifacts")
VOL_MODEL_PATH = ART_DIR / "volume_model.pkl"
VOL_FEATS_PATH = ART_DIR / "volume_features.json"
VOL_INTV_PATH  = ART_DIR / "volume_intervals.json"

DATA_CSV = Path("data/raw/data10yrs.csv")
WEATHER_OVERRIDES_JSON = Path("data/raw/weather_overrides.json")  # merged into history
NURSE_LOG_JSON = Path("data/raw/nurse_log.json")                  # nurse logs (per day)
INVENTORY_JSON = Path("data/raw/inventory.json")                  # inventory persistence

if not VOL_MODEL_PATH.exists() or not VOL_INTV_PATH.exists():
    raise RuntimeError("Volume artifacts missing. Ensure volume_model.pkl and volume_intervals.json exist in ml/artifacts/.")

if not DATA_CSV.exists():
    raise RuntimeError("data/raw/data10yrs.csv not found. Put your CSV there (same one used in Colab).")

# =========================
# Load model artifacts
# =========================
vol_model = joblib.load(VOL_MODEL_PATH)
vol_intervals = json.load(open(VOL_INTV_PATH))
vol_feat_list = json.load(open(VOL_FEATS_PATH)).get("features", []) if VOL_FEATS_PATH.exists() else None

# =========================
# History & weather merging
# =========================
def _load_hist() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df

def _safe_load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _safe_save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2)

def _load_weather_overrides() -> dict:
    return _safe_load_json(WEATHER_OVERRIDES_JSON)

def _save_weather_override(date_str: str, temperature: float | None, rainfall: float | None, humidity: float | None):
    data = _load_weather_overrides()
    data[date_str] = {"temperature": temperature, "rainfall": rainfall, "humidity": humidity}
    _safe_save_json(WEATHER_OVERRIDES_JSON, data)

def _apply_weather_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overrides by date (YYYY-MM-DD). Overrides win over CSV values."""
    overrides = _load_weather_overrides()
    if not overrides:
        return df
    dfo = pd.DataFrame([
        {"date": pd.to_datetime(k).normalize(),
         "temperature": v.get("temperature"),
         "rainfall": v.get("rainfall"),
         "humidity": v.get("humidity")}
        for k, v in overrides.items()
    ])
    dfo = dfo.sort_values("date")
    dfm = df.copy()
    dfm["date"] = pd.to_datetime(dfm["date"]).dt.normalize()
    dfm = dfm.merge(dfo, on="date", how="left", suffixes=("", "_ovr"))
    for col in ["temperature", "rainfall", "humidity"]:
        ocol = f"{col}_ovr"
        if ocol in dfm.columns:
            dfm[col] = np.where(dfm[ocol].notna(), dfm[ocol], dfm[col])
            dfm.drop(columns=[ocol], inplace=True)
    return dfm.sort_values("date").reset_index(drop=True)

_hist_base = _load_hist()
def _hist_with_weather() -> pd.DataFrame:
    return _apply_weather_overrides(_load_hist())  # re-read so weather edits reflect immediately

# =========================
# Feature builders
# =========================
def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d["total_patients"] = pd.to_numeric(d["total_patients"], errors="coerce")
    # lags
    for lag in [1, 7, 14, 28]:
        d[f"lag_{lag}"] = d["total_patients"].shift(lag)
    # rollings
    for w in [7, 14, 28]:
        d[f"roll_mean_{w}"] = d["total_patients"].rolling(w).mean()
        d[f"roll_std_{w}"]  = d["total_patients"].rolling(w).std()
    # calendar
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    # weather numeric
    for col in ["temperature", "rainfall", "humidity"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d

def prep_X_from_features(feat_df: pd.DataFrame, training_features: Optional[List[str]] = None) -> pd.DataFrame:
    X = feat_df.copy()
    X = X.drop(columns=[c for c in ["date", "total_patients"] if c in X.columns], errors="ignore")
    # drop constant cols
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
    # encode non-numeric
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Align
    if training_features:
        for col in training_features:
            if col not in X.columns:
                X[col] = 0
        X = X[training_features]
    return X

# =========================
# Pydantic models
# =========================
class VolumeReq(BaseModel):
    facility_id: str = "C001"

class VolumeRes(BaseModel):
    predicted_visits: float
    p10: float | None = None
    p90: float | None = None
    model_version: str = "v0.3.0"

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
    date: Optional[str] = None  # if None, use today
    lat: float
    lon: float
    units: str = "metric"
    provider: str = "openweather"

class NurseLogReq(BaseModel):
    date: Optional[str] = None               # "YYYY-MM-DD" (optional)
    fever: Optional[int] = None
    cough: Optional[int] = None
    diarrhea: Optional[int] = None
    vomiting: Optional[int] = None
    cold: Optional[int] = None
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
app = FastAPI(title="SmartCare API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: open; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helpers
# =========================
def _clean_num(x: float | None) -> float | None:
    if x is None:
        return None
    return round(max(0.0, float(x)), 2)

def compute_status_level(yhat: float) -> str:
    df = _hist_with_weather()
    last90 = df.tail(90)["total_patients"].astype(float)
    if len(last90) < 10:
        return "GREEN" if yhat < 50 else ("YELLOW" if yhat < 80 else "RED")
    p60 = np.percentile(last90, 60)
    p85 = np.percentile(last90, 85)
    if yhat <= p60: return "GREEN"
    if yhat <= p85: return "YELLOW"
    return "RED"

# =========================
# Root
# =========================
@app.get("/")
def root():
    return JSONResponse({
        "app": "SmartCare API",
        "status": "ok",
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
            "/debug/status-thresholds (GET)",
            "/debug/env (GET)",
            "/debug/where (GET)",
            "/debug/dotenv (GET)",
            "/debug/nurse-log (GET)"
        ]
    })

# =========================
# Predictions
# =========================
@app.post("/predict/volume", response_model=VolumeRes)
def predict_volume(req: VolumeReq):
    df = _hist_with_weather()
    feats = build_volume_features(df)
    need = [c for c in feats.columns if c.startswith("lag_") or c.startswith("roll_")]
    feats = feats.dropna(subset=need)
    if feats.empty:
        raise HTTPException(400, "Not enough history to form features (lags/rollings).")
    X_all = prep_X_from_features(feats, vol_feat_list)
    x = X_all.iloc[[-1]]
    pred_date = str(feats.iloc[-1]["date"].date())

    yhat = float(vol_model.predict(x)[0])

    p10_res = float(vol_intervals.get("residual_p10", -1.0))
    p90_res = float(vol_intervals.get("residual_p90",  1.0))
    p10 = yhat + p10_res
    p90 = yhat + p90_res

       # get the date for which prediction was made
    

    pred_date = _today_local_str()  # 👈 always use clinic's today in IST

    return {
    "predicted_visits": _clean_num(yhat),
    "p10": _clean_num(p10),
    "p90": _clean_num(p90),
    "model_version": "v0.3.0",
    "for_date": pred_date
}


@app.post("/predict/demand", response_model=List[DemandResItem])
def predict_demand(req: DemandReq):
    items = req.items or list_available_items()
    if not items:
        raise HTTPException(404, "No demand artifacts found under ml/artifacts/demand/.")
    df = _hist_with_weather()
    if "date" not in df or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise HTTPException(500, "History 'date' column invalid or missing.")
    out: List[DemandResItem] = []
    for item in items:
        try:
            pred = predict_one_item(df, item)
            out.append(DemandResItem(
                item_code = pred["item_code"],
                yhat = _clean_num(pred["yhat"]),
                p10  = _clean_num(pred["p10"]),
                p90  = _clean_num(pred["p90"]),
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
    df = _hist_with_weather()
    if "date" not in df or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise HTTPException(500, "History 'date' column invalid or missing.")
    syns = req.syndromes or list_available_syndromes()
    if not syns:
        raise HTTPException(404, "No syndrome artifacts found.")
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
# Nurse log (IST calendar)
# =========================
def _load_nurse_log() -> dict:
    return _safe_load_json(NURSE_LOG_JSON)

def _save_nurse_log_entry(date_str: str, payload: dict, merge: bool = True):
    data = _load_nurse_log()
    existing = data.get(date_str, {}) if merge else {}

    # 🔹 Always ensure all symptom fields exist
    for k in ["fever", "cough", "diarrhea", "vomiting", "cold"]:
        v_old = existing.get(k, 0)  # default to 0 if not present
        v_new = payload.get(k)
        if v_new is not None:
            if isinstance(v_old, (int, float)) and isinstance(v_new, (int, float)):
                existing[k] = int(v_old) + int(v_new)
            else:
                existing[k] = int(v_new)
        elif k not in existing:
            existing[k] = 0  # ensure key always exists

    # overwrite note/author if provided
    if payload.get("notes") is not None:
        existing["notes"] = payload["notes"]
    if payload.get("by") is not None:
        existing["by"] = payload["by"]

    existing["date"] = date_str
    data[date_str] = existing
    _safe_save_json(NURSE_LOG_JSON, data)

@app.post("/nurse/log")
def nurse_log(req: NurseLogReq):
    # normalize to IST calendar day (if no date supplied)
    if req.date:
        try:
            date_norm = pd.to_datetime(req.date).date().strftime("%Y-%m-%d")
        except Exception:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        date_norm = _today_local_str()
    payload = req.dict()
    payload.pop("date", None)
    _save_nurse_log_entry(date_norm, payload, merge=True)
    return {"ok": True, "saved": _load_nurse_log().get(date_norm, {})}

@app.get("/nurse/log/{date}")
def nurse_log_get(date: str):
    try:
        date_norm = pd.to_datetime(date).date().strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    return {"date": date_norm, "log": _load_nurse_log().get(date_norm, {})}

@app.get("/debug/nurse-log")
def debug_nurse_log():
    return _load_nurse_log()

# =========================
# Inventory (with persistence)
# =========================
def _load_inventory() -> dict:
    # default seed if no file yet
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
    row = inv.get(req.item_code, {"name": req.item_code, "on_hand": 0, "reorder_point": 0})
    if req.name is not None: row["name"] = req.name
    if req.on_hand is not None: row["on_hand"] = int(req.on_hand)
    if req.reorder_point is not None: row["reorder_point"] = int(req.reorder_point)
    inv[req.item_code] = row
    _save_inventory(inv)
    return {"ok": True, "item": req.item_code, "data": row}

# =========================
# Mobile aggregator
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

        # 🔹 Stock vs reorder threshold checks
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
                "item_code": d.item_code
            })

        # 🔹 Demand forecast checks
        if weekly_high > inv_row["on_hand"]:
            alerts.append({
                "type": "stockout_risk",
                "severity": "HIGH",
                "message": f"{inv_row['name']}: need {weekly_high:.0f}, only {inv_row['on_hand']} in stock",
                "item_code": d.item_code
            })
        elif weekly_high > inv_row["reorder_point"]:
            alerts.append({
                "type": "reorder",
                "severity": "MEDIUM",
                "message": f"{inv_row['name']}: need {weekly_high:.0f}, reorder level {inv_row['reorder_point']}",
                "item_code": d.item_code
            })

    # 🔹 Sort: HIGH first, then MEDIUM, then LOW
    alerts.sort(key=lambda a: 0 if a["severity"] == "HIGH" else (1 if a["severity"] == "MEDIUM" else 2))
    return alerts

@app.get("/mobile/today")
def mobile_today():
    # volume
    vol = predict_volume(VolumeReq())

    # demand
    items = list_available_items()
    demand_list = predict_demand(DemandReq(items=items))

    # inventory + alerts
    inv = _load_inventory()
    alerts = compute_critical_alerts(demand_list, inv)
    high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
    # syndromes
    try:
        syn_top = predict_syndromes(SyndromesReq(top_n=3))
        syn_payload = [s.dict() for s in syn_top]
    except Exception:
        syn_payload = []

    # nurse log for IST today
    nl = _load_nurse_log()
    today_local = _today_local_str()
    nurse_today = nl.get(today_local, {})

    # delta vs yesterday
    df = _hist_with_weather()
    try:
        yday = float(df.iloc[-2]["total_patients"])
        delta_pct = round(((vol["predicted_visits"] - yday) / max(1.0, yday)) * 100, 1)
    except Exception:
        delta_pct = 0

    # 👇 Add for_date (from volume prediction)
    return {
        "expected_patients": vol["predicted_visits"],
        "delta_vs_yesterday_pct": delta_pct,
        "status": {
            "level": compute_status_level(vol["predicted_visits"]),
            "reason": "Based on percentile thresholds (last 90 days)"
        },
        "top_syndromes": syn_payload,
        "critical_alerts": high_alerts,
        "demand_preview": [
            {
                "item_code": d.item_code,
                "yhat": _clean_num(d.yhat),
            } for d in demand_list
        ][:3],
        "nurse_log_today": nurse_today,
        "for_date": vol.get("for_date"),   # 👈 NEW
    }


# =========================
# Weather endpoints

# =========================
@app.get("/alerts")
def get_all_alerts():
    items = list_available_items()
    demand_list = predict_demand(DemandReq(items=items))
    inv = _load_inventory()
    alerts = compute_critical_alerts(demand_list, inv)
    return {"alerts": alerts}

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
    df = _hist_with_weather()
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
    """
    Fetch current weather from OpenWeather and upsert for 'date' (default today).
    Requires OPENWEATHER_API_KEY in environment.
    """
    if requests is None:
        raise HTTPException(500, "requests not available. pip install requests.")
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(500, "Server misconfig: OPENWEATHER_API_KEY is missing")
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        raise HTTPException(400, "Invalid lat/lon")

    date_norm = (req.date or _today_local_str())

    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(
            url,
            params={"lat": req.lat, "lon": req.lon, "appid": api_key, "units": req.units},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"Weather provider network error: {e}")

    if r.status_code == 401:
        raise HTTPException(502, f"Weather provider auth error (401): {r.text[:200]}")
    if r.status_code >= 400:
        raise HTTPException(502, f"Weather provider error {r.status_code}: {r.text[:200]}")

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
# Debug helpers
# =========================
@app.get("/debug/status-thresholds")
def debug_status_thresholds():
    df = _hist_with_weather()
    last90 = df.tail(90)["total_patients"].astype(float)
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
    import sys
    return {
        "cwd": os.getcwd(),
        "main_file": __file__,
        "env_found": _ENV_PATH,
        "cwd_has_env": ".env" in os.listdir(os.getcwd()),
        "sys_path_head": sys.path[:5],
    }

@app.get("/debug/dotenv")
def debug_dotenv():
    from dotenv import dotenv_values, find_dotenv
    env_path = _ENV_PATH or find_dotenv(usecwd=True)
    vals = dotenv_values(env_path) if env_path else {}
    return {
        "env_path": env_path,
        "has_key_in_file": "OPENWEATHER_API_KEY" in vals,
        "key_length_in_file": len(vals.get("OPENWEATHER_API_KEY", "")) if "OPENWEATHER_API_KEY" in vals else 0,
        "visible_to_os_getenv": bool(os.getenv("OPENWEATHER_API_KEY")),
    }
