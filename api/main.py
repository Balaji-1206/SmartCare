# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json, os, datetime as dt

# ---- Robust .env loading (works no matter where you launch uvicorn) ----
# Put your .env in project root (e.g., C:\...\smartcare\.env)
from pathlib import Path
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

# Optional live weather fetch
try:
    import requests
except Exception:
    requests = None

# services
from .services.demand import predict_one_item, list_available_items
from .services.syndromes import list_available_syndromes, predict_one_syn

# ========= Paths & artifacts =========
ART_DIR = Path("ml/artifacts")
VOL_MODEL_PATH = ART_DIR / "volume_model.pkl"
VOL_FEATS_PATH = ART_DIR / "volume_features.json"
VOL_INTV_PATH  = ART_DIR / "volume_intervals.json"
DATA_CSV = Path("data/raw/data10yrs.csv")

WEATHER_OVERRIDES_JSON = Path("data/raw/weather_overrides.json")  # simple persistence
WEATHER_API_JSON = Path("data/raw/weather_api.json")  # for API-fetched weather data
NURSE_LOGS_JSON = Path("data/raw/nurse_logs.json")  # for nurse logs
INVENTORY_JSON = Path("data/raw/inventory.json")  # for inventory persistence

if not VOL_MODEL_PATH.exists() or not VOL_INTV_PATH.exists():
    raise RuntimeError("Volume artifacts missing. Ensure volume_model.pkl and volume_intervals.json exist in ml/artifacts/.")

if not DATA_CSV.exists():
    raise RuntimeError("data/raw/data10yrs.csv not found. Put your CSV there (same one used in Colab).")

# ========= Load artifacts & history =========
vol_model = joblib.load(VOL_MODEL_PATH)
vol_intervals = json.load(open(VOL_INTV_PATH))
vol_feat_list = json.load(open(VOL_FEATS_PATH)).get("features", []) if VOL_FEATS_PATH.exists() else None

def _load_hist() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df

def _load_weather_overrides() -> dict:
    if WEATHER_OVERRIDES_JSON.exists():
        try:
            return json.load(open(WEATHER_OVERRIDES_JSON))
        except Exception:
            return {}
    return {}

def _save_weather_override(date_str: str, temperature: float | None, rainfall: float | None, humidity: float | None):
    data = _load_weather_overrides()
    data[date_str] = {
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity
    }
    WEATHER_OVERRIDES_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(WEATHER_OVERRIDES_JSON, "w"), indent=2)

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

# Keep a base history and always apply latest overrides when predicting
_hist_base = _load_hist()

def _hist_with_weather() -> pd.DataFrame:
    return _apply_weather_overrides(_hist_base)

# ========= Feature builders (mirror Colab) =========
def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d["total_patients"] = pd.to_numeric(d["total_patients"], errors="coerce")
    # lags
    for lag in [1,7,14,28]:
        d[f"lag_{lag}"] = d["total_patients"].shift(lag)
    # rollings
    for w in [7,14,28]:
        d[f"roll_mean_{w}"] = d["total_patients"].rolling(w).mean()
        d[f"roll_std_{w}"]  = d["total_patients"].rolling(w).std()
    # calendar
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    # weather
    for col in ["temperature","rainfall","humidity"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d

def prep_X_from_features(feat_df: pd.DataFrame, training_features: Optional[List[str]] = None) -> pd.DataFrame:
    X = feat_df.copy()
    X = X.drop(columns=[c for c in ["date","total_patients"] if c in X.columns], errors="ignore")

    # drop constant cols
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    # encode non-numeric
    for c in X.select_dtypes(include=["object","category"]).columns:
        X[c] = X[c].astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Align to training feature order if provided
    if training_features:
        for col in training_features:
            if col not in X.columns:
                X[col] = 0
        X = X[training_features]
    return X

# ========= Pydantic models =========
class VolumeReq(BaseModel):
    facility_id: str = "C001"  # reserved for future multi-center

class VolumeRes(BaseModel):
    predicted_visits: float
    p10: float | None = None
    p90: float | None = None
    model_version: str = "v0.3.0"

class DemandReq(BaseModel):
    items: Optional[List[str]] = None  # e.g., ["paracetamol","ors_packets"]

class DemandResItem(BaseModel):
    item_code: str
    yhat: float
    p10: float | None = None
    p90: float | None = None

class SyndromesReq(BaseModel):
    top_n: int = 3
    syndromes: Optional[List[str]] = None  # e.g., ["fever","diarrhea","cough"]

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
    units: str = "metric"  # metric/imperial
    provider: str = "openweather"  # placeholder for future providers

# ========= FastAPI app =========
app = FastAPI(title="SmartCare API", version="0.3.0")

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
            "/weather/upsert (POST)",
            "/weather/today (GET)",
            "/weather/fetch (POST)",
            "/debug/status-thresholds (GET)",
            "/debug/env (GET)",
            "/debug/where (GET)",
            "/debug/dotenv (GET)"
        ]
    })

# ========= Helpers =========
def _clean_num(x: float | None) -> float | None:
    if x is None: return None
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

# ========= /predict/volume =========
@app.post("/predict/volume", response_model=VolumeRes)
def predict_volume(req: VolumeReq):
    df = _hist_with_weather()  # <-- merge in overrides
    feats = build_volume_features(df)
    need = [c for c in feats.columns if c.startswith("lag_") or c.startswith("roll_")]
    feats = feats.dropna(subset=need)
    if feats.empty:
        raise HTTPException(400, "Not enough history to form features (lags/rollings).")

    X_all = prep_X_from_features(feats, vol_feat_list)
    x = X_all.iloc[[-1]]
    yhat = float(vol_model.predict(x)[0])

    p10_res = float(vol_intervals.get("residual_p10", -1.0))
    p90_res = float(vol_intervals.get("residual_p90",  1.0))
    p10 = yhat + p10_res
    p90 = yhat + p90_res

    return VolumeRes(
        predicted_visits=_clean_num(yhat),
        p10=_clean_num(p10),
        p90=_clean_num(p90),
        model_version="v0.3.0"
    )

# ========= Demand prediction =========
@app.post("/predict/demand", response_model=List[DemandResItem])
def predict_demand(req: DemandReq):
    items = req.items or list_available_items()
    if not items:
        raise HTTPException(404, "No demand artifacts found under ml/artifacts/demand/.")

    df = _hist_with_weather()  # demand features may also use weather
    if "date" not in df or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise HTTPException(500, "History 'date' column invalid or missing.")

    out: List[DemandResItem] = []
    for item in items:
        try:
            pred = predict_one_item(df, item)  # returns dict
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

# ========= Syndrome prediction =========
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

# ========= Mobile aggregator =========
INVENTORY = {
    "paracetamol":  {"name":"Paracetamol 500mg", "on_hand": 200, "reorder_point": 150},
    "ors_packets":  {"name":"ORS Sachets",       "on_hand":  45, "reorder_point":  60},
    "malaria_kits": {"name":"Malaria Test Kits", "on_hand":  30, "reorder_point":  35},
    "antibiotics":  {"name":"Antibiotics",       "on_hand":  40, "reorder_point":  30},
}

def compute_critical_alerts(demand_preds: List[DemandResItem]) -> List[dict]:
    alerts = []
    for d in demand_preds:
        inv = INVENTORY.get(d.item_code)
        if not inv:
            continue
        need = max(0.0, float(d.yhat or 0))
        high_today = max(0.0, float(d.p90 or need))
        weekly_high = high_today * 7.0
        if weekly_high > inv["on_hand"]:
            alerts.append({
                "type": "stockout_risk",
                "severity": "HIGH",
                "message": f"{d.item_code}: 7-day p90 {weekly_high:.0f} > on-hand {inv['on_hand']}",
                "item_code": d.item_code
            })
        elif weekly_high > inv["reorder_point"]:
            alerts.append({
                "type": "reorder",
                "severity": "MEDIUM",
                "message": f"{d.item_code}: 7-day p90 {weekly_high:.0f} near reorder point {inv['reorder_point']}",
                "item_code": d.item_code
            })
    alerts.sort(key=lambda a: 0 if a["severity"]=="HIGH" else 1)
    return alerts

@app.get("/mobile/today")
def mobile_today():
    # 1) Volume + delta vs yesterday
    vol = predict_volume(VolumeReq())

    df = _hist_with_weather()
    df = df.sort_values("date").reset_index(drop=True)

    # last two actuals for delta% context
    yhat_today = float(vol.predicted_visits)
    try:
        last_two = df.tail(2)["total_patients"].astype(float).tolist()
        yesterday = last_two[-1] if len(last_two) >= 1 else None
    except Exception:
        yesterday = None

    delta_pct = None
    if yesterday and yesterday > 0:
        delta_pct = round((yhat_today - yesterday) / yesterday * 100.0, 1)

    # 2) Demand + full alerts (not sliced to just 1)
    items = list_available_items()
    demand_list = predict_demand(DemandReq(items=items))
    alerts = compute_critical_alerts(demand_list)  # full list

    # 3) Top syndromes (best-effort)
    try:
        syn_top = predict_syndromes(SyndromesReq(top_n=3))
        syn_payload = [s.dict() for s in syn_top]
    except Exception:
        syn_payload = []

    # 4) Today’s nurse log snapshot
    today_str = dt.date.today().strftime("%Y-%m-%d")
    nurse_log_today = _get_nurse_log(today_str) if 'NURSE_LOGS_JSON' in globals() else {}

    return {
        "expected_patients": _clean_num(yhat_today),
        "delta_vs_yesterday_pct": delta_pct,  # e.g., 20.0 means +20%
        "status": {
            "level": compute_status_level(yhat_today),
            "reason": "Based on percentile thresholds (last 90 days)"
        },
        "top_syndromes": syn_payload,
        "critical_alerts": alerts,  # full list with severity + messages
        "demand_preview": [
            {
              "item_code": d.item_code,
              "yhat": _clean_num(d.yhat),
              "p10": _clean_num(d.p10),
              "p90": _clean_num(d.p90)
            } for d in demand_list
        ][:3],
        "nurse_log_today": nurse_log_today
    }


# ========= Weather endpoints =========
@app.post("/weather/upsert")
def weather_upsert(req: WeatherUpsertReq):
    """Manually upsert weather for a date; affects predictions immediately."""
    try:
        # Validate date string and normalize
        date_norm = pd.to_datetime(req.date).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    _save_weather_override(date_norm, req.temperature, req.rainfall, req.humidity)
    return {"ok": True, "date": date_norm, "applied": {"temperature": req.temperature, "rainfall": req.rainfall, "humidity": req.humidity}}

@app.get("/weather/today")
def weather_today():
    """See the weather row the model will use for the last date available."""
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

# ---- LIVE FETCH from OpenWeather, then upsert ----
@app.post("/weather/fetch")
def weather_fetch(req: WeatherFetchReq):
    """
    Fetch current weather from OpenWeather and upsert for 'date' (default today).
    Requires OPENWEATHER_API_KEY in environment (never hardcode keys).
    """
    if requests is None:
        raise HTTPException(500, "requests not available. pip install requests.")

    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(500, "Server misconfig: OPENWEATHER_API_KEY is missing")

    # Basic lat/lon sanity
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        raise HTTPException(400, "Invalid lat/lon")

    date_norm = req.date or dt.date.today().strftime("%Y-%m-%d")

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
        # pass through provider message for quick diagnosis
        raise HTTPException(502, f"Weather provider auth error (401): {r.text[:200]}")
    if r.status_code >= 400:
        raise HTTPException(502, f"Weather provider error {r.status_code}: {r.text[:200]}")

    data = r.json()
    main = data.get("main", {}) if isinstance(data, dict) else {}
    temp = main.get("temp")
    humid = main.get("humidity")

    # rainfall may appear under 'rain': {'1h': x} or {'3h': y}
    rain = None
    rain_obj = data.get("rain") if isinstance(data, dict) else None
    if isinstance(rain_obj, dict):
        rain = rain_obj.get("1h")
        if rain is None:
            rain = rain_obj.get("3h")

    _save_weather_override(date_norm, temp, rain, humid)
    return {
        "ok": True,
        "date": date_norm,
        "source": "openweather",
        "applied": {"temperature": temp, "rainfall": rain, "humidity": humid},
    }

@app.get("/debug/status-thresholds")
def debug_status_thresholds():
    df = _hist_with_weather()
    last90 = df.tail(90)["total_patients"].astype(float)
    if len(last90) < 10:
        return {"mode":"fallback", "green_lt":50, "yellow_lt":80}
    p60 = float(np.percentile(last90, 60))
    p85 = float(np.percentile(last90, 85))
    return {"mode":"percentile", "p60_green_max": round(p60,2), "p85_yellow_max": round(p85,2)}

def _safe_load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}

def _safe_save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w"), indent=2)

def _save_weather_api(date_str: str, temperature: float | None, rainfall: float | None, humidity: float | None):
    data = _safe_load_json(WEATHER_API_JSON)
    data[date_str] = {"temperature": temperature, "rainfall": rainfall, "humidity": humidity, "source": "api"}
    _safe_save_json(WEATHER_API_JSON, data)

# ========= Debug helpers (safe: do not expose secrets) =========
@app.get("/debug/env")
def debug_env():
    return {"OPENWEATHER_API_KEY_present": bool(os.getenv("OPENWEATHER_API_KEY"))}

@app.get("/debug/where")
def debug_where():
    import os, sys
    from dotenv import find_dotenv
    return {
        "cwd": os.getcwd(),
        "main_file": __file__,
        "env_found": find_dotenv(usecwd=True),
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
def _apply_weather_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather from API snapshots only.
    If API has a value, it overrides CSV; otherwise keep CSV.
    """
    api_map = _safe_load_json(WEATHER_API_JSON)

    def map_to_df(m: dict) -> pd.DataFrame:
        if not m:
            return pd.DataFrame(columns=["date","temperature","rainfall","humidity"])
        rows = []
        for k, v in m.items():
            rows.append({
                "date": pd.to_datetime(k).normalize(),
                "temperature": v.get("temperature"),
                "rainfall": v.get("rainfall"),
                "humidity": v.get("humidity"),
            })
        return pd.DataFrame(rows).sort_values("date")

    dfm = df.copy()
    dfm["date"] = pd.to_datetime(dfm["date"]).dt.normalize()

    api_df = map_to_df(api_map)
    if not api_df.empty:
        dfm = dfm.merge(api_df, on="date", how="left", suffixes=("", "_api"))
        for col in ["temperature", "rainfall", "humidity"]:
            api_col = f"{col}_api"
            if api_col in dfm.columns:
                dfm[col] = np.where(dfm[api_col].notna(), dfm[api_col], dfm[col])
        dfm.drop(columns=[c for c in dfm.columns if c.endswith("_api")], inplace=True, errors="ignore")

    return dfm.sort_values("date").reset_index(drop=True)
class NurseLogReq(BaseModel):
    date: str                      # YYYY-MM-DD
    fever: Optional[int] = None
    cough: Optional[int] = None
    diarrhea: Optional[int] = None
    vomiting: Optional[int] = None
    cold: Optional[int] = None
    notes: Optional[str] = None
    by: Optional[str] = None

def _save_nurse_log(date_str: str, payload: dict, merge: bool = True):
    data = _safe_load_json(NURSE_LOGS_JSON)
    existing = data.get(date_str, {}) if merge else {}
    for k in ["fever","cough","diarrhea","vomiting","cold"]:
        v_old = existing.get(k)
        v_new = payload.get(k)
        if v_new is not None:
            if isinstance(v_old, (int, float)) and isinstance(v_new, (int, float)):
                existing[k] = int(v_old) + int(v_new)
            else:
                existing[k] = int(v_new)
    if payload.get("notes") is not None:
        existing["notes"] = payload["notes"]
    if payload.get("by") is not None:
        existing["by"] = payload["by"]
    data[date_str] = existing
    _safe_save_json(NURSE_LOGS_JSON, data)

def _get_nurse_log(date_str: str) -> dict:
    return _safe_load_json(NURSE_LOGS_JSON).get(date_str, {})

@app.post("/nurse/log")
def nurse_log(req: NurseLogReq):
    try:
        date_norm = pd.to_datetime(req.date).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    payload = req.dict()
    payload.pop("date", None)
    _save_nurse_log(date_norm, payload, merge=True)
    return {"ok": True, "date": date_norm, "log": _get_nurse_log(date_norm)}

@app.get("/nurse/log/{date}")
def nurse_log_get(date: str):
    try:
        date_norm = pd.to_datetime(date).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    return {"date": date_norm, "log": _get_nurse_log(date_norm)}
def _load_inventory() -> dict:
    return _safe_load_json(INVENTORY_JSON) or {
        "paracetamol":  {"name":"Paracetamol 500mg", "on_hand": 200, "reorder_point": 150},
        "ors_packets":  {"name":"ORS Sachets",       "on_hand":  45, "reorder_point":  60},
        "malaria_kits": {"name":"Malaria Test Kits", "on_hand":  30, "reorder_point":  35},
        "antibiotics":  {"name":"Antibiotics",       "on_hand":  40, "reorder_point":  30},
    }

def _save_inventory(inv: dict):
    _safe_save_json(INVENTORY_JSON, inv)

    from pydantic import BaseModel

class InventoryUpsertReq(BaseModel):
    item_code: str
    name: Optional[str] = None
    on_hand: Optional[int] = None
    reorder_point: Optional[int] = None

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
    # refresh in-memory reference if you keep one
    global INVENTORY
    INVENTORY = inv
    return {"ok": True, "item": req.item_code, "data": row}
