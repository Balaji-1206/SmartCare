import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "SmartCare API"
    assert data["status"] == "ok"
    assert "endpoints" in data

def test_mobile_today_endpoint():
    response = client.get("/mobile/today")
    assert response.status_code == 200
    data = response.json()
    assert "expected_patients" in data
    assert "status" in data
    assert data["status"]["level"] in ["GREEN", "YELLOW", "RED"]
    assert "top_syndromes" in data
    assert isinstance(data["top_syndromes"], list)
    assert "critical_alerts" in data
    assert "demand_preview" in data
    assert "for_date" in data

def test_predict_volume_endpoint():
    response = client.post("/predict/volume", json={"facility_id": "C001"})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_visits" in data
    assert data["predicted_visits"] >= 0
    assert "p10" in data
    assert "p90" in data
    assert data["p90"] >= data["p10"]
    assert "model_version" in data
    assert "for_date" in data

def test_predict_demand_endpoint():
    response = client.post("/predict/demand", json={"items": ["paracetamol", "ors_packets"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert item["item_code"] in ["paracetamol", "ors_packets"]
        assert item["yhat"] >= 0
        assert item["p90"] >= item["p10"]

def test_predict_syndromes_endpoint():
    response = client.post("/predict/syndromes", json={"top_n": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 3
    for s in data:
        assert "syndrome" in s
        assert 0.0 <= s["prob"] <= 1.0
        assert "rank" in s

def test_nurse_log_roundtrip():
    test_date = "2026-08-21"
    payload = {
        "date": test_date,
        "fever": 4,
        "cough": 2,
        "diarrhea": 1,
        "notes": "Automated test entry",
        "by": "TestNurse"
    }
    post_res = client.post("/nurse/log", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["ok"] is True
    
    get_res = client.get(f"/nurse/log/{test_date}")
    assert get_res.status_code == 200
    log = get_res.json()["log"]
    assert log["fever"] >= 4
    assert log["notes"] == "Automated test entry"

def test_inventory_endpoints():
    get_res = client.get("/inventory")
    assert get_res.status_code == 200
    inv = get_res.json()
    assert "paracetamol" in inv
    
    # Upsert test
    upsert_res = client.post("/inventory/upsert", json={
        "item_code": "paracetamol",
        "on_hand": 180,
        "reorder_point": 100
    })
    assert upsert_res.status_code == 200
    assert upsert_res.json()["ok"] is True
    assert upsert_res.json()["data"]["on_hand"] == 180

def test_alerts_endpoint():
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

def test_weather_endpoints():
    test_date = "2026-08-21"
    upsert_res = client.post("/weather/upsert", json={
        "date": test_date,
        "temperature": 28.5,
        "rainfall": 10.0,
        "humidity": 75.0
    })
    assert upsert_res.status_code == 200
    assert upsert_res.json()["ok"] is True
    
    get_res = client.get("/weather/today")
    assert get_res.status_code == 200
