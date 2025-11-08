from fastapi.testclient import TestClient
from api.app import app 
client = TestClient(app)

def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["status"] in ("UP", "OK")

def test_predict_minimal():
    payload = {"features": [0.1]*30}
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 202, 201)
