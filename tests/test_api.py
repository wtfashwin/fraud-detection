from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["status"] in ("UP", "OK")


def test_predict_minimal():
    # Use a minimal features payload that the model can accept (fallback)
    payload = {"features": [0.1]*3}
    r = client.post("/predict", json=payload)
    # We expect either 200 or 202 depending on implementation; check for not server error
    assert r.status_code in (200, 202, 201)
