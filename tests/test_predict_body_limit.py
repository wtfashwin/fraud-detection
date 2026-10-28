"""A prediction must not parse an unbounded incoming body before validation."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool


@pytest.fixture
def prediction_app(monkeypatch):
    module = importlib.import_module("api.app")
    from db.models import Explanation

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Explanation.__table__.create(engine)
    pending = []
    bundle = SimpleNamespace(columns=[f"f{i}" for i in range(1024)],
                             score=lambda values: (0, 0.5), drift_baseline=None)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "BUNDLE", bundle)
    monkeypatch.setattr(module.celery_app, "send_task", lambda *args, **kwargs: pending.append(args))
    monkeypatch.setattr(module, "DRIFT_MONITOR", None)

    yield module.app, engine, pending
    engine.dispose()


def prediction_payload(transaction_id="synthetic-predict"):
    return json.dumps({"transaction_id": transaction_id,
                       "features": [1.7976931348623157e308] * 1024}, separators=(",", ":")).encode()


def test_oversized_prediction_is_rejected_before_persistence_and_dispatch(prediction_app):
    app, engine, pending = prediction_app
    from db.models import Explanation

    payload = b" " * (96 * 1024) + prediction_payload()
    with TestClient(app) as client:
        response = client.post("/predict", content=payload, headers={"content-type": "application/json"})
    assert response.status_code == 413
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Explanation)).scalar_one() == 0
    assert pending == []


def test_chunked_prediction_stops_reading_when_limit_is_crossed(prediction_app):
    app, engine, pending = prediction_app
    from db.models import Explanation
    consumed = []

    async def chunks():
        for index in range(24):
            consumed.append(index)
            yield b" " * 4096
        yield prediction_payload("synthetic-chunked")

    async def submit():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/predict", content=chunks(), headers={"content-type": "application/json"})

    response = asyncio.run(submit())
    assert response.status_code == 413
    assert len(consumed) == 17  # 16 x 4096 is the limit; stop at the first excess chunk.
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Explanation)).scalar_one() == 0
    assert pending == []


def test_maximum_feature_count_with_large_finite_values_is_accepted(prediction_app):
    app, _, pending = prediction_app
    with TestClient(app) as client:
        response = client.post("/predict", content=prediction_payload(),
                               headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert response.json()["transaction_id"] == "synthetic-predict"
    assert len(pending) == 1


def test_prediction_body_limit_includes_the_exact_boundary(prediction_app):
    app, _, pending = prediction_app
    payload = prediction_payload("synthetic-boundary")
    assert len(payload) < 64 * 1024
    at_limit = b" " * (64 * 1024 - len(payload)) + payload
    with TestClient(app) as client:
        accepted = client.post("/predict", content=at_limit, headers={"content-type": "application/json"})
        rejected = client.post("/predict", content=b" " + at_limit, headers={"content-type": "application/json"})
    assert accepted.status_code == 200
    assert rejected.status_code == 413
    assert len(pending) == 1


def test_valid_chunked_prediction_is_replayed_to_the_route(prediction_app):
    app, _, pending = prediction_app
    payload = prediction_payload("synthetic-valid-chunked")

    async def chunks():
        for start in range(0, len(payload), 3000):
            yield payload[start:start + 3000]

    async def submit():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/predict", content=chunks(), headers={"content-type": "application/json"})

    response = asyncio.run(submit())
    assert response.status_code == 200
    assert response.json()["transaction_id"] == "synthetic-valid-chunked"
    assert len(pending) == 1


def test_disconnected_prediction_does_not_raise_or_dispatch(prediction_app):
    app, engine, pending = prediction_app
    from db.models import Explanation

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/predict", "raw_path": b"/predict",
             "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
             "headers": [(b"content-type", b"application/json")]}
    incoming = iter([{"type": "http.request", "body": b'{"features":[1,', "more_body": True},
                     {"type": "http.disconnect"}])
    sent = []

    async def receive():
        return next(incoming, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [400]
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Explanation)).scalar_one() == 0
    assert pending == []


def test_non_prediction_upload_path_is_not_limited(prediction_app):
    app, _, _ = prediction_app

    async def synthetic_upload(request: Request):
        return {"bytes": len(await request.body())}

    app.add_api_route("/__synthetic_upload__", synthetic_upload, methods=["POST"])
    try:
        with TestClient(app) as client:
            response = client.post("/__synthetic_upload__", content=b"x" * (96 * 1024))
        assert response.status_code == 200
        assert response.json() == {"bytes": 96 * 1024}
    finally:
        app.router.routes[:] = [route for route in app.router.routes if route.path != "/__synthetic_upload__"]


def test_mounted_prediction_route_is_also_limited(prediction_app):
    app, _, _ = prediction_app

    async def submit():
        transport = httpx.ASGITransport(app=app, root_path="/api")
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/predict", content=b" " * (96 * 1024) + prediction_payload(),
                                     headers={"content-type": "application/json"})

    assert asyncio.run(submit()).status_code == 413
