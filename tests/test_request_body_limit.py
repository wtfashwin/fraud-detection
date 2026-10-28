"""Prediction request limits with real ASGI requests and isolated SQLite storage.

The fixed scoring double avoids model fitting; no broker or remote service is used.
"""
import asyncio
import importlib
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import StaticPool


@pytest.fixture
def prediction_app(monkeypatch):
    module = importlib.import_module("api.app")
    from db.models import Explanation

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Explanation.__table__.create(engine)
    scored, dispatched, statements = [], [], []

    def score(features):
        scored.append(features)
        return 0, 0.25

    def dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))

    event.listen(engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace(columns=["Time", "V1", "Amount"], score=score))
    monkeypatch.setattr(module, "check_broker", lambda: True)
    monkeypatch.setattr(module.celery_app, "send_task", dispatch)
    monkeypatch.setattr(module, "get_drift_monitor", lambda: SimpleNamespace(
        observe=lambda features: None, report=lambda: {"status": "UNAVAILABLE"}))
    yield SimpleNamespace(app=module.app, engine=engine, scored=scored, dispatched=dispatched,
                          statements=statements)
    engine.dispose()


def prediction_body(size):
    payload = b'{"transaction_id":"synthetic-body-limit","features":[1,-1,2]}'
    return payload + b" " * (size - len(payload))


def post_chunks(app, chunks, content_length=None):
    async def request():
        async def stream():
            for chunk in chunks:
                yield chunk

        headers = {"Content-Type": "application/json"}
        if content_length is not None:
            headers["Content-Length"] = content_length
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/predict", headers=headers, content=stream())

    return asyncio.run(request())


@pytest.mark.parametrize("content_length", [None, "1", "65537"])
def test_oversized_prediction_rejected_without_scoring_storage_or_dispatch(prediction_app, content_length):
    # Removing the byte gate must permit these valid but oversized JSON requests.
    body = prediction_body(65537)
    response = post_chunks(prediction_app.app, [body[:32000], body[32000:]], content_length)
    assert response.status_code == 413
    assert "detail" in response.json()
    assert response.headers["X-Correlation-ID"]
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_oversized_malformed_json_rejected_before_json_validation(prediction_app):
    # A JSON validation response would show that the parser ran before the gate.
    response = post_chunks(prediction_app.app, [b"{" * 65537])
    assert response.status_code == 413
    assert response.headers["X-Correlation-ID"]
    assert prediction_app.statements == []


def test_auth_off_oversized_body_retains_correlation_without_no_store(prediction_app, monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_MODE", "off")
    monkeypatch.delenv("DEMO_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("DEMO_AUTH_PASSWORD", raising=False)
    monkeypatch.setattr(prediction_app.app, "middleware_stack", None)
    response = post_chunks(prediction_app.app, [prediction_body(65537)])
    assert response.status_code == 413
    uuid.UUID(response.headers["x-correlation-id"])
    assert "cache-control" not in response.headers
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_prediction_limit_counts_utf8_bytes(prediction_app):
    # This body has fewer than 64 Ki characters but more than 64 KiB on the wire.
    body = json.dumps({"features": [1, -1, 2], "ignored": "é" * 33000}, ensure_ascii=False).encode()
    response = post_chunks(prediction_app.app, [body])
    assert response.status_code == 413
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_declared_oversized_prediction_does_not_read_body(prediction_app):
    chunks_read = []

    async def request():
        async def stream():
            chunks_read.append(True)
            yield prediction_body(100)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prediction_app.app),
                                     base_url="http://test") as client:
            return await client.post("/predict", headers={"Content-Length": "65537"}, content=stream())

    response = asyncio.run(request())
    assert response.status_code == 413
    assert chunks_read == []
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_streamed_oversized_prediction_stops_reading_at_limit(prediction_app):
    chunks_read = []

    async def request():
        async def stream():
            for index, chunk in enumerate([prediction_body(65536), b" ", b"unread"]):
                chunks_read.append(index)
                yield chunk

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prediction_app.app),
                                     base_url="http://test") as client:
            return await client.post("/predict", headers={"Content-Type": "application/json"},
                                     content=stream())

    response = asyncio.run(request())
    assert response.status_code == 413
    assert chunks_read == [0, 1]
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_oversized_prediction_with_asgi_root_path_is_rejected(prediction_app):
    async def request():
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=prediction_app.app, root_path="/api"),
                base_url="http://test") as client:
            return await client.post("/api/predict", content=prediction_body(65537),
                                     headers={"Content-Type": "application/json"})

    response = asyncio.run(request())
    assert response.status_code == 413
    assert response.headers["X-Correlation-ID"]
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


@pytest.mark.parametrize("content_length", [None, "1", "65536"])
def test_prediction_exactly_at_limit_scores_and_persists(prediction_app, content_length):
    from db.models import Explanation

    body = prediction_body(65536)
    response = post_chunks(prediction_app.app, [body[:32000], body[32000:]], content_length)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] == 0
    assert result["score"] == 0.25
    assert response.headers["X-Correlation-ID"] == result["correlation_id"]
    with prediction_app.engine.connect() as connection:
        row = connection.execute(select(Explanation.status).where(
            Explanation.transaction_id == result["transaction_id"])).scalar_one()
    assert row == "PENDING"
    assert prediction_app.scored == [[1, -1, 2]]
    assert len(prediction_app.dispatched) == 1


def test_small_invalid_prediction_keeps_validation_response(prediction_app):
    response = post_chunks(prediction_app.app, [b'{"features":["invalid"]}'])
    assert response.status_code == 422
    assert response.headers["X-Correlation-ID"]
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_disconnected_prediction_does_not_process_partial_body(prediction_app):
    async def request():
        messages = iter([
            {"type": "http.request", "body": prediction_body(100), "more_body": True},
            {"type": "http.disconnect"},
        ])
        responses = []

        async def receive():
            return next(messages, {"type": "http.disconnect"})

        async def send(message):
            responses.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/predict",
                 "raw_path": b"/predict", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")],
                 "server": ("test", 80), "client": ("test", 1234)}
        await prediction_app.app(scope, receive, send)
        return responses

    responses = asyncio.run(request())
    assert responses[0]["status"] == 400
    assert prediction_app.scored == []
    assert prediction_app.statements == []
    assert prediction_app.dispatched == []


def test_request_limit_keeps_health_readiness_and_demo_available(prediction_app):
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prediction_app.app),
                                     base_url="http://test") as client:
            return [await client.get(path) for path in ("/health", "/ready", "/demo/")]

    health, readiness, demo = asyncio.run(request())
    assert health.status_code == 200
    assert health.json()["status"] == "UP"
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "OK"
    assert demo.status_code == 200
    assert "text/html" in demo.headers["Content-Type"]
