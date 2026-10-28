"""Access boundary tests use invented credentials, fixed scoring and SQLite only."""
import asyncio
import base64
import importlib
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, insert, select
from sqlalchemy.pool import StaticPool


USERNAME = "synthetic-demo"
PASSWORD = "synthetic:password-for-local-tests-only"
AUTH = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()


@pytest.fixture
def protected_app(monkeypatch):
    module = importlib.import_module("api.app")
    from db.models import Explanation
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Explanation.__table__.create(engine)
    scored, dispatched, statements = [], [], []

    def score(features):
        scored.append(features)
        return 0, 0.25

    event.listen(engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    monkeypatch.setenv("DEMO_AUTH_MODE", "basic")
    monkeypatch.setenv("DEMO_AUTH_USERNAME", USERNAME)
    monkeypatch.setenv("DEMO_AUTH_PASSWORD", PASSWORD)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace(columns=["Time", "V1", "Amount"],
        explanation_columns=["Time", "V1", "Amount"], source="synthetic-fixed", score=score))
    monkeypatch.setattr(module, "check_broker", lambda: True)
    monkeypatch.setattr(module.celery_app, "send_task", lambda *args, **kwargs: dispatched.append((args, kwargs)))
    monkeypatch.setattr(module, "get_drift_monitor", lambda: SimpleNamespace(
        observe=lambda features: None, report=lambda: {"status": "UNAVAILABLE", "samples": None, "features": {}}))
    # Build a fresh middleware stack without replacing the real routes/static mount.
    monkeypatch.setattr(module.app, "middleware_stack", None)
    with TestClient(module.app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, module=module, engine=engine,
                              scored=scored, dispatched=dispatched, statements=statements)
    engine.dispose()


@pytest.mark.parametrize("method,path", [
    ("GET", "/demo"), ("GET", "/demo/"), ("GET", "/demo/app.mjs"),
    ("GET", "/demo/api-client.mjs"), ("GET", "/demo/styles.css"),
    ("POST", "/predict"), ("GET", "/explain/synthetic-absent"),
    ("GET", "/model-info"), ("GET", "/health"), ("GET", "/ready"),
    ("GET", "/metrics"), ("GET", "/docs"), ("GET", "/redoc"),
    ("GET", "/openapi.json"), ("GET", "/unknown"), ("GET", "/status/"),
    ("HEAD", "/status"), ("OPTIONS", "/status"), ("GET", "/Status"),
    ("POST", "/status"), ("GET", "/status-extra"),
])
def test_anonymous_request_cannot_reach_protected_surface(protected_app, method, path):
    response = protected_app.client.request(method, path, follow_redirects=False)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Correlation-ID"]
    assert protected_app.scored == []
    assert protected_app.dispatched == []
    assert protected_app.statements == []


def test_exact_get_status_is_minimal_liveness(protected_app):
    response = protected_app.client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
    assert protected_app.statements == []


@pytest.mark.parametrize("headers", [
    [], [("Authorization", "Bearer synthetic")], [("Authorization", "Basic !!!")],
    [("Authorization", "Basic Zm9v")], [("Authorization", "Basic ")],
    [("Authorization", "Basic " + base64.b64encode(b"wrong:wrong").decode())],
    [("Authorization", AUTH), ("Authorization", AUTH)],
    [("Authorization", AUTH), ("Authorization", "Bearer other")],
    [("Authorization", "Basic " + base64.b64encode(b"synthetic-demo:\xff").decode())],
])
def test_malformed_or_duplicate_headers_have_same_denial(protected_app, headers, caplog):
    response = protected_app.client.get("/model-info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
    assert PASSWORD not in response.text + caplog.text
    assert AUTH not in response.text + caplog.text


def test_authenticated_assets_metadata_and_metrics_are_available(protected_app):
    for path in ["/demo/", "/demo/app.mjs", "/model-info", "/health", "/ready", "/docs", "/openapi.json", "/metrics"]:
        response = protected_app.client.get(path, headers={"Authorization": AUTH})
        assert response.status_code == 200, (path, response.text)
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Correlation-ID"]
    assert "predictions_submitted_total" in protected_app.client.get("/metrics", headers={"Authorization": AUTH}).text


def test_authenticated_prediction_and_saved_status_contract(protected_app):
    from db.models import Explanation
    client = protected_app.client
    response = client.post("/predict", json={"transaction_id": "synthetic-auth-predict", "features": [1, -1, 2]},
                           headers={"Authorization": AUTH})
    assert response.status_code == 200
    assert response.json()["score"] == 0.25
    assert protected_app.scored == [[1, -1, 2]]
    assert len(protected_app.dispatched) == 1
    with protected_app.engine.begin() as connection:
        assert connection.execute(select(Explanation.status).where(
            Explanation.transaction_id == "synthetic-auth-predict")).scalar_one() == "PENDING"
        connection.execute(insert(Explanation), [
            {"transaction_id": "synthetic-auth-failed", "correlation_id": "synthetic-correlation", "status": "FAILED"},
            {"transaction_id": "synthetic-auth-completed", "correlation_id": "synthetic-correlation", "status": "COMPLETED"},
        ])
    for transaction_id, expected in [("synthetic-auth-predict", 202), ("synthetic-auth-failed", 503),
                                      ("synthetic-auth-completed", 200), ("synthetic-absent", 404)]:
        response = client.get("/explain/" + transaction_id, headers={"Authorization": AUTH})
        assert response.status_code == expected
    assert client.get("/explain/synthetic-auth-completed").status_code == 401


def test_unauthorized_stream_is_not_consumed(protected_app):
    reads = []

    async def exercise():
        async def stream():
            reads.append(True)
            yield b"{" * 70000
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=protected_app.module.app), base_url="http://test") as client:
            return await client.post("/predict", content=stream())

    response = asyncio.run(exercise())
    assert response.status_code == 401
    assert reads == []
    assert protected_app.statements == []
    assert protected_app.scored == []
    assert protected_app.dispatched == []


def test_unauthorized_declared_oversized_body_is_denied_without_reading(protected_app):
    reads = []

    async def exercise():
        async def stream():
            reads.append(True)
            yield b"{" * 70000

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=protected_app.module.app),
                                     base_url="http://test") as client:
            return await client.post("/predict", headers={"Content-Length": "65537"}, content=stream())

    response = asyncio.run(exercise())
    assert response.status_code == 401
    assert response.headers.get_list("cache-control") == ["no-store"]
    uuid.UUID(response.headers["x-correlation-id"])
    assert reads == []
    assert protected_app.statements == []
    assert protected_app.scored == []
    assert protected_app.dispatched == []


def test_authorized_oversized_body_is_rejected_before_side_effects(protected_app):
    prefix = b'{"transaction_id":"synthetic-bounded","features":[1,-1,2]}'
    body = prefix + b" " * (65537 - len(prefix))
    response = protected_app.client.post("/predict", content=body,
        headers={"Authorization": AUTH, "Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.headers.get_list("cache-control") == ["no-store"]
    uuid.UUID(response.headers["x-correlation-id"])
    assert protected_app.statements == []
    assert protected_app.scored == []
    assert protected_app.dispatched == []


@pytest.mark.parametrize("content_length", [None, "1"])
@pytest.mark.parametrize("root_path", ["", "/api"])
def test_authenticated_oversized_stream_stops_before_side_effects(protected_app, content_length, root_path):
    reads = []
    prefix = b'{"transaction_id":"synthetic-stream","features":[1,-1,2]}'

    async def exercise():
        async def stream():
            chunks = [prefix + b" " * (65536 - len(prefix)), b" ", b" "]
            for index, chunk in enumerate(chunks):
                reads.append(index)
                yield chunk

        headers = {"Authorization": AUTH, "Content-Type": "application/json"}
        if content_length is not None:
            headers["Content-Length"] = content_length
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=protected_app.module.app, root_path=root_path),
                base_url="http://test") as client:
            return await client.post(root_path + "/predict", headers=headers, content=stream())

    submitted_before = protected_app.module.predictions_submitted._value.get()
    response = asyncio.run(exercise())
    assert response.status_code == 413
    assert response.headers.get_list("cache-control") == ["no-store"]
    uuid.UUID(response.headers["x-correlation-id"])
    assert reads == [0, 1]
    assert protected_app.statements == []
    assert protected_app.scored == []
    assert protected_app.dispatched == []
    assert protected_app.module.predictions_submitted._value.get() == submitted_before


def test_application_root_path_does_not_bypass_gate(protected_app):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=protected_app.module.app, root_path="/api"),
                                     base_url="http://test") as client:
            return [await client.get(path) for path in ["/api/status", "/api/status/", "/api/demo/", "/api/explain/synthetic"]]
    responses = asyncio.run(exercise())
    assert [response.status_code for response in responses] == [200, 401, 401, 401]


@pytest.mark.parametrize("settings", [
    {"DEMO_AUTH_MODE": "typo"}, {"DEMO_AUTH_MODE": "BASIC"}, {"DEMO_AUTH_MODE": ""},
    {"DEMO_AUTH_MODE": "basic"},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": USERNAME},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_PASSWORD": PASSWORD},
    {"DEMO_AUTH_MODE": "off", "DEMO_AUTH_USERNAME": USERNAME, "DEMO_AUTH_PASSWORD": PASSWORD},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": "bad:user", "DEMO_AUTH_PASSWORD": PASSWORD},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": "bad\nuser", "DEMO_AUTH_PASSWORD": PASSWORD},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": "   ", "DEMO_AUTH_PASSWORD": PASSWORD},
    {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": USERNAME, "DEMO_AUTH_PASSWORD": "   "},
])
def test_invalid_configuration_prevents_startup(monkeypatch, settings):
    module = importlib.import_module("api.app")
    for name in ["DEMO_AUTH_MODE", "DEMO_AUTH_USERNAME", "DEMO_AUTH_PASSWORD"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(module.app, "middleware_stack", None)
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace())
    with pytest.raises(ValueError, match="Demo authentication"):
        with TestClient(module.app):
            pass


def test_unset_mode_preserves_local_developer_access(monkeypatch):
    module = importlib.import_module("api.app")
    for name in ["DEMO_AUTH_MODE", "DEMO_AUTH_USERNAME", "DEMO_AUTH_PASSWORD"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module.app, "middleware_stack", None)
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace())
    with TestClient(module.app) as client:
        assert client.get("/demo/").status_code == 200
        assert client.get("/status").json() == {"status": "UP"}


def test_uvicorn_auto_lifespan_recognizes_configuration_failure(monkeypatch):
    from uvicorn import Config
    from uvicorn.lifespan.on import LifespanOn
    module = importlib.import_module("api.app")
    monkeypatch.setenv("DEMO_AUTH_MODE", "invalid-mode")
    monkeypatch.setattr(module.app, "middleware_stack", None)
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace())

    async def exercise():
        lifecycle = LifespanOn(Config(module.app, lifespan="auto", ws="none", log_config=None))
        await lifecycle.startup()
        if not lifecycle.should_exit:
            await lifecycle.shutdown()
        return lifecycle

    lifecycle = asyncio.run(exercise())
    assert lifecycle.startup_failed
    assert lifecycle.should_exit


def test_invalid_configuration_cannot_serve_when_lifespan_is_disabled(monkeypatch):
    module = importlib.import_module("api.app")
    monkeypatch.setenv("DEMO_AUTH_MODE", "invalid-mode")
    monkeypatch.setattr(module.app, "middleware_stack", None)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app, raise_app_exceptions=False),
                                     base_url="http://test") as client:
            return await client.get("/status")

    response = asyncio.run(exercise())
    assert response.status_code == 503
    assert response.json() == {"detail": "Demo authentication configuration is invalid"}
    assert response.headers["Cache-Control"] == "no-store"
