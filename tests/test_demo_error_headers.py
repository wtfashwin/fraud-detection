"""Unexpected responses are tested in process with invented credentials."""
import asyncio
import base64
import importlib
import uuid

import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse


USERNAME = "synthetic-error-user"
PASSWORD = "synthetic-error-password"
AUTH = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
FAILURE_PATH = "/__synthetic_unexpected_failure__"


@pytest.fixture
def app_with_failure(monkeypatch):
    module = importlib.import_module("api.app")
    observed = []

    async def fail(request: Request):
        observed.append(request.state.correlation_id)
        raise RuntimeError("synthetic internal detail")

    route = APIRoute(FAILURE_PATH, fail, methods=["GET"])
    monkeypatch.setattr(module.app.router, "routes", [*module.app.router.routes, route])
    monkeypatch.setattr(module.app, "middleware_stack", None)
    monkeypatch.setattr(module, "BUNDLE", object())  # do not load model artifacts
    for name in ("DEMO_AUTH_MODE", "DEMO_AUTH_USERNAME", "DEMO_AUTH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    return module.app, observed


def enable_basic(monkeypatch):
    monkeypatch.setenv("DEMO_AUTH_MODE", "basic")
    monkeypatch.setenv("DEMO_AUTH_USERNAME", USERNAME)
    monkeypatch.setenv("DEMO_AUTH_PASSWORD", PASSWORD)


def test_basic_unexpected_error_keeps_no_store_and_request_correlation(app_with_failure, monkeypatch):
    app, observed = app_with_failure
    enable_basic(monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(FAILURE_PATH, headers={"Authorization": AUTH})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers.get_list("cache-control") == ["no-store"]
    assert len(observed) == 1
    assert response.headers["x-correlation-id"] == observed[0]
    assert "synthetic internal detail" not in response.text


def test_basic_error_before_correlation_still_has_safe_headers(app_with_failure, monkeypatch):
    app, observed = app_with_failure
    enable_basic(monkeypatch)

    class FailBeforeCorrelation:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                raise RuntimeError("synthetic outer failure")
            await self.app(scope, receive, send)

    monkeypatch.setattr(app, "user_middleware", [Middleware(FailBeforeCorrelation), *app.user_middleware])
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(FAILURE_PATH)
    assert observed == []
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers.get_list("cache-control") == ["no-store"]
    uuid.UUID(response.headers["x-correlation-id"])


@pytest.mark.parametrize("initial_mode", ["basic", "off"])
def test_error_cache_policy_uses_active_mode_when_environment_changes(app_with_failure, monkeypatch, initial_mode):
    app, _ = app_with_failure
    if initial_mode == "basic":
        enable_basic(monkeypatch)
    else:
        monkeypatch.setenv("DEMO_AUTH_MODE", "off")
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/status").status_code == 200  # ensure the stack is built
        if initial_mode == "basic":
            monkeypatch.setenv("DEMO_AUTH_MODE", "off")
            monkeypatch.delenv("DEMO_AUTH_USERNAME")
            monkeypatch.delenv("DEMO_AUTH_PASSWORD")
        else:
            enable_basic(monkeypatch)
        response = client.get(FAILURE_PATH, headers={"Authorization": AUTH})
    assert response.status_code == 500
    assert response.headers.get("cache-control") == ("no-store" if initial_mode == "basic" else None)


def test_off_mode_keeps_existing_generic_error_response(app_with_failure, monkeypatch):
    app, observed = app_with_failure
    monkeypatch.setenv("DEMO_AUTH_MODE", "off")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(FAILURE_PATH)
    assert len(observed) == 1
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "cache-control" not in response.headers
    assert "x-correlation-id" not in response.headers


def test_application_state_contains_only_the_effective_mode(app_with_failure, monkeypatch):
    app, _ = app_with_failure
    enable_basic(monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/status").status_code == 200
        assert app.state.demo_auth_mode == "basic"
        assert USERNAME not in repr(app.state._state)
        assert PASSWORD not in repr(app.state._state)


def test_gate_can_still_be_constructed_without_application_state(monkeypatch):
    from api.demo_auth import DemoAccessGateMiddleware

    enable_basic(monkeypatch)
    downstream = Starlette()
    downstream.add_route("/private", lambda request: PlainTextResponse("synthetic"))
    gate = DemoAccessGateMiddleware(downstream)
    with TestClient(gate) as client:
        denied = client.get("/private")
        allowed = client.get("/private", headers={"Authorization": AUTH})
    assert denied.status_code == 401
    assert denied.headers["cache-control"] == "no-store"
    assert allowed.status_code == 200
    assert allowed.text == "synthetic"
    assert allowed.headers["cache-control"] == "no-store"


def test_error_handler_accepts_the_existing_pr27_minimal_request(app_with_failure, monkeypatch):
    app, _ = app_with_failure
    enable_basic(monkeypatch)
    module = importlib.import_module("api.app")
    request = Request({"type": "http", "method": "GET", "path": "/synthetic", "headers": [], "state": {}})
    with TestClient(app, raise_server_exceptions=False):
        response = asyncio.run(module.unexpected_error(request, RuntimeError("synthetic failure")))
    assert response.status_code == 500
    assert response.body == b"Internal Server Error"
    assert response.headers["cache-control"] == "no-store"
    uuid.UUID(response.headers["x-correlation-id"])


def test_error_handler_defaults_to_no_store_when_active_mode_is_missing(app_with_failure, monkeypatch):
    app, _ = app_with_failure
    module = importlib.import_module("api.app")
    monkeypatch.delattr(app.state, "demo_auth_mode", raising=False)
    request = Request({"type": "http", "method": "GET", "path": "/synthetic", "headers": [], "state": {}})
    response = asyncio.run(module.unexpected_error(request, RuntimeError("synthetic failure")))
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    uuid.UUID(response.headers["x-correlation-id"])
