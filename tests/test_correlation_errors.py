"""Unhandled HTTP error headers with fixed doubles; no model fitting or services."""
import asyncio
import importlib
from types import SimpleNamespace
import uuid

from fastapi import Request
from fastapi.testclient import TestClient
import httpx
import pytest


@pytest.fixture
def api_module(monkeypatch):
    module = importlib.import_module("api.app")
    # Avoid startup model loading and any external broker use.
    monkeypatch.setattr(module, "BUNDLE", SimpleNamespace(
        columns=["Time", "V1", "Amount"], source="synthetic-fixed-double",
        explanation_columns=["Time", "V1", "Amount"]))
    monkeypatch.setattr(module, "check_broker", lambda: False)
    return module


def test_unexpected_model_info_error_keeps_generic_500_and_correlation_header(api_module, monkeypatch):
    # The real route reads bundle.columns; an unexpected property error must not
    # bypass the API's per-request correlation header or expose private detail.
    class BrokenBundle:
        @property
        def columns(self):
            raise RuntimeError("SYNTHETIC_PRIVATE_DEBUG_CONTEXT")

    monkeypatch.setattr(api_module, "BUNDLE", BrokenBundle())
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        response = client.get("/model-info", headers={"X-Correlation-ID": "untrusted-client-value"})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert "SYNTHETIC_PRIVATE_DEBUG_CONTEXT" not in response.text
    assert "X-Correlation-ID" in response.headers
    correlation = response.headers["X-Correlation-ID"]
    assert str(uuid.UUID(correlation)) == correlation
    assert correlation != "untrusted-client-value"


def test_unexpected_error_reuses_the_request_id(api_module):
    seen = []

    async def failing(request: Request):
        seen.append(request.state.correlation_id)
        raise RuntimeError("synthetic unexpected endpoint failure")

    api_module.app.add_api_route("/_test/correlation-error", failing, methods=["GET"])
    route = api_module.app.router.routes[-1]
    try:
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            responses = [client.get("/_test/correlation-error") for _ in range(2)]
        assert all(response.status_code == 500 for response in responses)
        assert [response.headers["X-Correlation-ID"] for response in responses] == seen
        assert seen[0] != seen[1]
    finally:
        api_module.app.router.routes.remove(route)


def test_error_before_request_id_assignment_gets_a_safe_id(api_module):
    request = Request({"type": "http", "method": "GET", "path": "/model-info", "headers": [], "state": {}})
    response = asyncio.run(api_module.unexpected_error(request, RuntimeError("SYNTHETIC_PRIVATE_DETAIL")))
    assert response.status_code == 500
    assert response.body == b"Internal Server Error"
    correlation = response.headers["X-Correlation-ID"]
    assert str(uuid.UUID(correlation)) == correlation


@pytest.mark.parametrize("status", [422, 503])
def test_handled_errors_keep_their_body_and_correlation_header(api_module, monkeypatch, status):
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        if status == 422:
            response = client.post("/predict", json={"features": ["not-a-number"]})
            assert isinstance(response.json()["detail"], list)
        else:
            monkeypatch.setattr(api_module, "BUNDLE", None)
            response = client.get("/model-info")
            assert response.json() == {"detail": "Model is unavailable"}
    assert response.status_code == status
    assert str(uuid.UUID(response.headers["X-Correlation-ID"])) == response.headers["X-Correlation-ID"]


def test_unexpected_error_still_propagates_to_server_or_debug_client(api_module, monkeypatch):
    class BrokenBundle:
        @property
        def columns(self):
            raise RuntimeError("synthetic unhandled error")

    monkeypatch.setattr(api_module, "BUNDLE", BrokenBundle())
    with TestClient(api_module.app, raise_server_exceptions=True) as client:
        with pytest.raises(RuntimeError, match="synthetic unhandled error"):
            client.get("/model-info")


def test_cancelled_request_is_not_converted_to_an_error_response(api_module):
    async def exercise():
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def waiting(request: Request):
            entered.set()
            await blocked.wait()

        api_module.app.add_api_route("/_test/correlation-cancel", waiting, methods=["GET"])
        route = api_module.app.router.routes[-1]
        try:
            transport = httpx.ASGITransport(app=api_module.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                pending = asyncio.create_task(client.get("/_test/correlation-cancel"))
                await asyncio.wait_for(entered.wait(), timeout=2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        finally:
            api_module.app.router.routes.remove(route)

    asyncio.run(exercise())
