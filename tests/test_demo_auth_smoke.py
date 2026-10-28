"""Authenticated smoke behavior uses invented loopback responses, not a worker."""
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
USERNAME = "synthetic-http-user"
PASSWORD = "synthetic-http-password-only"
AUTH = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()


def smoke_module(monkeypatch):
    target = ROOT / "scripts/smoke_demo_auth.py"
    assert target.exists(), "Authenticated smoke client has not been implemented"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("auth_smoke_under_test", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def synthetic_server(*, leak=False, explanation="COMPLETED", explanation_delay=0,
        redirect=False, accept_any_password=False, slow_headers=False, slow_body=False, truncated_asset=False):
    state = {"accepted": None, "events": [], "polls": 0}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json", location=None):
            raw = json.dumps(body).encode() if content_type == "application/json" else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            extra = 8 if truncated_asset and code == 200 and self.path == "/demo/app.mjs" else 0
            self.send_header("Content-Length", str(len(raw) + extra))
            if code == 401:
                self.send_header("WWW-Authenticate", 'Basic realm="Synthetic"')
            if location:
                self.send_header("Location", location)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except BrokenPipeError:
                pass

        def handle_request(self):
            admitted = self.headers.get("Authorization") == AUTH
            if accept_any_password:
                encoded = self.headers.get("Authorization", "").removeprefix("Basic ")
                try:
                    admitted = base64.b64decode(encoded).decode().split(":", 1)[0] == USERNAME
                except (ValueError, UnicodeError):
                    admitted = False
            state["events"].append((self.command, self.path, admitted))
            if self.path in ["/chunked-body", "/close-body"]:
                header = b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n'
                tail = (b'Transfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n'
                    if self.path == "/chunked-body" else b'Connection: close\r\n\r\nabc')
                self.connection.sendall(header + tail)
                return
            if (slow_headers and self.path == "/slow-headers") or (slow_body and self.path == "/slow-body"):
                body = b'"' + b'x' * 64 + b'"'
                headers = (b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                    + str(len(body)).encode() + b'\r\n\r\n')
                try:
                    if slow_body:
                        self.connection.sendall(headers)
                    for byte in headers + body if slow_headers else body:
                        self.connection.sendall(bytes([byte]))
                        time.sleep(0.025)
                except OSError:
                    pass
                return
            if self.path == "/status":
                return self.reply(200, {"status": "UP"})
            if not admitted:
                return self.reply(200 if leak else 401, {"detail": "Authentication required"})
            if redirect:
                return self.reply(302, {}, location="http://127.0.0.1:9/credential-trap")
            if self.path == "/demo/":
                return self.reply(200, b"<html>Invented demo</html>", "text/html")
            if self.path in ["/demo/app.mjs", "/demo/api-client.mjs"]:
                return self.reply(200, b"export const invented = true;", "text/javascript")
            if self.path == "/demo/styles.css":
                return self.reply(200, b"body { color: black; }", "text/css")
            if self.path == "/ready":
                return self.reply(200, {"status": "OK", "dependencies": {"model": "UP", "postgres": "UP", "redis_broker": "UP"}})
            if self.path == "/model-info":
                return self.reply(200, {"feature_count": 3, "feature_names": ["Time", "V1", "Amount"],
                    "explanation_feature_names": ["Time", "V1", "Amount"]})
            if self.command == "POST" and self.path == "/predict":
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["accepted"] = data["transaction_id"]
                return self.reply(200, {"transaction_id": state["accepted"], "prediction": 0, "score": 0.25,
                    "correlation_id": "synthetic-correlation", "explanation_status": "Calculation queued"})
            if self.path.startswith("/explain/"):
                if state["accepted"] is None:
                    return self.reply(404, {"detail": "Missing"})
                state["polls"] += 1
                if explanation_delay:
                    time.sleep(explanation_delay)
                if state["polls"] == 1 and explanation == "COMPLETED" and not explanation_delay:
                    return self.reply(202, {"transaction_id": state["accepted"], "status": "PENDING"})
                return self.reply(200, {"transaction_id": state["accepted"], "status": explanation,
                    "feature_names": ["Time", "V1", "Amount"], "shap_values": [0.1, 0.2, 0.3], "base_value": 0.0})
            return self.reply(404, {})

        do_GET = handle_request
        do_POST = handle_request

        def log_message(self, *args):
            pass

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}", state
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


def test_real_loopback_flow_requires_denial_assets_and_completed_explanation(monkeypatch, capsys):
    module = smoke_module(monkeypatch)
    with synthetic_server() as (url, state):
        report = module.smoke(url, USERNAME, PASSWORD, timeout=2)
    assert report["status"] == "PASS"
    assert report["denied_prediction_saved_status"] == 404
    assert report["explanation_response"]["status"] == "COMPLETED"
    assert report["pending_observations"] == 1
    assert state["polls"] == 2
    assert ("POST", "/predict", False) in state["events"]
    assert ("POST", "/predict", True) in state["events"]
    for path in ["/demo/", "/demo/app.mjs", "/demo/api-client.mjs", "/demo/styles.css", "/ready"]:
        assert ("GET", path, True) in state["events"]
    output = json.dumps(report) + capsys.readouterr().out
    assert all(secret not in output for secret in [USERNAME, PASSWORD, AUTH])


@pytest.mark.parametrize("settings", [{"leak": True}, {"explanation": "FAILED"}, {"redirect": True}])
def test_false_or_redirected_success_is_rejected(monkeypatch, settings):
    module = smoke_module(monkeypatch)
    with synthetic_server(**settings) as (url, _):
        with pytest.raises(module.SmokeFailure):
            module.smoke(url, USERNAME, PASSWORD, timeout=1)


def test_completion_observed_after_deadline_is_rejected(monkeypatch):
    module = smoke_module(monkeypatch)
    with synthetic_server(explanation_delay=0.05) as (url, _):
        with pytest.raises(module.SmokeFailure):
            module.smoke(url, USERNAME, PASSWORD, timeout=0.01)


def test_server_ignoring_password_is_rejected(monkeypatch):
    module = smoke_module(monkeypatch)
    with synthetic_server(accept_any_password=True) as (url, _):
        with pytest.raises(module.SmokeFailure, match="password"):
            module.smoke(url, USERNAME, PASSWORD, timeout=1)


def test_truncated_demo_asset_cannot_count_as_success(monkeypatch):
    module = smoke_module(monkeypatch)
    with synthetic_server(truncated_asset=True) as (url, state):
        try:
            with pytest.raises(module.SmokeFailure, match="declared length"):
                module.smoke(url, USERNAME, PASSWORD, timeout=1)
        finally:
            assert ("GET", "/demo/app.mjs", True) in state["events"]


@pytest.mark.parametrize("path", ["/chunked-body", "/close-body"])
def test_complete_response_without_content_length_is_supported(monkeypatch, path):
    module = smoke_module(monkeypatch)
    with synthetic_server() as (url, _):
        code, _, body = module.Client(url, USERNAME, PASSWORD).request(path)
    assert (code, body) == (200, b"abc")


@pytest.mark.parametrize("part", ["headers", "body"])
def test_trickled_response_cannot_extend_request_deadline(monkeypatch, part):
    module = smoke_module(monkeypatch)
    with synthetic_server(**{"slow_" + part: True}) as (url, _):
        client = module.Client(url, USERNAME, PASSWORD)
        start = time.monotonic()
        with pytest.raises(module.SmokeFailure):
            client.request("/slow-" + part, timeout=0.05)
        elapsed = time.monotonic() - start
    assert elapsed < 1, f"{part} reads exceeded the total budget: {elapsed:.3f}s"


@pytest.mark.parametrize("url", ["https://example.com", "http://localhost:8000", "http://user:password@127.0.0.1:8000", "http://127.0.0.1:8000/path"])
def test_credentials_cannot_be_sent_to_an_unapproved_destination(monkeypatch, url):
    module = smoke_module(monkeypatch)
    with pytest.raises(module.SmokeFailure):
        module.smoke(url, USERNAME, PASSWORD, timeout=1)


def test_explicit_port_zero_is_rejected_before_opening_a_connection(monkeypatch):
    module = smoke_module(monkeypatch)
    with pytest.raises(module.SmokeFailure):
        module.Client("http://127.0.0.1:0", USERNAME, PASSWORD)


def workflow_step():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-cd.yml").read_text())
    matches = [step for step in workflow["jobs"]["container"]["steps"] if step.get("name") == "Authenticated demo and queued explanation"]
    assert len(matches) == 1, "Hosted authenticated stack step has not been added"
    return matches[0]


@pytest.mark.parametrize("failure", ["", "up", "client", "down", "occupied", "not-hosted", "unsafe-project"])
def test_actual_workflow_block_is_isolated_and_cleans_only_its_project(tmp_path, failure):
    step = workflow_step()
    tools = tmp_path / "bin"
    tools.mkdir()
    trace = tmp_path / "trace.jsonl"
    shim = "#!" + sys.executable + "\n" + '''import json, os, sys
name=os.path.basename(sys.argv[0]); args=sys.argv[1:]
if name == "python3" and args[:1] == ["-c"]:
    print("SYNTHETIC_GENERATED_CREDENTIAL_ONLY"); raise SystemExit(0)
with open(os.environ["AUTH_TEST_TRACE"], "a") as f:
    f.write(json.dumps({"tool":name,"args":args,"port":os.getenv("API_PORT"),"mode":os.getenv("DEMO_AUTH_MODE"),
        "username_set":bool(os.getenv("DEMO_AUTH_USERNAME")),"password_set":bool(os.getenv("DEMO_AUTH_PASSWORD")),
        "profiles":os.getenv("COMPOSE_PROFILES")})+"\\n")
failure=os.getenv("AUTH_TEST_FAILURE")
if name == "docker" and "ps" in args and failure == "occupied": print("existing-container")
if name == "docker" and "up" in args and failure == "up": raise SystemExit(1)
if name == "docker" and "down" in args and failure == "down": raise SystemExit(1)
if name == "python3" and failure == "client": raise SystemExit(1)
'''
    for name in ["docker", "python3"]:
        p = tools / name
        p.write_text(shim)
        p.chmod(0o700)
    env = {**os.environ, **{k: str(v) for k, v in step["env"].items()},
        "PATH": str(tools) + os.pathsep + os.environ["PATH"], "RUNNER_ENVIRONMENT": "github-hosted",
        "AUTH_COMPOSE_PROJECT": "fraud-auth-1234-1", "AUTH_TEST_TRACE": str(trace), "AUTH_TEST_FAILURE": failure,
        "GITHUB_ENV": str(tmp_path / "github-env"), "GITHUB_REPOSITORY": "ordinary-fork/fraud-detection"}
    if failure == "not-hosted":
        env["RUNNER_ENVIRONMENT"] = "self-hosted"
    if failure == "unsafe-project":
        env["AUTH_COMPOSE_PROJECT"] = "fraud-detection"
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step["run"]], env=env, cwd=tmp_path, text=True, capture_output=True)
    assert (result.returncode == 0) == (failure == "")
    if failure in ["not-hosted", "unsafe-project"]:
        assert not trace.exists()
        return
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    docker = [call for call in calls if call["tool"] == "docker"]
    clients = [call for call in calls if call["tool"] == "python3"]
    assert len(clients) == (1 if failure in ["", "client", "down"] else 0)
    for client in clients:
        assert client["args"] == ["scripts/smoke_demo_auth.py", "--url", "http://127.0.0.1:18001",
            "--features-csv", "tests/fixtures/synthetic_transaction.csv", "--output", "reports/runtime/auth-http-smoke.json"]
    assert all("--project-name" in call["args"] and "fraud-auth-1234-1" in call["args"] for call in docker)
    assert all(call["port"] == "18001" and call["profiles"] == "" for call in calls)
    mutation_calls = [call for call in docker if "up" in call["args"] or "down" in call["args"]]
    if failure == "occupied":
        assert mutation_calls == []
    else:
        assert ["up" in call["args"] for call in mutation_calls] == [True, False]
        assert all(call["mode"] == "basic" and call["username_set"] and call["password_set"] for call in mutation_calls)
        assert "--no-build" in mutation_calls[0]["args"]
        assert "--volumes" in mutation_calls[-1]["args"]
    assert "SYNTHETIC_GENERATED_CREDENTIAL_ONLY" not in result.stdout + result.stderr + trace.read_text()
    assert not (tmp_path / "github-env").exists()
