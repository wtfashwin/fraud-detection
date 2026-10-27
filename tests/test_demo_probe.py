"""Probe subprocess checks use a short-lived synthetic loopback HTTP server."""
import base64
import errno
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
USERNAME = "synthetic-probe"
PASSWORD = "synthetic-probe-password-only"
AUTH = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()


@contextmanager
def server(port=8000, status=200, redirect=None, delay=0, malformed=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            time.sleep(delay)
            if malformed:
                self.wfile.write(b"SYNTHETIC_PRIVATE_BAD_STATUS_LINE\r\n")
                return
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.end_headers()
            try:
                self.wfile.write(b"SYNTHETIC_RESPONSE_MUST_NOT_BE_PRINTED")
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    try:
        service = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            pytest.skip(f"Loopback port {port} is already in use; probe check was not run")
        raise
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield service, requests
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


def run_probe(settings=None, args=()):
    env = dict(os.environ)
    for name in ["DEMO_AUTH_MODE", "DEMO_AUTH_USERNAME", "DEMO_AUTH_PASSWORD"]:
        env.pop(name, None)
    env.update(settings or {})
    return subprocess.run([sys.executable, "-m", "api.probe", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=7)


@pytest.mark.parametrize("secured", [False, True])
def test_probe_uses_fixed_ready_and_only_configured_credentials(secured):
    settings = {"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": USERNAME, "DEMO_AUTH_PASSWORD": PASSWORD} if secured else {}
    with server() as (_, requests):
        result = run_probe(settings)
    assert result.returncode == 0
    assert requests == [("/ready", AUTH if secured else None)]
    assert result.stdout == result.stderr == ""
    assert PASSWORD not in " ".join(result.args)


@pytest.mark.parametrize("status", [401, 503])
def test_probe_fails_without_printing_http_error_or_response(status):
    with server(status=status) as (_, requests):
        result = run_probe()
    assert result.returncode == 1
    assert len(requests) == 1
    assert result.stdout == result.stderr == ""


def test_invalid_probe_configuration_fails_before_http():
    with server() as (_, requests):
        result = run_probe({"DEMO_AUTH_MODE": "basic"})
    assert result.returncode == 1
    assert requests == []
    assert result.stdout == result.stderr == ""


def test_probe_does_not_forward_credentials_on_redirect_or_use_proxy():
    with server(port=0) as (trap, leaked):
        target = f"http://127.0.0.1:{trap.server_port}/trap"
        with server(status=302, redirect=target) as (_, requests):
            result = run_probe({"DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": USERNAME,
                "DEMO_AUTH_PASSWORD": PASSWORD, "HTTP_PROXY": target, "http_proxy": target, "NO_PROXY": "", "no_proxy": ""})
    assert result.returncode == 1
    assert len(requests) == 1
    assert leaked == []
    assert result.stdout == result.stderr == ""


def test_probe_times_out_without_response_data():
    with server(delay=4) as (_, requests):
        result = run_probe()
    assert result.returncode == 1
    assert len(requests) == 1
    assert result.stdout == result.stderr == ""


def test_probe_rejects_url_arguments_without_making_request():
    with server() as (_, requests):
        result = run_probe(args=("--url", "http://example.invalid/"))
    assert result.returncode == 1
    assert requests == []
    assert result.stdout == result.stderr == ""


def test_probe_fails_silently_on_malformed_http_response():
    with server(malformed=True) as (_, requests):
        result = run_probe()
    assert result.returncode == 1
    assert len(requests) == 1
    assert result.stdout == result.stderr == ""
