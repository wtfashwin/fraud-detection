"""Real loopback HTTP checks of the smoke client, not fraud-model quality."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from scripts import smoke_test
from scripts.smoke_test import smoke


@contextmanager
def concurrent_api(*, statuses=(200, 409), completes=True, wrong_id=None, slow_body=False):
    """Require both HTTP submissions before returning either prediction."""
    barrier = threading.Barrier(2, timeout=3)
    lock = threading.Lock()
    transactions = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if slow_body:
                try:
                    for byte in data:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.wfile.write(data)

        def do_GET(self):
            if self.path == "/demo/":
                self.reply(200, "synthetic page", "text/html")
            elif self.path == "/demo/app.mjs":
                self.reply(200, "synthetic asset", "text/javascript")
            elif self.path == "/ready":
                self.reply(200, {"status": "OK"})
            elif self.path == "/model-info":
                self.reply(200, {"feature_count": 3, "feature_names": ["Time", "V1", "Amount"]})
            elif self.path.startswith("/explain/"):
                transaction_id = self.path.rsplit("/", 1)[1]
                if not completes:
                    self.reply(202, {"transaction_id": transaction_id, "status": "PENDING"})
                else:
                    self.reply(200, {"transaction_id": "unrelated" if wrong_id == "explanation" else transaction_id,
                                     "status": "COMPLETED", "feature_names": ["Time", "V1", "Amount"],
                                     "shap_values": [0.0, 0.0, 0.0], "base_value": 0.0,
                                     "explanation_method": "linear_log_odds_zero_reference"})
            else:
                self.reply(404, {"detail": "unknown path"})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if any(isinstance(value, bool) for value in payload["features"]):
                self.reply(422, {"detail": "invalid features"})
                return
            with lock:
                index = len(transactions)
                transactions.append(payload["transaction_id"])
            if index < 2:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    self.reply(503, {"detail": "client did not submit both requests concurrently"})
                    return
                status = statuses[index]
            else:
                status = 409
            body = ({"transaction_id": "unrelated" if wrong_id == "prediction" else payload["transaction_id"],
                     "prediction": 0, "score": 0.5,
                     "correlation_id": "synthetic-correlation", "explanation_status": "Calculation queued"}
                    if status == 200 else {"detail": "duplicate ID"})
            self.reply(status, body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", transactions
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("statuses", [(200, 409), (409, 200)])
def test_smoke_submits_same_id_concurrently_and_saves_both_responses(statuses):
    with concurrent_api(statuses=statuses) as (base, transactions):
        result = smoke(base)
    responses = result["concurrent_prediction_responses"]
    assert sorted(item["status_code"] for item in responses) == [200, 409]
    assert result["status"] == "PASS"
    assert len(transactions) >= 2
    assert set(transactions) == {result["transaction_id"]}
    assert result["explanation_response"]["transaction_id"] == result["transaction_id"]


@pytest.mark.parametrize("statuses", [(200, 200), (409, 409), (200, 503)])
def test_smoke_rejects_broken_duplicate_contract(statuses):
    with concurrent_api(statuses=statuses) as (base, _):
        with pytest.raises(AssertionError):
            smoke(base)


def test_smoke_requires_worker_completion_after_concurrent_acceptance():
    with concurrent_api(completes=False) as (base, _):
        with pytest.raises(TimeoutError, match="did not complete"):
            smoke(base, timeout=0)


@pytest.mark.parametrize("wrong_id", ["prediction", "explanation"])
def test_smoke_rejects_response_for_another_transaction(wrong_id):
    with concurrent_api(wrong_id=wrong_id) as (base, _):
        with pytest.raises(AssertionError):
            smoke(base)


def test_response_body_deadline_stops_continuously_trickling_bytes(monkeypatch):
    # A socket inactivity timeout alone never fires while bytes keep arriving.
    monkeypatch.setattr(smoke_test, "HTTP_TIMEOUT", 0.08, raising=False)
    with concurrent_api(slow_body=True) as (base, _):
        with pytest.raises(TimeoutError):
            smoke_test.request(base, "/ready")
