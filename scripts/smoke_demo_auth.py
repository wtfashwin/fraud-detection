"""HTTP proof for a synthetic authenticated stack; no browser or quality claim."""
import argparse
import base64
from http.client import HTTPConnection, HTTPException
import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.parse
import uuid

from smoke_test import csv_features


class SmokeFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


class DeadlineSocket(socket.socket):
    """Apply one total budget to connect, writes and buffered HTTP reads."""
    def __init__(self, deadline):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.deadline = deadline

    def remaining_timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP deadline expired")
        self.settimeout(remaining)

    def connect(self, address):
        self.remaining_timeout()
        return super().connect(address)

    def sendall(self, data, flags=0):
        self.remaining_timeout()
        return super().sendall(data, flags)

    def recv_into(self, buffer, nbytes=0, flags=0):
        self.remaining_timeout()
        return super().recv_into(buffer, nbytes, flags)


class DeadlineConnection(HTTPConnection):
    def __init__(self, host, port, deadline):
        super().__init__(host, port)
        self.deadline = deadline

    def connect(self):
        self.sock = DeadlineSocket(self.deadline)
        self.sock.connect((self.host, self.port))


class Client:
    def __init__(self, base, username, password):
        try:
            address = urllib.parse.urlsplit(base)
            port = 80 if address.port is None else address.port
            valid = (address.scheme == "http" and address.hostname == "127.0.0.1" and
                not address.username and not address.password and address.path in {"", "/"} and
                not address.query and not address.fragment and 1 <= port <= 65535)
        except ValueError:
            valid = False
        require(valid, "Use a plain HTTP loopback origin without credentials or a path")
        require(bool(username.strip() and password.strip()) and ":" not in username,
                "Synthetic username and password are required")
        self.host, self.port = address.hostname, port
        self.authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        self.wrong_authorizations = {
            "username": "Basic " + base64.b64encode(f"{username}-invalid:{password}".encode()).decode("ascii"),
            "password": "Basic " + base64.b64encode(f"{username}:{password}-invalid".encode()).decode("ascii")}

    def request(self, path, *, admitted=True, wrong=None, payload=None, timeout=5):
        require(math.isfinite(timeout) and timeout > 0, "HTTP timeout must be positive")
        deadline = time.monotonic() + timeout
        headers = {"Accept": "application/json"}
        if admitted:
            headers["Authorization"] = self.wrong_authorizations[wrong] if wrong else self.authorization
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection = DeadlineConnection(self.host, self.port, deadline)
        try:
            # Direct HTTPConnection does not consult proxy settings or follow redirects.
            connection.request("GET" if payload is None else "POST", path,
                body=None if payload is None else json.dumps(payload).encode(), headers=headers)
            response = connection.getresponse()
            with response:
                chunks, size = [], 0
                while True:
                    require(time.monotonic() < deadline, "HTTP response body exceeded its deadline")
                    chunk = response.read1(8192)
                    require(time.monotonic() < deadline, "HTTP response body exceeded its deadline")
                    if not chunk:
                        break
                    size += len(chunk)
                    require(size <= 65536, "HTTP response body exceeded its size bound")
                    chunks.append(chunk)
                require(response.length in {None, 0}, "HTTP response body ended before its declared length")
                return response.status, response.headers, b"".join(chunks)
        except (OSError, HTTPException):
            raise SmokeFailure("Loopback HTTP request failed") from None
        finally:
            connection.close()


def object_body(raw):
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise SmokeFailure("Expected a JSON response") from None
    require(isinstance(data, dict), "Expected a JSON object")
    return data


def finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def smoke(base, username, password, *, features_csv=None, timeout=60):
    require(math.isfinite(timeout) and timeout > 0, "Explanation timeout must be positive")
    client = Client(base, username, password)
    transaction_id = "auth-smoke-" + str(uuid.uuid4())
    code, _, raw = client.request("/status", admitted=False)
    require(code == 200 and object_body(raw) == {"status": "UP"}, "Minimal liveness failed")
    anonymous = {}
    for path in ["/demo/", "/demo/app.mjs", "/ready", "/model-info", "/metrics", "/docs", "/explain/" + transaction_id]:
        code, headers, _ = client.request(path, admitted=False)
        require(code == 401 and headers.get("WWW-Authenticate", "").startswith("Basic "), "Anonymous access was not denied")
        anonymous[path] = code
    code, _, _ = client.request("/predict", admitted=False,
        payload={"transaction_id": transaction_id, "features": [0]})
    require(code == 401, "Anonymous prediction was not denied")
    anonymous["/predict"] = code
    for component in ["username", "password"]:
        code, _, _ = client.request("/ready", wrong=component)
        require(code == 401, f"Wrong {component} was not denied")
    assets = []
    for path, expected_type in [("/demo/", "text/html"), ("/demo/app.mjs", "javascript"),
            ("/demo/api-client.mjs", "javascript"), ("/demo/styles.css", "text/css")]:
        code, headers, raw = client.request(path)
        require(code == 200 and raw and expected_type in headers.get("Content-Type", ""), "Authenticated demo asset failed")
        assets.append({"path": path, "status": code, "bytes": len(raw)})
    code, _, raw = client.request("/ready")
    readiness = object_body(raw)
    require(code == 200 and readiness.get("status") == "OK" and readiness.get("dependencies") ==
        {"model": "UP", "postgres": "UP", "redis_broker": "UP"}, "Authenticated readiness failed")
    code, _, raw = client.request("/model-info")
    info = object_body(raw)
    names = info.get("feature_names")
    require(code == 200 and isinstance(names, list) and names and all(isinstance(x, str) and x for x in names)
        and len(names) == info.get("feature_count") and len(set(names)) == len(names), "Model metadata failed")
    explanation_names = info.get("explanation_feature_names", names)
    require(isinstance(explanation_names, list) and explanation_names, "Explanation metadata failed")
    features = csv_features(features_csv, names) if features_csv else [100.0 if name == "Amount" else 0.0 for name in names]
    code, _, _ = client.request("/explain/" + transaction_id)
    require(code == 404, "Denied prediction unexpectedly has saved state")
    code, _, raw = client.request("/predict", payload={"transaction_id": transaction_id, "features": features})
    prediction = object_body(raw)
    require(code == 200 and prediction.get("transaction_id") == transaction_id and
        type(prediction.get("prediction")) is int and prediction["prediction"] in {0, 1} and
        finite(prediction.get("score")) and 0 <= prediction["score"] <= 1 and
        prediction.get("explanation_status") == "Calculation queued", "Authenticated prediction failed")
    deadline, pending = time.monotonic() + timeout, 0
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Explanation observation deadline expired")
        code, _, raw = client.request("/explain/" + transaction_id, timeout=min(5, remaining))
        require(time.monotonic() < deadline, "Explanation observation deadline expired")
        explanation = object_body(raw)
        require(explanation.get("transaction_id") == transaction_id, "Explanation ID mismatch")
        if code == 200:
            values = explanation.get("shap_values")
            require(explanation.get("status") == "COMPLETED" and explanation.get("feature_names") == explanation_names
                and isinstance(values, list) and len(values) == len(explanation_names) and all(finite(x) for x in values)
                and finite(explanation.get("base_value")), "Completed explanation failed validation")
            break
        require(code == 202 and explanation.get("status") == "PENDING", "Explanation was not pending or completed")
        pending += 1
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return {"status": "PASS", "input": "synthetic transport fixture", "transaction_id": transaction_id,
        "anonymous_statuses": anonymous, "wrong_credentials_statuses": {"username": 401, "password": 401}, "assets": assets,
        "readiness": readiness, "denied_prediction_saved_status": 404, "feature_count": len(names),
        "prediction_response": prediction, "explanation_response": explanation, "pending_observations": pending}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Refusing an existing receipt")
        report = smoke(args.url, os.environ.get("DEMO_AUTH_USERNAME", ""), os.environ.get("DEMO_AUTH_PASSWORD", ""),
            features_csv=args.features_csv, timeout=args.timeout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
        print("Authenticated HTTP smoke passed with a synthetic completed explanation")
    except (SmokeFailure, OSError, ValueError):
        raise SystemExit("Authenticated HTTP smoke failed; no credential or response details logged")
