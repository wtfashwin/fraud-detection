"""Prove real input-drift telemetry and local Grafana firing with synthetic traffic.

This checks plumbing and alert evaluation, not fraud accuracy or an AUC decrease.
Grafana is read-only. No notification/configuration API is invoked.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

RULE_UID = "fraud-input-distribution-drift"
RULE_STATES_PATH = "/api/prometheus/grafana/api/v1/rules"
# Grafana11.3 registers the live-state GET route (provisioning GET is definitions only):
# https://github.com/grafana/grafana/blob/v11.3.0/pkg/services/ngalert/api/generated_base_api_prometheus.go
RAW_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class SmokeFailure(RuntimeError):
    pass


class HTTPFailure(SmokeFailure):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def retryable_http(error, startup=False):
    return error.status is None or 500 <= error.status < 600 or (startup and error.status == 404)


def grafana_database_ready(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("database"), str):
        raise SmokeFailure("Grafana health response lacks database status")
    return payload["database"] == "ok"


def wait_for_grafana(base, deadline, auth):
    while time.monotonic() < deadline:
        try:
            if grafana_database_ready(http(base, "/api/health", deadline, auth=auth)):
                definition = http(base, "/api/v1/provisioning/alert-rules/" + RULE_UID, deadline, auth=auth)
                validate_definition(definition)
                return definition
        except HTTPFailure as error:
            if not retryable_http(error, startup=True):
                raise
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise SmokeFailure("Grafana database and managed rule were not ready before the smoke deadline")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, target):
        return None


def checked_url(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise SmokeFailure("Use an HTTP(S) base URL without embedded credentials, query, or fragment")
    return value.rstrip("/")


def http(base, path, deadline, payload=None, auth=None, text=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SmokeFailure("Drift smoke deadline expired")
    headers = {"Accept": "text/plain" if text else "application/json"}
    if auth:
        headers["Authorization"] = auth
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        # Do not forward an environment-supplied credential through redirects.
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=min(10, remaining)) as response:
            if response.status != 200:
                raise HTTPFailure(f"Expected HTTP200 from {path.split('?')[0]}", response.status)
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise SmokeFailure("HTTP response exceeded the bounded smoke limit")
        return body.decode() if text else json.loads(body)
    except urllib.error.HTTPError as error:
        raise HTTPFailure(f"HTTP{error.code} from {path.split('?')[0]}", error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise HTTPFailure(f"HTTP transport unavailable at {path.split('?')[0]}") from None
    except (ValueError, UnicodeError):
        raise SmokeFailure(f"Invalid response body at {path.split('?')[0]}") from None


def drift_evidence(health, metrics):
    drift = health.get("drift", {})
    baseline = drift.get("baseline", {})
    feature = drift.get("features", {}).get("V1", {})
    statistic = feature.get("statistic")
    samples = drift.get("samples")
    minimum = drift.get("min_samples")
    reference_rows = baseline.get("reference_rows")
    if (health.get("status") != "UP" or drift.get("status") != "DRIFT"
            or not isinstance(samples, int) or not isinstance(minimum, int)
            or samples < minimum or minimum < 20 or not isinstance(reference_rows, int)
            or reference_rows < minimum or baseline.get("source", {}).get("split") != "training_only"
            or not isinstance(statistic, (int, float)) or not math.isfinite(statistic) or not 0.05 < statistic <= 1):
        raise SmokeFailure("Health did not prove a valid training-baseline V1 drift comparison")
    matches = []
    for line in metrics.splitlines():
        match = re.match(r'^fraud_drift_ks\{([^}]*)\}\s+(\S+)', line)
        if match and re.search(r'(?:^|,)\s*feature="V1"\s*(?:,|$)', match.group(1)):
            try:
                matches.append(float(match.group(2)))
            except ValueError:
                raise SmokeFailure("V1 KS metric is not numeric") from None
    if not matches or any(not math.isfinite(value) or not 0.05 < value <= 1 for value in matches):
        raise SmokeFailure("Metrics did not prove an actual V1 KS series above0.05")
    fingerprint = baseline.get("fingerprint", "")
    if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        raise SmokeFailure("Health did not identify a valid baseline fingerprint")
    return {"status": "DRIFT", "samples": samples, "reference_rows": reference_rows,
            "baseline_fingerprint": fingerprint, "baseline_data_kind": baseline["source"].get("data_kind", "unspecified"),
            "v1_ks": float(statistic), "metric_v1_ks": max(matches)}


def validate_definition(definition):
    if (definition.get("uid") != RULE_UID or definition.get("isPaused") is True
            or any(not isinstance(definition.get(key), str) or not 1 <= len(definition[key]) <= 200
                   for key in ("title", "folderUID", "ruleGroup"))
            or not definition.get("annotations", {}).get("runbook_url")):
        raise SmokeFailure("Expected managed Grafana rule is missing, paused, or lacks its runbook")
    queries = {entry.get("refId"): entry.get("model", {}) for entry in definition.get("data", [])}
    condition = queries.get(definition.get("condition"), {})
    source = queries.get(condition.get("expression"), {})
    evaluators = [entry.get("evaluator", {}) for entry in condition.get("conditions", [])]
    if (condition.get("type") != "threshold" or not re.search(r"\bfraud_drift_ks\b", source.get("expr", ""))
            or len(evaluators) != 1 or evaluators[0].get("type") != "gt" or evaluators[0].get("params") != [0.05]):
        raise SmokeFailure("Managed Grafana rule does not evaluate the required KS>0.05 threshold")


def firing_rule_evidence(payload, definition, after):
    if (not isinstance(payload, dict) or payload.get("status") != "success"
            or not isinstance(payload.get("data"), dict) or not isinstance(payload["data"].get("groups"), list)):
        raise SmokeFailure("Grafana live rule response does not match the expected schema")
    matches = [rule for group in payload.get("data", {}).get("groups", [])
               if group.get("name") == definition["ruleGroup"]
               for rule in group.get("rules", []) if rule.get("name") == definition["title"]]
    if len(matches) != 1:
        return None
    rule = matches[0]
    try:
        evaluated = datetime.fromisoformat(rule.get("lastEvaluation", "").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if (evaluated.tzinfo is None or evaluated < after or rule.get("state") != "firing"
            or rule.get("health") != "ok"):
        return None
    features = sorted({alert.get("labels", {}).get("feature") for alert in rule.get("alerts", [])
                       if str(alert.get("state", "")).lower() in {"alerting", "firing"}
                       and alert.get("labels", {}).get("feature") in RAW_FEATURES})
    if "V1" not in features:
        return None
    return {"uid": definition["uid"], "state": "firing", "health": "ok",
            "last_evaluation": evaluated.isoformat(), "firing_features": features}


def smoke(base, grafana_base, timeout=120):
    base, grafana_base = checked_url(base), checked_url(grafana_base)
    if not 0 < timeout <= 600:
        raise SmokeFailure("Smoke timeout must be between zero and600 seconds")
    password = os.getenv("GRAFANA_ADMIN_PASSWORD")
    if not password:
        raise SmokeFailure("Set GRAFANA_ADMIN_PASSWORD through the environment")
    username = os.getenv("GRAFANA_ADMIN_USER", "admin")
    auth = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    deadline = time.monotonic() + timeout
    ready = http(base, "/ready", deadline)
    if ready.get("status") != "OK":
        raise SmokeFailure("API dependencies are not ready")
    info = http(base, "/model-info", deadline)
    if info.get("feature_names") != RAW_FEATURES or info.get("feature_count") != 30:
        raise SmokeFailure("Expected the canonical30 raw serving inputs")
    initial = http(base, "/health", deadline).get("drift", {})
    minimum = initial.get("min_samples", 20)
    existing = initial.get("samples") or 0
    if (not initial.get("baseline") or not isinstance(minimum, int) or not isinstance(existing, int)
            or not 20 <= minimum <= 512 or not 0 <= existing <= 512):
        raise SmokeFailure("A usable raw training baseline is required before synthetic drift smoke")
    # A fresh CI stack submits20. An existing window is replaced with at most512 test observations.
    count = max(minimum, existing)
    features = [1_000_000.0 if name == "V1" else 100.0 if name == "Amount" else 0.0 for name in RAW_FEATURES]
    for _ in range(count):
        transaction_id = "synthetic-drift-smoke-" + str(uuid.uuid4())
        result = http(base, "/predict", deadline, {"transaction_id": transaction_id, "features": features})
        score = result.get("score")
        if (result.get("transaction_id") != transaction_id or result.get("prediction") not in (0, 1)
                or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                or result.get("explanation_status") != "Calculation queued"):
            raise SmokeFailure("Scoring did not accept a synthetic drift transaction")
    submitted_at = datetime.now(timezone.utc)
    drift = drift_evidence(http(base, "/health", deadline), http(base, "/metrics", deadline, text=True))
    definition = wait_for_grafana(grafana_base, deadline, auth)
    query = urllib.parse.urlencode({"folder_uid": definition["folderUID"], "rule_group": definition["ruleGroup"],
                                    "rule_name": definition["title"]})
    while time.monotonic() < deadline:
        try:
            state = http(grafana_base, RULE_STATES_PATH + "?" + query, deadline, auth=auth)
            grafana = firing_rule_evidence(state, definition, submitted_at)
        except HTTPFailure as error:
            if not retryable_http(error):
                raise
            grafana = None
        if grafana is not None:
            return {"status": "PASS", "input_scope": "explicitly synthetic shifted raw inputs; not fraud accuracy evidence",
                    "submitted_predictions": count, "raw_feature_count": 30, "drift": drift, "grafana": grafana,
                    "scope": "Real HTTP, shared drift telemetry, and local Grafana evaluation; no external notifications"}
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise SmokeFailure("Grafana did not reach a fresh healthy firing V1 state before the smoke deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, help="Save a bounded aggregate receipt without credentials or input rows")
    args = parser.parse_args()
    try:
        report = smoke(args.url, args.grafana_url, args.timeout)
        code = 0
    except SmokeFailure as error:
        report = {"status": "FAIL", "error": str(error), "input_scope": "synthetic drift integration fixture"}
        code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
