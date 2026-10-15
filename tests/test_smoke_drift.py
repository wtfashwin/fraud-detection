"""Validate drift-smoke receipts against synthetic HTTP response shapes only.

Real HTTP/Grafana evaluation is performed by scripts/smoke_drift.py in integration CI.
"""
from copy import deepcopy
from datetime import datetime, timezone

import pytest


@pytest.fixture
def healthy_drift():
    return {"status": "UP", "drift": {"status": "DRIFT", "samples": 20, "min_samples": 20,
        "features": {"V1": {"statistic": 0.95, "pvalue": 1e-9}},
        "baseline": {"fingerprint": "a" * 64, "reference_rows": 100,
                     "source": {"split": "training_only", "data_kind": "synthetic"}}}}


@pytest.fixture
def definition():
    return {"uid": "fraud-input-distribution-drift", "title": "Raw fraud input distribution shifted",
            "folderUID": "synthetic-folder", "ruleGroup": "fraud-input-distribution", "isPaused": False,
            "condition": "B", "annotations": {"runbook_url": "https://example.test/PredictionDrift.md"},
            "data": [{"refId": "A", "model": {"expr": "max by (feature) (fraud_drift_ks)"}},
                     {"refId": "B", "model": {"type": "threshold", "expression": "A",
                         "conditions": [{"evaluator": {"type": "gt", "params": [0.05]}}]}}]}


@pytest.fixture
def firing_rules(definition):
    return {"status": "success", "data": {"groups": [{"name": definition["ruleGroup"],
        "rules": [{"name": definition["title"], "state": "firing", "health": "ok",
                   "lastEvaluation": "2026-10-15T00:00:05Z",
                   "alerts": [{"state": "Alerting", "labels": {"feature": "V1"}}]}]}]}}


def test_drift_receipt_requires_both_http_health_and_real_metric(healthy_drift):
    from scripts.smoke_drift import drift_evidence
    report = drift_evidence(healthy_drift, 'fraud_drift_ks{feature="V1"} 0.95\n')
    assert report["v1_ks"] == 0.95
    assert report["metric_v1_ks"] == 0.95
    assert report["reference_rows"] == 100
    assert "reference_samples" not in report


@pytest.mark.parametrize("metric", ['', 'fraud_drift_ks{feature="V10"} 1.0\n',
    'fraud_drift_ks{feature="V1"} NaN\n', 'fraud_drift_ks{feature="V1"} 0.05\n'])
def test_missing_nonfinite_or_boundary_metric_cannot_pass(healthy_drift, metric):
    from scripts.smoke_drift import SmokeFailure, drift_evidence
    with pytest.raises(SmokeFailure): drift_evidence(healthy_drift, metric)


def test_unavailable_baseline_cannot_pass_as_drift(healthy_drift):
    from scripts.smoke_drift import SmokeFailure, drift_evidence
    healthy_drift["drift"]["status"] = "UNAVAILABLE"
    with pytest.raises(SmokeFailure): drift_evidence(healthy_drift, 'fraud_drift_ks{feature="V1"} 1.0\n')


def test_grafana_receipt_requires_matching_fresh_firing_v1_instance(definition, firing_rules):
    from scripts.smoke_drift import firing_rule_evidence
    after = datetime(2026, 10, 15, tzinfo=timezone.utc)
    report = firing_rule_evidence(firing_rules, definition, after)
    assert report["uid"] == definition["uid"]
    assert report["state"] == "firing"
    assert report["firing_features"] == ["V1"]


@pytest.mark.parametrize("field,value", [("state", "pending"), ("health", "error"),
                                        ("health", "nodata"), ("lastEvaluation", "2026-10-14T23:59:59Z")])
def test_pending_error_nodata_or_stale_rules_cannot_pass(definition, firing_rules, field, value):
    from scripts.smoke_drift import firing_rule_evidence
    firing_rules["data"]["groups"][0]["rules"][0][field] = value
    assert firing_rule_evidence(firing_rules, definition, datetime(2026, 10, 15, tzinfo=timezone.utc)) is None


def test_other_rule_or_other_feature_cannot_satisfy_receipt(definition, firing_rules):
    from scripts.smoke_drift import firing_rule_evidence
    after = datetime(2026, 10, 15, tzinfo=timezone.utc)
    other = deepcopy(firing_rules)
    other["data"]["groups"][0]["rules"][0]["name"] = "Unrelated firing rule"
    assert firing_rule_evidence(other, definition, after) is None
    firing_rules["data"]["groups"][0]["rules"][0]["alerts"][0]["labels"]["feature"] = "Amount"
    assert firing_rule_evidence(firing_rules, definition, after) is None


def test_definition_must_use_ks_statistic_threshold(definition):
    from scripts.smoke_drift import SmokeFailure, validate_definition
    validate_definition(definition)
    definition["data"][1]["model"]["conditions"][0]["evaluator"]["params"] = [0.5]
    with pytest.raises(SmokeFailure): validate_definition(definition)


def test_startup_requires_grafana_database_ready():
    from scripts.smoke_drift import SmokeFailure, grafana_database_ready
    assert grafana_database_ready({"database": "ok", "version": "11.3.0"}) is True
    assert grafana_database_ready({"database": "starting"}) is False
    with pytest.raises(SmokeFailure): grafana_database_ready({"message": "not health metadata"})


@pytest.mark.parametrize("status,startup,expected", [(401, True, False), (403, True, False),
    (404, True, True), (404, False, False), (503, True, True), (503, False, True),
    (None, True, True), (None, False, True), (400, False, False)])
def test_http_retry_policy_distinguishes_startup_from_authorization_and_schema(status, startup, expected):
    from scripts.smoke_drift import HTTPFailure, retryable_http
    assert retryable_http(HTTPFailure("synthetic HTTP failure", status), startup=startup) is expected


def test_malformed_live_rule_response_is_schema_failure(definition):
    from scripts.smoke_drift import SmokeFailure, firing_rule_evidence
    with pytest.raises(SmokeFailure):
        firing_rule_evidence({"message": "not rule metadata"}, definition, datetime.now(timezone.utc))
