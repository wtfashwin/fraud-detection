"""Invented metric declarations stop at an offline tracking boundary, never a registry."""
import math
from fractions import Fraction

import numpy as np
import pytest


class TrackingReached(RuntimeError):
    """Stop before even configuring MLflow; no tracking or registration is performed."""


@pytest.fixture
def registration_gate(monkeypatch, tmp_path):
    import mlflow
    from training.train import log_to_mlflow

    calls = []

    def stop_before_tracking(uri):
        calls.append(uri)
        raise TrackingReached("Offline tracking boundary reached")

    monkeypatch.setattr(mlflow, "set_tracking_uri", stop_before_tracking)
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "production")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "synthetic-offline-boundary")
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", "0.95")

    def invoke(auc=0.5, data_kind="real"):
        # This is an invented declaration testing the gate, not real evaluation evidence.
        return log_to_mlflow({"metrics": {"data_kind": data_kind, "roc_auc": auc}},
                             tracking_uri=(tmp_path / "unused-tracking").as_uri())

    return invoke, calls


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.01", "1.01", "invalid", ""])
def test_invalid_production_threshold_stops_before_tracking(registration_gate, monkeypatch, threshold):
    invoke, calls = registration_gate
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", threshold)
    with pytest.raises(ValueError):
        invoke()
    assert calls == []


@pytest.mark.parametrize("auc", [math.nan, math.inf, -math.inf, -0.01, 1.01,
    True, np.bool_(True), np.array([0.97]), np.array(0.97), np.complex128(0.97 + 0.1j),
    0.97 + 0.1j, "0.97", None, [0.97]])
def test_invalid_production_auc_stops_before_tracking(registration_gate, auc):
    invoke, calls = registration_gate
    with pytest.raises(ValueError, match="ROC AUC"):
        invoke(auc)
    assert calls == []


@pytest.mark.parametrize("auc,threshold", [(0.95, "0.95"), (0.94, "0.95"), (1.0, "1"), (0.0, "0")])
def test_auc_must_strictly_exceed_threshold(registration_gate, monkeypatch, auc, threshold):
    invoke, calls = registration_gate
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", threshold)
    with pytest.raises(ValueError, match="exceed"):
        invoke(auc)
    assert calls == []


@pytest.mark.parametrize("auc,threshold", [(0.96, "0.95"), (1.0, "0.95"), (0.01, "0"),
    (1, "0.95"), (np.float32(0.96), "0.95"), (np.float64(0.96), "0.95"),
    (np.longdouble(0.96), "0.95"), (np.int64(1), "0.95"), (Fraction(24, 25), "0.95")])
def test_valid_production_gate_can_reach_offline_tracking(registration_gate, monkeypatch, auc, threshold):
    invoke, calls = registration_gate
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", threshold)
    with pytest.raises(TrackingReached):
        invoke(auc)
    assert len(calls) == 1


def test_production_mode_still_rejects_synthetic_declaration(registration_gate):
    invoke, calls = registration_gate
    with pytest.raises(ValueError, match="real evaluation data"):
        invoke(0.99, data_kind="synthetic")
    assert calls == []


@pytest.mark.parametrize("mode", ["smoke", "logging-only"])
def test_separate_nonproduction_paths_do_not_use_production_threshold(registration_gate, monkeypatch, mode):
    invoke, calls = registration_gate
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", "nan")
    if mode == "smoke":
        monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "smoke")
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "ci-smoke-offline-boundary")
    else:
        monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "0")
    with pytest.raises(TrackingReached):
        invoke(data_kind="synthetic")
    assert len(calls) == 1
