"""Synthetic checks against a private local MLflow store, never a remote registry."""
import json
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_mlflow_process_state(monkeypatch):
    from mlflow.tracking import fluent
    from mlflow.tracking._model_registry import utils as registry_utils
    from mlflow.tracking._tracking_service import utils as tracking_utils

    # MLflow setters change module globals as well as environment variables.
    monkeypatch.setattr(registry_utils, "_registry_uri", registry_utils._registry_uri)
    monkeypatch.setattr(tracking_utils, "_tracking_uri", tracking_utils._tracking_uri)
    monkeypatch.setattr(fluent, "_active_experiment_id", fluent._active_experiment_id)
    for name in ("MLFLOW_REGISTRY_URI", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID",
                 "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    old_last_run_id = fluent._last_active_run_id.get()
    old_run_stack = list(fluent._active_run_stack.get())
    try:
        yield
    finally:
        fluent._last_active_run_id.set(old_last_run_id)
        fluent._active_run_stack.set(old_run_stack)


@pytest.fixture(scope="module")
def synthetic_result():
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.train import train_and_evaluate

    frame = generate_synthetic_data(120, fraud_ratio=0.2, random_state=123)
    return train_and_evaluate(
        frame, n_splits=3, n_estimators=2, source_info={"data_kind": "synthetic"}
    )


def log_local_result(tmp_path, monkeypatch, synthetic_result, mode, auc):
    import mlflow
    from mlflow.tracking import MlflowClient
    from training.train import log_to_mlflow

    uri = (tmp_path / "local-mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "0" if mode == "logging-only" else "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "smoke" if mode == "smoke" else "production")
    # AUCs below this threshold must still be logged in nonproduction paths.
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", "0.95" if mode == "production" else "0.999")
    model_name = "ci-smoke-auc-contract" if mode == "smoke" else "synthetic-local-auc-contract"
    monkeypatch.setenv("MLFLOW_MODEL_NAME", model_name)
    mlflow.set_registry_uri(uri)

    result = {**synthetic_result, "metrics": {**synthetic_result["metrics"], "roc_auc": auc}}
    if mode == "production":
        # This invented label exercises the gate; the dataset is wholly synthetic.
        result["metrics"]["data_kind"] = "real"
    before = result["metrics"].copy()
    model_uri = log_to_mlflow(result, tracking_uri=uri, experiment_name="synthetic-local-auc-contract")
    assert result["metrics"] == before
    assert result["metrics"]["roc_auc"] is auc

    run_id = mlflow.models.get_model_info(model_uri).run_id
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    run = client.get_run(run_id)
    with open(client.download_artifacts(run_id, "evaluation/metrics.json")) as handle:
        saved = json.load(handle)
    versions = client.search_model_versions(f"name='{model_name}'")
    assert len(versions) == (0 if mode == "logging-only" else 1)
    return run.data.metrics, saved


@pytest.mark.parametrize("mode", ["production", "smoke", "logging-only"])
@pytest.mark.parametrize(
    "auc, expected",
    [
        (0.96, 0.96),
        (np.float32(0.96), 0.9599999785423279),
        (Fraction(24, 25), 0.96),
        (np.longdouble("0.96"), 0.96),
        (np.int64(1), 1.0),
    ],
    ids=["python-float", "numpy-float32", "fraction", "numpy-longdouble", "numpy-int"],
)
def test_valid_auc_is_saved_as_a_numeric_metric_and_artifact(
    tmp_path, monkeypatch, synthetic_result, mode, auc, expected
):
    metrics, saved = log_local_result(tmp_path, monkeypatch, synthetic_result, mode, auc)
    assert metrics["roc_auc"] == pytest.approx(expected, rel=0, abs=1e-15)
    assert isinstance(saved["roc_auc"], (int, float)) and not isinstance(saved["roc_auc"], bool)
    assert saved["roc_auc"] == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize("mode", ["smoke", "logging-only"])
@pytest.mark.parametrize(
    "auc, metric_expected, artifact_expected",
    [
        (np.float32(1.1), None, "1.1"),
        (Fraction(-1, 2), None, "-1/2"),
        (Decimal("0.4"), None, "0.4"),
        ("0.96", None, "0.96"),
        (None, None, None),
        (True, 1.0, True),
        (-0.5, -0.5, -0.5),
    ],
    ids=["out-of-range-numpy", "out-of-range-fraction", "decimal", "string", "none", "bool", "out-of-range-float"],
)
def test_nonproduction_invalid_and_nonreal_values_keep_existing_logging_behavior(
    tmp_path, monkeypatch, synthetic_result, mode, auc, metric_expected, artifact_expected
):
    metrics, saved = log_local_result(tmp_path, monkeypatch, synthetic_result, mode, auc)
    if metric_expected is None:
        assert "roc_auc" not in metrics
    else:
        assert metrics["roc_auc"] == metric_expected
    assert saved["roc_auc"] == artifact_expected
    assert type(saved["roc_auc"]) is type(artifact_expected)


def test_fraction_that_rounds_to_threshold_stops_before_tracking(tmp_path, monkeypatch):
    import mlflow
    from training.train import log_to_mlflow

    def tracking_must_not_start(_uri):
        raise AssertionError("tracking started for an AUC that rounds to the threshold")

    monkeypatch.setattr(mlflow, "set_tracking_uri", tracking_must_not_start)
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "production")
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", "0.95")
    threshold = 0.95
    auc = Fraction.from_float(threshold) + Fraction(1, 10**40)
    assert auc > threshold and float(auc) == threshold
    with pytest.raises(ValueError, match="threshold"):
        log_to_mlflow({"metrics": {"data_kind": "real", "roc_auc": auc}},
                      tracking_uri=(tmp_path / "unused").as_uri())


def test_next_python_float_above_threshold_still_reaches_tracking(tmp_path, monkeypatch):
    import mlflow
    from training.train import log_to_mlflow

    class TrackingReached(Exception):
        pass

    def stop_before_tracking(_uri):
        raise TrackingReached

    monkeypatch.setattr(mlflow, "set_tracking_uri", stop_before_tracking)
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "production")
    monkeypatch.setenv("MLFLOW_AUC_THRESHOLD", "0.95")
    with pytest.raises(TrackingReached):
        log_to_mlflow({"metrics": {"data_kind": "real", "roc_auc": math.nextafter(0.95, 1.0)}},
                      tracking_uri=(tmp_path / "unused").as_uri())
