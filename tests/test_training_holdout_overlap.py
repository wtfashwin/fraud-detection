"""Outer holdout feature matches are observable without asserting transaction identity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.preprocess import FEATURE_NAMES
from training.train import fit_and_evaluate_partition, save_bundle


@pytest.mark.parametrize(
    ("case", "expected_overlap"),
    [("same_label", 1), ("conflicting_label", 1), ("no_overlap", 0)],
)
def test_training_reports_raw_holdout_feature_matches_without_rejecting_rows(tmp_path, case, expected_overlap):
    rng = np.random.default_rng(903)
    values = rng.normal(size=(42, len(FEATURE_NAMES)))
    values[:, -1] = np.abs(values[:, -1])
    train = pd.DataFrame(values[:40], columns=FEATURE_NAMES)
    train_labels = pd.Series([0, 1] * 20)
    holdout = pd.DataFrame(values[40:], columns=FEATURE_NAMES)
    if case != "no_overlap":
        holdout.iloc[0] = train.iloc[0]
    first_label = 1 if case == "conflicting_label" else 0
    holdout_labels = pd.Series([first_label, 1 - first_label])

    result = fit_and_evaluate_partition(
        train, holdout, train_labels, holdout_labels,
        n_splits=2, n_estimators=2, source_info={"data_kind": "synthetic"},
    )
    assert result["metrics"]["holdout_feature_overlap_rows"] == expected_overlap
    assert "labels excluded" in result["metrics"]["holdout_feature_overlap_scope"]
    assert result["metrics"]["data_kind"] == "synthetic"
    assert result["metrics"]["test_rows"] == 2
    save_bundle(result, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["holdout_feature_overlap_rows"] == expected_overlap
    assert "labels excluded" in saved["holdout_feature_overlap_scope"]


def test_holdout_feature_overlap_counts_each_matching_row():
    rng = np.random.default_rng(904)
    values = rng.normal(size=(41, len(FEATURE_NAMES)))
    values[:, -1] = np.abs(values[:, -1])
    train = pd.DataFrame(values[:40], columns=FEATURE_NAMES)
    train_labels = pd.Series([0, 1] * 20)
    holdout = pd.DataFrame([values[0], values[0], values[0], values[40]], columns=FEATURE_NAMES)
    holdout_labels = pd.Series([0, 1, 0, 1])

    result = fit_and_evaluate_partition(train, holdout, train_labels, holdout_labels, n_splits=2, n_estimators=2)
    assert result["metrics"]["test_rows"] == 4
    assert result["metrics"]["holdout_feature_overlap_rows"] == 3


def test_tuning_persists_nonzero_feature_overlap_in_mlflow_and_bundle(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.tune import tune_model

    monkeypatch.delenv("MLFLOW_REGISTER_MODEL", raising=False)
    frame = generate_synthetic_data(120, fraud_ratio=0.2, random_state=905)
    frame[FEATURE_NAMES] = 0.0
    tracking_uri = (tmp_path / "mlruns").as_uri()
    result, study = tune_model(frame, n_trials=1, n_estimators=2, max_majority=30,
                               tracking_uri=tracking_uri, study_name="synthetic-overlap-test",
                               source_info={"data_kind": "synthetic"})

    assert len(study.trials) == 1
    assert result["metrics"]["test_rows"] == 24
    assert result["metrics"]["holdout_feature_overlap_rows"] == 24
    info = mlflow.models.get_model_info(result["mlflow_model_uri"])
    client = MlflowClient(tracking_uri=tracking_uri)
    assert client.get_run(info.run_id).data.metrics["holdout_feature_overlap_rows"] == 24
    logged_metrics = json.loads(Path(client.download_artifacts(info.run_id, "evaluation/metrics.json")).read_text())
    assert logged_metrics["holdout_feature_overlap_rows"] == 24
    assert "labels excluded" in logged_metrics["holdout_feature_overlap_scope"]
    output = tmp_path / "bundle"
    save_bundle(result, output)
    saved = json.loads((output / "metrics.json").read_text())
    assert saved["holdout_feature_overlap_rows"] == 24
    assert "labels excluded" in saved["holdout_feature_overlap_scope"]
