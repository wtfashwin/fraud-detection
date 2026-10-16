"""Synthetic fixtures verify provenance checks, not fraud model quality."""
import json
import pytest

from training.evaluate import main


@pytest.fixture
def saved_training_bundle(tmp_path, monkeypatch):
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.train import load_training_source, save_bundle, train_and_evaluate

    for name in ("MODEL_PATH", "COLUMNS_PATH", "SCALER_PATH", "MLFLOW_MODEL_URI"):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "training.csv"
    generate_synthetic_data(120, fraud_ratio=0.2).to_csv(source, index=False)
    frame, provenance = load_training_source(source)
    result = train_and_evaluate(frame, n_splits=3, n_estimators=2, source_info=provenance)
    directory = tmp_path / "bundle"
    save_bundle(result, directory)
    output = tmp_path / "evaluation"
    monkeypatch.setenv("MODEL_DIR", str(directory))
    monkeypatch.setenv("EVAL_DATA_CSV", str(source))
    monkeypatch.setenv("EVALUATION_OUTPUT_DIR", str(output))
    return source, directory, output


def test_rejects_original_training_source_before_scoring_or_report(saved_training_bundle, monkeypatch):
    # Removing the guard would score the original 120-row source (96 train, 24 holdout).
    from api.inference import InferenceBundle

    _, _, output = saved_training_bundle

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Training-source evaluation reached model scoring")

    monkeypatch.setattr(InferenceBundle, "score", unexpected_score)
    with pytest.raises(ValueError, match="training.source"):
        main()
    assert not output.exists()


def test_rejects_byte_identical_copy_at_another_path(saved_training_bundle, monkeypatch):
    source, _, output = saved_training_bundle
    copy = source.with_name("renamed.csv")
    copy.write_bytes(source.read_bytes())
    monkeypatch.setenv("EVAL_DATA_CSV", str(copy))
    with pytest.raises(ValueError, match="training.source"):
        main()
    assert not output.exists()


@pytest.mark.parametrize("relative_directory", [False, True])
def test_distinct_source_reports_byte_check_without_claiming_independence(
        saved_training_bundle, monkeypatch, relative_directory):
    import os
    from api.inference import ROOT
    from scripts.generate_synthetic_data import generate_synthetic_data

    source, directory, output = saved_training_bundle
    distinct = source.with_name("distinct.csv")
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(distinct, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(distinct))
    if relative_directory:
        monkeypatch.setenv("MODEL_DIR", os.path.relpath(directory, ROOT))
        monkeypatch.chdir(source.parent)
    main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["training_source_check"] == "different_hash"
    assert report["training_dataset_sha256"] == json.loads((directory / "metrics.json").read_text())["dataset_sha256"]
    assert "independence not verified" in report["evaluation_scope"]
    assert (output / "evaluation_curves.png").is_file()


@pytest.mark.parametrize("metadata", [None, {}])
def test_legacy_bundle_allows_evaluation_with_unavailable_provenance(
        saved_training_bundle, metadata):
    _, directory, output = saved_training_bundle
    path = directory / "metrics.json"
    if metadata is None:
        path.unlink()
    else:
        path.write_text(json.dumps(metadata))
    main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["training_source_check"] == "unavailable_missing_provenance"
    assert "training_dataset_sha256" not in report
    assert "independence not verified" in report["evaluation_scope"]


@pytest.mark.parametrize("metadata", ["{", "[]", '{"dataset_sha256": null}',
                                     '{"dataset_sha256": 123}', '{"dataset_sha256": "oops"}'])
def test_malformed_associated_metadata_fails_before_report(saved_training_bundle, metadata):
    _, directory, output = saved_training_bundle
    (directory / "metrics.json").write_text(metadata)
    with pytest.raises(ValueError, match="training provenance.*metrics.json"):
        main()
    assert not output.exists()


@pytest.mark.parametrize("override", ["MODEL_PATH", "COLUMNS_PATH", "SCALER_PATH", "MLFLOW_MODEL_URI"])
def test_overrides_do_not_use_model_directory_provenance(saved_training_bundle, monkeypatch, override):
    import joblib

    _, directory, output = saved_training_bundle
    if override == "MLFLOW_MODEL_URI":
        import mlflow.sklearn
        registered = directory.parent / "mlflow-model"
        mlflow.sklearn.save_model(joblib.load(directory / "pipeline.joblib"), str(registered))
        monkeypatch.setenv(override, str(registered))
    else:
        filename = {"MODEL_PATH": "pipeline.joblib", "COLUMNS_PATH": "columns.joblib",
                    "SCALER_PATH": "scaler.joblib"}[override]
        monkeypatch.setenv(override, str(directory / filename))
    # This unrelated local metadata must neither reject reuse nor break evaluation.
    (directory / "metrics.json").write_text("{")
    main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["training_source_check"] == "unavailable_artifact_overrides"
    assert "training_dataset_sha256" not in report
    assert "independence not verified" in report["evaluation_scope"]


def test_empty_registry_uri_still_rejects_training_source_before_scoring(
        saved_training_bundle, monkeypatch):
    from api.inference import InferenceBundle

    _, _, output = saved_training_bundle
    monkeypatch.setenv("MLFLOW_MODEL_URI", "")

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Empty registry URI bypassed local training-source guard")

    monkeypatch.setattr(InferenceBundle, "score", unexpected_score)
    with pytest.raises(ValueError, match="training.source"):
        main()
    assert not output.exists()


def test_replacing_csv_after_parse_does_not_change_hash_of_evaluated_rows(
        saved_training_bundle, monkeypatch):
    import pandas as pd
    from api.inference import InferenceBundle
    from scripts.generate_synthetic_data import generate_synthetic_data

    source, _, output = saved_training_bundle
    replacement = generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(index=False)
    read_csv = pd.read_csv

    def read_then_replace(*args, **kwargs):
        frame = read_csv(*args, **kwargs)
        source.write_text(replacement)
        return frame

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Parsed training rows were scored using replacement file's hash")

    monkeypatch.setattr(pd, "read_csv", read_then_replace)
    monkeypatch.setattr(InferenceBundle, "score", unexpected_score)
    with pytest.raises(ValueError, match="training.source"):
        main()
    assert not output.exists()
