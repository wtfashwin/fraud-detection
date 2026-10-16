"""Synthetic fixtures verify provenance checks, not fraud model quality."""
import json
import hashlib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

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
    assert report["training_row_check"] == "no_exact_overlap"
    assert report["training_unique_rows"] == 96
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
    assert report["training_row_check"] == "unavailable_missing_provenance"
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
    assert report["training_row_check"] == "unavailable_artifact_overrides"
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


@pytest.mark.parametrize("change", ["reordered", "subset", "relabelled", "reformatted"])
def test_rejects_reused_training_feature_rows_before_loading_model(
        saved_training_bundle, monkeypatch, change):
    source, _, output = saved_training_bundle
    frame = pd.read_csv(source, float_precision="round_trip")
    train, _ = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.Class)
    reused = frame.iloc[::-1] if change == "reordered" else train.groupby("Class").head(2)
    if change == "relabelled":
        reused = reused.copy()
        reused["Class"] = 1 - reused.Class
    evaluation = source.with_name(f"reused-{change}.csv")
    if change == "reformatted":
        reused[reused.columns[::-1]].to_csv(evaluation, index=False, float_format="%.17e")
    else:
        reused.to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Reused fitted rows reached model loading")

    monkeypatch.setattr("training.evaluate.load_bundle", unexpected_load)
    with pytest.raises(ValueError, match="overlap.*training|training.*overlap"):
        main()
    assert not output.exists()


def test_saved_internal_holdout_has_no_fitted_row_overlap(saved_training_bundle, monkeypatch):
    source, _, output = saved_training_bundle
    frame = pd.read_csv(source, float_precision="round_trip")
    _, holdout = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.Class)
    evaluation = source.with_name("internal-holdout.csv")
    holdout.to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))
    main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["training_row_check"] == "no_exact_overlap"
    assert report["fraud_rows"] + report["legitimate_rows"] == 24
    assert "independence not verified" in report["evaluation_scope"]


def test_declared_missing_row_manifest_fails_before_model_loading(saved_training_bundle, monkeypatch):
    source, directory, output = saved_training_bundle
    manifest = directory / "training_rows.json"
    manifest.unlink(missing_ok=True)
    evaluation = source.with_name("distinct.csv")
    from scripts.generate_synthetic_data import generate_synthetic_data
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Missing declared row provenance reached model loading")

    monkeypatch.setattr("training.evaluate.load_bundle", unexpected_load)
    with pytest.raises(ValueError, match="training row provenance"):
        main()
    assert not output.exists()


@pytest.mark.parametrize("manifest", ["{", "[]", '{"version": 99}',
    '{"version": 1, "algorithm": "sha256-float64-be-v1", "feature_names": [], "row_count": 96, "fingerprints": []}'])
def test_malformed_declared_row_manifest_fails(saved_training_bundle, monkeypatch, manifest):
    source, directory, output = saved_training_bundle
    path = directory / "training_rows.json"
    path.write_text(manifest)
    metadata_path = directory / "metrics.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["training_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    evaluation = source.with_name("distinct.csv")
    from scripts.generate_synthetic_data import generate_synthetic_data
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))
    with pytest.raises(ValueError, match="training row provenance"):
        main()
    assert not output.exists()


def test_altered_row_manifest_checksum_fails(saved_training_bundle, monkeypatch):
    source, directory, output = saved_training_bundle
    path = directory / "training_rows.json"
    path.write_text('{"version": 1}\n')
    evaluation = source.with_name("distinct.csv")
    from scripts.generate_synthetic_data import generate_synthetic_data
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))
    with pytest.raises(ValueError, match="training row provenance.*SHA-256"):
        main()
    assert not output.exists()


@pytest.mark.parametrize("pipeline_change", ["replace", "remove"])
def test_row_manifest_cannot_describe_a_replaced_or_unselected_pipeline(
        saved_training_bundle, monkeypatch, pipeline_change):
    source, directory, output = saved_training_bundle
    pipeline = directory / "pipeline.joblib"
    if pipeline_change == "replace":
        from training.train import inference_pipeline, train_and_evaluate
        from scripts.generate_synthetic_data import generate_synthetic_data
        import joblib
        other = train_and_evaluate(generate_synthetic_data(120, fraud_ratio=0.2, random_state=99),
                                   n_splits=3, n_estimators=2)
        joblib.dump(inference_pipeline(other["pipeline"]), pipeline)
    else:
        pipeline.unlink()
    evaluation = source.with_name("distinct.csv")
    from scripts.generate_synthetic_data import generate_synthetic_data
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))
    with pytest.raises(ValueError, match="training row provenance.*pipeline"):
        main()
    assert not output.exists()


def test_signed_zero_and_changed_labels_do_not_hide_fitted_inputs(tmp_path, monkeypatch):
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.train import save_bundle, train_and_evaluate

    frame = generate_synthetic_data(120, fraud_ratio=0.2)
    frame["Time"] = -0.0
    result = train_and_evaluate(frame, n_splits=3, n_estimators=2)
    directory = tmp_path / "bundle"
    save_bundle(result, directory)
    reused = frame.loc[result["training_indices"]].groupby("Class").head(2).copy()
    reused["Time"] = 0.0
    reused["Class"] = 1 - reused.Class
    source = tmp_path / "reused.csv"
    reused.to_csv(source, index=False)
    output = tmp_path / "evaluation"
    monkeypatch.setenv("MODEL_DIR", str(directory))
    monkeypatch.setenv("EVAL_DATA_CSV", str(source))
    monkeypatch.setenv("EVALUATION_OUTPUT_DIR", str(output))
    with pytest.raises(ValueError, match="overlap.*training|training.*overlap"):
        main()
    assert not output.exists()


def test_duplicate_inputs_crossing_partitions_are_rejected(tmp_path, monkeypatch):
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.preprocess import FEATURE_NAMES
    from training.train import save_bundle, train_and_evaluate

    frame = generate_synthetic_data(120, fraud_ratio=0.2)
    frame[FEATURE_NAMES] = 0.0
    result = train_and_evaluate(frame, n_splits=3, n_estimators=2)
    directory = tmp_path / "bundle"
    save_bundle(result, directory)
    source = tmp_path / "holdout.csv"
    frame.drop(index=result["training_indices"]).to_csv(source, index=False)
    output = tmp_path / "evaluation"
    monkeypatch.setenv("MODEL_DIR", str(directory))
    monkeypatch.setenv("EVAL_DATA_CSV", str(source))
    monkeypatch.setenv("EVALUATION_OUTPUT_DIR", str(output))
    with pytest.raises(ValueError, match="overlap.*training|training.*overlap"):
        main()
    assert not output.exists()
    manifest = json.loads((directory / "training_rows.json").read_text())
    assert manifest["row_count"] == 96
    assert len(manifest["fingerprints"]) == 1


@pytest.mark.parametrize("field,value", [
    ("training_rows_sha256", None), ("training_rows_sha256", "oops"),
    ("training_pipeline_sha256", 1), ("training_pipeline_sha256", None),
])
def test_incomplete_or_invalid_row_provenance_declaration_fails(
        saved_training_bundle, monkeypatch, field, value):
    source, directory, output = saved_training_bundle
    path = directory / "metrics.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    evaluation = source.with_name("distinct.csv")
    from scripts.generate_synthetic_data import generate_synthetic_data
    generate_synthetic_data(40, fraud_ratio=0.2, random_state=99).to_csv(evaluation, index=False)
    monkeypatch.setenv("EVAL_DATA_CSV", str(evaluation))
    with pytest.raises(ValueError, match="training row provenance"):
        main()
    assert not output.exists()


def test_row_fingerprint_protocol_uses_big_endian_float64():
    from training.preprocess import FEATURE_NAMES
    from training.provenance import training_row_manifest

    # SHA-256 of the independently packed big-endian doubles 0 through 29.
    frame = pd.DataFrame([list(range(30))], columns=FEATURE_NAMES)
    manifest = training_row_manifest(frame)
    assert manifest["fingerprints"] == ["b1607d093a679e96a4804f355da2902d868b5899ebf37b1ddb32dc9052dfaff1"]


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
