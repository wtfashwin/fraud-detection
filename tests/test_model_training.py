"""Small synthetic fixtures check mechanics, not real fraud-model quality."""
import importlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from scripts.generate_synthetic_data import generate_synthetic_data


def fixture_data(rows=120):
    return generate_synthetic_data(rows, fraud_ratio=0.2)


def test_synthetic_generator_does_not_change_global_rng():
    np.random.seed(7)
    expected = np.random.random(3)
    np.random.seed(7)
    generate_synthetic_data(100)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_synthetic_generator_rejects_incompatible_feature_count():
    with pytest.raises(ValueError, match="30"):
        generate_synthetic_data(100, n_features=10)


def test_synthetic_generator_main_preserves_requested_ci_size(tmp_path, monkeypatch):
    from scripts.generate_synthetic_data import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI_SYNTHETIC_SAMPLES", "60")
    monkeypatch.setenv("TEST_SYNTHETIC_SAMPLES", "200")
    main()
    assert len(pd.read_csv("data/creditcard.csv")) == 60


def test_preprocessing_import_has_no_file_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("training.preprocess", None)
    importlib.import_module("training.preprocess")
    assert list(tmp_path.iterdir()) == []


def test_preprocess_scaler_only_sees_training_rows():
    from training.preprocess import prepare_data
    df = fixture_data()
    df.loc[0, "Amount"] = 1e9
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.Class)
    result = prepare_data(df)
    np.testing.assert_allclose(result["scaler"].mean_, train.drop(columns="Class").mean())
    assert result["scaler"].n_samples_seen_ == len(train)
    assert len(result["y_test"]) == len(test)


def test_training_uses_fold_specific_scalers_and_one_imbalance_correction():
    from training.train import fit_training_fold
    df = fixture_data()
    X, y = df.drop(columns="Class"), df.Class
    pipeline = fit_training_fold(X.iloc[:80], y.iloc[:80], n_estimators=2)
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, X.iloc[:80].mean())
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == 1


def test_training_writes_complete_isolated_bundle(tmp_path):
    from training.train import train_and_evaluate, save_bundle
    result = train_and_evaluate(fixture_data(), n_splits=3, n_estimators=2)
    save_bundle(result, tmp_path)
    assert {"pipeline.joblib", "xgb_model.joblib", "scaler.joblib", "columns.joblib",
            "feature_names.json", "metrics.json", "drift_baseline.json"} <= {p.name for p in tmp_path.iterdir()}
    assert joblib.load(tmp_path / "columns.joblib") == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assert 0 <= result["metrics"]["average_precision"] <= 1
    assert result["metrics"]["test_rows"] == 24


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "bad"])
def test_training_rejects_invalid_feature_values(bad_value):
    from training.preprocess import validate_dataset
    df = fixture_data()
    df["Amount"] = df["Amount"].astype(object)
    df.loc[0, "Amount"] = bad_value
    with pytest.raises(ValueError, match="numeric|finite"):
        validate_dataset(df)


def test_training_reorders_columns_and_rejects_nonbinary_labels():
    from training.preprocess import validate_dataset
    df = fixture_data()
    X, y = validate_dataset(df[df.columns[::-1]])
    assert list(X.columns) == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df["Class"] = df["Class"].astype(float)
    df.loc[0, "Class"] = 0.5
    with pytest.raises(ValueError, match="0 and 1"):
        validate_dataset(df)


def test_metrics_report_imbalanced_outcomes_at_explicit_threshold():
    from training.evaluate import evaluate_predictions
    metrics = evaluate_predictions([0, 0, 0, 1], [0.1, 0.2, 0.9, 0.8], threshold=0.75)
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["fraud_prevalence"] == 0.25
    assert metrics["average_precision"] == 0.5
    assert metrics["roc_auc"] == pytest.approx(2 / 3)


def test_auc_validation_requires_real_supplied_holdout():
    from scripts.validate_auc import validate_auc
    with pytest.raises(ValueError, match="held-out"):
        validate_auc("unused-model-uri")


def test_mlflow_probability_signature_matches_raw_input_prediction(tmp_path, monkeypatch):
    import mlflow
    from training.train import log_to_mlflow, train_and_evaluate
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "synthetic-test-only")
    monkeypatch.delenv("MLFLOW_REGISTER_MODEL", raising=False)
    result = train_and_evaluate(fixture_data(), n_splits=3, n_estimators=2)
    model_uri = log_to_mlflow(result)
    loaded = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded.predict(result["input_example"])
    np.testing.assert_allclose(predictions, result["pipeline"].predict_proba(result["input_example"]))
    assert predictions.shape == (5, 2)
    from scripts.validate_auc import validate_auc
    holdout = generate_synthetic_data(50, fraud_ratio=0.2, random_state=43)
    holdout_path = tmp_path / "explicit-synthetic-holdout.csv"
    holdout.to_csv(holdout_path, index=False)
    score, passed = validate_auc(model_uri, threshold=0.0, data_path=holdout_path)
    assert isinstance(score, float) and 0 <= score <= 1
    assert passed is True


@pytest.mark.parametrize("labels, probabilities", [([0, 0], [.1, .2]), ([0, 1], [.1, np.nan]), ([0, 1], [.1, 1.1])])
def test_evaluation_rejects_undefined_or_invalid_scores(labels, probabilities):
    from training.evaluate import evaluate_predictions
    with pytest.raises(ValueError):
        evaluate_predictions(labels, probabilities)


def test_training_holdout_is_excluded_from_final_scaler():
    from training.train import train_and_evaluate
    df = fixture_data()
    train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df.Class)
    result = train_and_evaluate(df, n_splits=3, n_estimators=2)
    np.testing.assert_allclose(result["pipeline"].named_steps["scaler"].mean_, train.drop(columns="Class").mean())
    assert result["pipeline"].named_steps["scaler"].n_samples_seen_ == 96


def test_training_output_does_not_follow_serving_model_dir(tmp_path, monkeypatch):
    from training.train import main
    source = tmp_path / "fixture.csv"
    fixture_data().to_csv(source, index=False)
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "keep.txt").write_text("existing serving artifacts")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_CSV", str(source))
    monkeypatch.setenv("MODEL_DIR", str(serving))
    monkeypatch.setattr("training.train.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MODEL_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("CV_FOLDS", "3")
    monkeypatch.setenv("N_ESTIMATORS", "2")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    main()
    assert {p.name for p in serving.iterdir()} == {"keep.txt"}
    assert (tmp_path / "models/trained/xgb_model.joblib").is_file()


def test_training_records_xgboost_version_without_standard_distribution_metadata(monkeypatch):
    """Linux xgboost-cpu exposes xgboost but has no xgboost distribution metadata."""
    from importlib.metadata import PackageNotFoundError
    from training import train as train_module
    import xgboost
    installed_version = train_module.version

    def cpu_wheel_metadata(package):
        if package == "xgboost":
            raise PackageNotFoundError(package)
        return installed_version(package)

    monkeypatch.setattr(train_module, "version", cpu_wheel_metadata)
    result = train_module.train_and_evaluate(fixture_data(), n_splits=3, n_estimators=2)
    assert result["metrics"]["package_versions"]["xgboost"] == xgboost.__version__


def test_training_default_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from training import train as train_module
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    fixture_data().to_csv(project / "data/creditcard.csv", index=False)
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(train_module, "PROJECT_ROOT", project, raising=False)
    monkeypatch.delenv("DATA_CSV", raising=False)
    monkeypatch.delenv("MODEL_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("CV_FOLDS", "3")
    monkeypatch.setenv("N_ESTIMATORS", "2")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    train_module.main()
    assert (project / "models/trained/xgb_model.joblib").is_file()
    assert list(unrelated.iterdir()) == []


def test_smote_limits_minority_generation_to_half_majority():
    from training.preprocess import make_smote
    labels = np.array([0] * 90 + [1] * 10)
    features = np.arange(200, dtype=float).reshape(100, 2)
    _, resampled = make_smote(labels).fit_resample(features, labels)
    assert np.bincount(resampled).tolist() == [90, 45]


def test_smote_skips_already_sufficient_minority_ratio():
    from training.preprocess import make_smote
    assert make_smote([0] * 6 + [1] * 4) == "passthrough"


def test_bundle_contains_complete_raw_input_pipeline(tmp_path):
    from training.train import save_bundle, train_and_evaluate
    result = train_and_evaluate(fixture_data(), n_splits=5, n_estimators=2)
    save_bundle(result, tmp_path)
    restored = joblib.load(tmp_path / "pipeline.joblib")
    assert "smote" not in restored.named_steps
    np.testing.assert_allclose(restored.predict_proba(result["input_example"]),
                               result["pipeline"].predict_proba(result["input_example"]))


def test_optuna_selection_keeps_holdout_outside_all_trials(tmp_path):
    from training.tune import tune_model
    frame = fixture_data(160)
    result, study = tune_model(frame, n_trials=2, n_estimators=2, max_majority=60,
                               tracking_uri=(tmp_path / "mlruns").as_uri())
    assert len(study.trials) == 2
    assert result["metrics"]["test_rows"] == 32
    assert result["metrics"]["cv_folds"] == 5
    assert result["metrics"]["optimization_metric"] == "training_cv_roc_auc"
    _, holdout = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.Class)
    assert not set(result["training_indices"]) & set(holdout.index)
    assert result["metrics"]["train_rows"] == 86  # 60 legitimate + all 26 fraud training rows.
    assert 0.01 <= study.best_params["learning_rate"] <= 0.3
    assert 3 <= study.best_params["max_depth"] <= 10
    assert 1 <= study.best_params["scale_pos_weight"] <= 2
    assert result["metrics"]["target_auc"] == 0.95
    assert result["metrics"]["target_met"] is False  # Unidentified synthetic fixtures are not real quality proof.


def test_optional_feature_engineering_is_fitted_only_on_fold_training_rows():
    from training.train import fit_training_fold, train_and_evaluate
    frame = fixture_data()
    X, y = frame.drop(columns="Class"), frame.Class
    pipeline = fit_training_fold(X.iloc[:80], y.iloc[:80], n_estimators=2, feature_engineering=True)
    np.testing.assert_allclose(pipeline.named_steps["features"].pca_.mean_,
                               X.iloc[:80][[f"V{i}" for i in range(1, 29)]].mean())
    assert pipeline.named_steps["model"].n_features_in_ == 12
    result = train_and_evaluate(frame, n_estimators=2, feature_engineering=True)
    assert result["metrics"]["model_feature_names"] == ["Time", "logAmount"] + [f"PC{i}" for i in range(1, 11)]
    assert result["features"] == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def test_smoke_mlflow_registration_has_five_raw_csv_rows(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient
    from training.train import log_to_mlflow, train_and_evaluate
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "ci-smoke-unit-tests")
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "smoke")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "ci-smoke-unit-model")
    result = train_and_evaluate(fixture_data(), n_estimators=2, source_info={"data_kind": "synthetic"})
    uri = log_to_mlflow(result)
    info = mlflow.models.get_model_info(uri)
    client = MlflowClient()
    sample = pd.read_csv(client.download_artifacts(info.run_id, "inputs/sample.csv"))
    assert sample.shape == (5, 30)
    np.testing.assert_allclose(sample, result["input_example"])
    registered = client.get_model_version("ci-smoke-unit-model", "1")
    assert registered.tags["production_approved"] == "false"
    assert registered.tags["data_kind"] == "synthetic"


def test_smote_skips_when_integer_target_would_add_no_samples():
    from training.preprocess import make_smote
    assert make_smote([0] * 5 + [1] * 2) == "passthrough"


def test_smoke_registration_cannot_use_production_namespace(tmp_path, monkeypatch):
    from training.train import log_to_mlflow, train_and_evaluate
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "1")
    monkeypatch.setenv("MLFLOW_REGISTRATION_MODE", "smoke")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "fraud-production")
    result = train_and_evaluate(fixture_data(), n_estimators=2)
    with pytest.raises(ValueError, match="ci-smoke"):
        log_to_mlflow(result)
