import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training.predict import FraudDetector


@pytest.fixture
def detector(tmp_path):
    X = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [4., 3., 2., 1.]})
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(solver="liblinear").fit(scaler.transform(X), [0, 0, 1, 1])
    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    (tmp_path / "features.json").write_text(json.dumps(["a", "b"]))
    return FraudDetector(tmp_path / "model.joblib", tmp_path / "scaler.joblib", tmp_path / "features.json")


def test_single_prediction_rejects_multiple_rows(detector):
    with pytest.raises(ValueError, match="one"):
        detector.predict(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_single_prediction_rejects_extra_columns(detector):
    with pytest.raises(ValueError, match="Unexpected"):
        detector.predict({"a": 1, "b": 2, "Class": 1})


@pytest.mark.parametrize("value", [np.nan, np.inf, "not-numeric"])
def test_single_prediction_rejects_invalid_values(detector, value):
    with pytest.raises(ValueError, match="numeric|finite"):
        detector.predict({"a": value, "b": 2})


def test_single_prediction_reorders_features(detector):
    assert detector.predict({"a": 1, "b": 2}) == detector.predict({"b": 2, "a": 1})


def test_prediction_module_cli_accepts_newly_trained_bundle(tmp_path):
    import os
    import subprocess
    import sys
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.train import save_bundle, train_and_evaluate

    # Explicitly synthetic training checks artifact selection, not fraud quality.
    result = train_and_evaluate(generate_synthetic_data(120, fraud_ratio=0.2), n_splits=3, n_estimators=2)
    save_bundle(result, tmp_path)
    environment = os.environ.copy()
    environment["MODEL_DIR"] = str(tmp_path)
    environment.pop("MODEL_PATH", None)
    completed = subprocess.run([sys.executable, "-m", "training.predict"], env=environment,
                               text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert "Fraud Probability:" in completed.stdout


@pytest.fixture
def engineered_bundle(tmp_path):
    from sklearn.pipeline import Pipeline
    from scripts.generate_synthetic_data import generate_synthetic_data
    from training.features import FraudFeatureEngineer

    # This fixture deliberately contains only the authoritative raw-input pipeline,
    # never a legacy scaler/model pair that could conceal double preprocessing.
    frame = generate_synthetic_data(100, fraud_ratio=0.2, random_state=23)
    features = frame.drop(columns='Class')
    pipeline = Pipeline([('features', FraudFeatureEngineer()), ('scaler', StandardScaler()),
                         ('model', LogisticRegression(solver='liblinear'))]).fit(features, frame.Class)
    joblib.dump(pipeline, tmp_path / 'pipeline.joblib')
    joblib.dump(list(features.columns), tmp_path / 'columns.joblib')
    (tmp_path / 'feature_names.json').write_text(json.dumps(list(features.columns)))
    return tmp_path, pipeline, frame


def test_predictor_directory_prefers_full_engineered_pipeline(engineered_bundle, monkeypatch):
    directory, pipeline, frame = engineered_bundle
    monkeypatch.delenv('MODEL_PATH', raising=False)
    monkeypatch.delenv('MLFLOW_MODEL_URI', raising=False)
    detector = FraudDetector(model_dir=directory)
    row = frame.drop(columns='Class').iloc[[0]]
    label, score = detector.predict(row)
    assert label == pipeline.predict(row)[0]
    assert score == pytest.approx(pipeline.predict_proba(row)[0, 1])
    assert len(detector.features) == 30


def test_prediction_cli_accepts_pipeline_only_engineered_bundle(engineered_bundle):
    import os
    import subprocess
    import sys
    directory, _, _ = engineered_bundle
    environment = dict(os.environ, MODEL_DIR=str(directory))
    for key in ['MODEL_PATH', 'SCALER_PATH', 'COLUMNS_PATH', 'MLFLOW_MODEL_URI']:
        environment.pop(key, None)
    completed = subprocess.run([sys.executable, '-m', 'training.predict'], env=environment,
                               text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert 'Fraud Probability:' in completed.stdout


def test_evaluation_cli_scores_raw_input_through_authoritative_pipeline(engineered_bundle, monkeypatch):
    from training.evaluate import main, evaluate_predictions
    directory, pipeline, frame = engineered_bundle
    source = directory / 'synthetic-holdout.csv'
    frame.to_csv(source, index=False)
    output = directory / 'evaluation'
    monkeypatch.setenv('MODEL_DIR', str(directory))
    monkeypatch.setenv('EVAL_DATA_CSV', str(source))
    monkeypatch.setenv('EVALUATION_OUTPUT_DIR', str(output))
    monkeypatch.setenv('EVAL_DATA_KIND', 'synthetic')
    for key in ['MODEL_PATH', 'SCALER_PATH', 'COLUMNS_PATH', 'MLFLOW_MODEL_URI']:
        monkeypatch.delenv(key, raising=False)
    main()
    actual = json.loads((output / 'metrics.json').read_text())
    expected = evaluate_predictions(frame.Class, pipeline.predict_proba(frame.drop(columns='Class'))[:, 1])
    for key in ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']:
        assert actual[key] == pytest.approx(expected[key])
    assert actual['data_kind'] == 'synthetic'
    assert (output / 'evaluation_curves.png').is_file()
