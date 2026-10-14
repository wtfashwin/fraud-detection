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
