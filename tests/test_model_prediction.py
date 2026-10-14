import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from predict_single import FraudDetector


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
