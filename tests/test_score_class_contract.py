"""Synthetic model class boundaries, not claims about fraud model quality."""
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from api.inference import InferenceBundle, load_bundle


class ContradictoryFraudOnlyEstimator:
    classes_ = np.array([1])
    n_features_in_ = 1

    def predict(self, data):
        return np.array([0])

    def predict_proba(self, data):
        return np.array([[1.0]])


def test_multiclass_model_is_rejected_at_startup(tmp_path):
    raw = np.arange(-4.0, 5.0).reshape(-1, 1)
    scaler = StandardScaler().fit(raw)
    model = LogisticRegression(random_state=1).fit(
        scaler.transform(raw), [0, 0, 0, 1, 1, 1, 2, 2, 2]
    )
    # Its zero-input prediction is 1, so checking just that label would pass.
    assert model.predict(scaler.transform([[0.0]])).tolist() == [1]
    joblib.dump(model, tmp_path / "logistic_model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(["x"], tmp_path / "columns.joblib")

    with pytest.raises(ValueError, match="class"):
        load_bundle(tmp_path)


@pytest.mark.parametrize("classes", [[1, 2], [2, 1], [0], [], [1, 1], 1, [[0, 1]]])
def test_invalid_class_metadata_cannot_return_score(classes):
    class Estimator:
        classes_ = np.asarray(classes)
        n_features_in_ = 1

        def predict(self, data):
            return np.array([1])

        def predict_proba(self, data):
            return np.array([[0.4, 0.6]])

    scaler = StandardScaler().fit([[0.0], [1.0]])
    bundle = InferenceBundle(Estimator(), scaler, ["x"])
    with pytest.raises(ValueError, match="class"):
        bundle.score([0.0])


def test_fraud_only_model_still_scores():
    raw = np.array([[0.0], [1.0], [2.0]])
    scaler = StandardScaler().fit(raw)
    model = DecisionTreeClassifier(random_state=1).fit(
        scaler.transform(raw), [1, 1, 1]
    )
    assert InferenceBundle(model, scaler, ["x"]).score([0.0]) == (1, 1.0)


def test_prediction_outside_model_classes_cannot_return_score():
    scaler = StandardScaler().fit([[0.0], [1.0]])
    bundle = InferenceBundle(ContradictoryFraudOnlyEstimator(), scaler, ["x"])
    with pytest.raises(ValueError, match="class"):
        bundle.score([0.0])


def test_prediction_outside_model_classes_is_rejected_at_startup(tmp_path):
    scaler = StandardScaler().fit([[0.0], [1.0]])
    joblib.dump(ContradictoryFraudOnlyEstimator(), tmp_path / "logistic_model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(["x"], tmp_path / "columns.joblib")
    with pytest.raises(ValueError, match="class"):
        load_bundle(tmp_path)
