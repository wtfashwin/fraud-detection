"""Synthetic inference fixtures test correctness, never fraud-model quality."""
import numpy as np
import pytest


def test_missing_bundle_fails_with_clear_error(tmp_path):
    from api.inference import load_bundle
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


def test_mismatched_artifacts_rejected(tmp_path):
    import joblib
    from api.inference import load_bundle
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    raw = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    joblib.dump(LogisticRegression(solver="liblinear").fit(raw, [0, 0, 1, 1]), tmp_path / "logistic_model.joblib")
    joblib.dump(StandardScaler().fit(raw), tmp_path / "scaler.joblib")
    joblib.dump(["only_one_column"], tmp_path / "columns.joblib")
    with pytest.raises(ValueError, match="feature"):
        load_bundle(tmp_path)


def test_positive_probability_uses_class_label_not_position():
    from api.inference import InferenceBundle
    from sklearn.preprocessing import StandardScaler
    # Explicit estimator boundary double exercises reverse class ordering and tiny values.
    class ReversedEstimator:
        classes_ = np.array([1, 0])
        n_features_in_ = 2
        def predict(self, data):
            return np.array([0])
        def predict_proba(self, data):
            return np.array([[1e-12, 1.0 - 1e-12]])
    scaler = StandardScaler().fit(np.array([[0., 0.], [1., 1.]]))
    bundle = InferenceBundle(ReversedEstimator(), scaler, ["a", "b"])
    assert bundle.score([0., 0.]) == (0, 1e-12)


def test_invalid_model_probability_is_not_replaced_with_half():
    from api.inference import InferenceBundle
    from sklearn.preprocessing import StandardScaler
    class BrokenEstimator:
        classes_ = np.array([0, 1])
        n_features_in_ = 2
        def predict(self, data):
            return np.array([0])
        def predict_proba(self, data):
            return np.array([[float("nan"), float("nan")]])
    scaler = StandardScaler().fit(np.array([[0., 0.], [1., 1.]]))
    bundle = InferenceBundle(BrokenEstimator(), scaler, ["a", "b"])
    with pytest.raises(ValueError, match="probability"):
        bundle.score([0., 0.])


def test_linear_explanation_tracks_positive_class_when_classes_reversed():
    from api.inference import InferenceBundle
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.special import expit
    model = LogisticRegression(solver="liblinear").fit(
        np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]]), [0, 0, 1, 1])
    # Reversing the labels preserves the real estimator's probability semantics.
    model.classes_ = model.classes_[::-1].copy()
    scaler = StandardScaler().fit(np.array([[-1., -1.], [1., 1.]]))
    bundle = InferenceBundle(model, scaler, ["a", "b"])
    values, base_value, method = bundle.explain([2., 1.])
    assert values == pytest.approx(-np.array([2., 1.]) * model.coef_[0])
    assert base_value == pytest.approx(-model.intercept_[0])
    assert expit(sum(values) + base_value) == pytest.approx(bundle.score([2., 1.])[1])
    assert method == "linear_log_odds_zero_reference"


def test_registered_training_pipeline_skips_smote_during_explanation():
    from api.inference import InferenceBundle
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    # Synthetic data checks preprocessing parity, not real fraud model quality.
    data = pd.DataFrame([[i, i * 2] for i in range(10)], columns=["a", "b"])
    model = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(k_neighbors=1, random_state=42)),
                      ("model", LogisticRegression(solver="liblinear"))]).fit(data, [0] * 6 + [1] * 4)
    bundle = InferenceBundle(model, model.named_steps["scaler"], ["a", "b"], raw_pipeline=True)
    prediction, score = bundle.score([7., 14.])
    values, base_value, method = bundle.explain([7., 14.])
    reconstructed = 1 / (1 + np.exp(-(sum(values) + base_value)))
    assert reconstructed == pytest.approx(score)
    assert prediction == 1


def test_complete_pipeline_bundle_preserves_raw_inputs_and_transformed_explanation_names(tmp_path):
    import joblib
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from training.features import FraudFeatureEngineer
    from api.inference import load_bundle
    from api.drift import create_baseline
    from training.preprocess import FEATURE_NAMES
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.normal(size=(40, 30)), columns=FEATURE_NAMES)
    raw["Amount"] = rng.uniform(0, 1000, size=40)
    raw["Time"] = np.arange(40)
    model = Pipeline([("features", FraudFeatureEngineer()), ("scaler", StandardScaler()),
                      ("model", LogisticRegression(solver="liblinear"))]).fit(raw, [0, 1] * 20)
    joblib.dump(model, tmp_path / "pipeline.joblib")
    joblib.dump(FEATURE_NAMES, tmp_path / "columns.joblib")
    import json
    (tmp_path / "drift_baseline.json").write_text(json.dumps(create_baseline(raw, FEATURE_NAMES, source={"data_kind": "synthetic"})))
    bundle = load_bundle(tmp_path)
    assert bundle.raw_pipeline is True
    assert bundle.columns == FEATURE_NAMES
    assert bundle.explanation_columns == ["Time", "logAmount"] + [f"PC{i}" for i in range(1, 11)]
    prediction, score = bundle.score(raw.iloc[0].tolist())
    values, base_value, _ = bundle.explain(raw.iloc[0].tolist())
    assert len(values) == len(bundle.explanation_columns) == 12
    assert 1 / (1 + np.exp(-(sum(values) + base_value))) == pytest.approx(score)
    assert bundle.drift_baseline["reference_rows"] == 40
