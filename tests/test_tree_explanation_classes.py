"""Synthetic trees exercise class selection, not fraud model quality."""
import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from api.inference import InferenceBundle, load_bundle


TRAINING = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 0.], [2., 1.]])
LABELS = np.array([0, 0, 0, 1, 1, 1])
INPUT = [1., 0.]


def load_saved_tree(tmp_path, model, reversed_classes=False, labels=LABELS, weights=None):
    scaler = StandardScaler().fit(TRAINING)
    model.fit(scaler.transform(TRAINING), labels, sample_weight=weights)
    if reversed_classes:
        # Emulate a valid estimator whose stored class labels are reversed.
        model.classes_ = model.classes_[::-1]
    joblib.dump(model, tmp_path / "xgb_model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(["a", "b"], tmp_path / "columns.joblib")
    return load_bundle(tmp_path)


@pytest.mark.parametrize("estimator", [
    DecisionTreeClassifier(random_state=17),
    RandomForestClassifier(n_estimators=9, random_state=17),
])
@pytest.mark.parametrize("reversed_classes", [False, True])
def test_multi_output_tree_explains_the_probability_for_fraud(tmp_path, estimator, reversed_classes):
    bundle = load_saved_tree(tmp_path, estimator, reversed_classes)
    _, score = bundle.score(INPUT)
    values, base, method = bundle.explain(INPUT)
    assert method == "tree_shap_raw_output"
    assert sum(values) + base == pytest.approx(score)


@pytest.mark.parametrize("loss,scale", [("log_loss", 1), ("exponential", 2)])
@pytest.mark.parametrize("reversed_classes", [False, True])
def test_gradient_boosting_margin_explains_fraud_probability(tmp_path, loss, scale, reversed_classes):
    bundle = load_saved_tree(tmp_path, GradientBoostingClassifier(loss=loss, random_state=17), reversed_classes)
    _, score = bundle.score(INPUT)
    values, base, _ = bundle.explain(INPUT)
    reconstructed = 1 / (1 + np.exp(-scale * (sum(values) + base)))
    assert reconstructed == pytest.approx(score)


@pytest.mark.parametrize("loss,scale", [("log_loss", 1), ("exponential", 2)])
@pytest.mark.parametrize("reversed_classes", [False, True])
@pytest.mark.parametrize("labels,weights", [
    (np.array([0, 0, 0, 0, 1, 1]), None),
    (LABELS, np.array([3., 3., 3., 1., 1., 1.])),
])
def test_imbalanced_gradient_boosting_explains_fraud_probability(
        tmp_path, loss, scale, reversed_classes, labels, weights):
    bundle = load_saved_tree(tmp_path, GradientBoostingClassifier(
        loss=loss, n_estimators=4, max_depth=1, random_state=17),
                             reversed_classes, labels, weights)
    for features in ([0., 0.], INPUT, [2., 1.]):
        _, score = bundle.score(features)
        values, base, _ = bundle.explain(features)
        reconstructed = 1 / (1 + np.exp(-scale * (sum(values) + base)))
        assert reconstructed == pytest.approx(score)


@pytest.mark.parametrize("loss", ["log_loss", "exponential"])
@pytest.mark.parametrize("reversed_classes", [False, True])
@pytest.mark.parametrize("strategy,constant", [
    ("prior", None), ("uniform", None), ("most_frequent", None), ("constant", 0),
])
def test_gradient_boosting_constant_initializers_explain_actual_margin(
        tmp_path, loss, reversed_classes, strategy, constant):
    estimator = GradientBoostingClassifier(
        loss=loss, n_estimators=4, max_depth=1, random_state=17,
        init=DummyClassifier(strategy=strategy, constant=constant))
    bundle = load_saved_tree(tmp_path, estimator, reversed_classes,
                             labels=np.array([0, 0, 0, 0, 1, 1]))
    for features in ([0., 0.], [2., 1.]):
        values, base, _ = bundle.explain(features)
        margin = bundle.model.decision_function(bundle.prepare(features))[0]
        if reversed_classes:
            margin = -margin
        assert sum(values) + base == pytest.approx(margin)


@pytest.mark.parametrize("loss", ["log_loss", "exponential"])
def test_gradient_boosting_random_initialization_cannot_be_explained_as_constant(tmp_path, loss):
    estimator = GradientBoostingClassifier(
        loss=loss, n_estimators=4, max_depth=1, random_state=17,
        init=DummyClassifier(strategy="stratified", random_state=17))
    bundle = load_saved_tree(tmp_path, estimator)
    with pytest.raises(ValueError, match="initial"):
        bundle.explain(INPUT)


@pytest.mark.parametrize("estimator", [
    DecisionTreeClassifier(random_state=17),
    RandomForestClassifier(n_estimators=9, random_state=17),
])
def test_one_class_fraud_tree_still_explains_probability(tmp_path, estimator):
    bundle = load_saved_tree(tmp_path, estimator, labels=np.ones(len(TRAINING), dtype=int))
    assert bundle.score(INPUT) == (1, 1.0)
    values, base, _ = bundle.explain(INPUT)
    assert sum(values) + base == pytest.approx(1.0)


def test_pipeline_multi_output_tree_explains_reversed_fraud_class(tmp_path):
    import pandas as pd
    estimator = DecisionTreeClassifier(random_state=17)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)]).fit(
        pd.DataFrame(TRAINING, columns=["a", "b"]), LABELS)
    estimator.classes_ = estimator.classes_[::-1]
    joblib.dump(pipeline, tmp_path / "pipeline.joblib")
    joblib.dump(["a", "b"], tmp_path / "columns.joblib")
    bundle = load_bundle(tmp_path)
    _, score = bundle.score(INPUT)
    values, base, _ = bundle.explain(INPUT)
    assert sum(values) + base == pytest.approx(score)


@pytest.mark.parametrize("loss,scale", [("log_loss", 1), ("exponential", 2)])
@pytest.mark.parametrize("reversed_classes", [False, True])
def test_pipeline_gradient_boosting_uses_correct_initial_margin(tmp_path, loss, scale, reversed_classes):
    import pandas as pd
    estimator = GradientBoostingClassifier(loss=loss, n_estimators=4, max_depth=1, random_state=17)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)]).fit(
        pd.DataFrame(TRAINING, columns=["a", "b"]), np.array([0, 0, 0, 0, 1, 1]))
    if reversed_classes:
        estimator.classes_ = estimator.classes_[::-1]
    joblib.dump(pipeline, tmp_path / "pipeline.joblib")
    joblib.dump(["a", "b"], tmp_path / "columns.joblib")
    bundle = load_bundle(tmp_path)
    for features in ([0., 0.], [2., 1.]):
        _, score = bundle.score(features)
        values, base, _ = bundle.explain(features)
        assert 1 / (1 + np.exp(-scale * (sum(values) + base))) == pytest.approx(score)


def test_xgboost_single_margin_explains_fraud_probability(tmp_path):
    from xgboost import XGBClassifier
    bundle = load_saved_tree(tmp_path, XGBClassifier(n_estimators=9, n_jobs=1, random_state=17))
    _, score = bundle.score(INPUT)
    values, base, _ = bundle.explain(INPUT)
    assert 1 / (1 + np.exp(-(sum(values) + base))) == pytest.approx(score)


@pytest.mark.parametrize("reversed_classes", [False, True])
def test_histogram_boosting_single_margin_explains_fraud_probability(tmp_path, reversed_classes):
    estimator = HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=1, random_state=17)
    bundle = load_saved_tree(tmp_path, estimator, reversed_classes)
    _, score = bundle.score(INPUT)
    values, base, _ = bundle.explain(INPUT)
    assert score != pytest.approx(0.5)
    assert 1 / (1 + np.exp(-(sum(values) + base))) == pytest.approx(score)


def test_unknown_single_output_is_rejected_instead_of_guessing(monkeypatch):
    import shap
    class Tree:
        classes_ = np.array([1, 0])
    class Explainer:
        model = type("Details", (), {"tree_output": "unknown"})()
        expected_value = 0.5
        def shap_values(self, data):
            return np.array([[1., 2.]])
    monkeypatch.setattr(shap, "TreeExplainer", lambda model: Explainer())
    scaler = StandardScaler().fit(TRAINING)
    bundle = InferenceBundle(Tree(), scaler, ["a", "b"])
    with pytest.raises(ValueError, match="output"):
        bundle.explain(INPUT)


def test_multi_output_with_ambiguous_base_is_rejected(monkeypatch):
    import shap
    class Tree:
        classes_ = np.array([0, 1])
    class Explainer:
        expected_value = 0.5
        def shap_values(self, data):
            return np.array([[[1., 2.], [3., 4.]]])
    monkeypatch.setattr(shap, "TreeExplainer", lambda model: Explainer())
    scaler = StandardScaler().fit(TRAINING)
    bundle = InferenceBundle(Tree(), scaler, ["a", "b"])
    with pytest.raises(ValueError, match="shape"):
        bundle.explain(INPUT)
