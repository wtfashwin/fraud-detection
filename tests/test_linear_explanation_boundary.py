"""Tiny invented bundles test explanation semantics, never fraud-model quality."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from api.inference import load_bundle
from training.preprocess import FEATURE_NAMES


class CustomProbabilityLogistic(LogisticRegression):
    def predict_proba(self, data):
        return np.tile([0.25, 0.75], (len(data), 1))


@pytest.fixture
def invented_rows(monkeypatch):
    for key in ('MODEL_DIR', 'MODEL_PATH', 'SCALER_PATH', 'COLUMNS_PATH',
                'MLFLOW_MODEL_URI', 'DRIFT_BASELINE_PATH', 'MLFLOW_DRIFT_BASELINE_URI'):
        monkeypatch.delenv(key, raising=False)
    raw = pd.DataFrame(np.zeros((8, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    raw['Time'] = np.arange(8)
    raw['V1'] = [-3, -2, -1, -.5, .5, 1, 2, 3]
    raw['V2'] = [-1, 1, -1, 1, -1, 1, -1, 1]
    raw['Amount'] = [10, 20, 30, 40, 40, 30, 20, 10]
    return raw, [0, 0, 0, 0, 1, 1, 1, 1]


def save_pipeline(directory, raw, labels, estimator, reverse=False):
    model = Pipeline([('scaler', StandardScaler()), ('model', estimator)]).fit(raw, labels)
    if reverse:
        model.named_steps['model'].classes_ = model.named_steps['model'].classes_[::-1].copy()
    joblib.dump(model, directory / 'pipeline.joblib')
    joblib.dump(FEATURE_NAMES, directory / 'columns.joblib')
    return model


def select_legacy_artifact(directory, model, monkeypatch):
    joblib.dump(model.named_steps['model'], directory / 'custom.joblib')
    joblib.dump(model.named_steps['scaler'], directory / 'scaler.joblib')
    monkeypatch.setenv('MODEL_PATH', str(directory / 'custom.joblib'))


@pytest.mark.parametrize('kind,layout', [('svc', 'pipeline'), ('svc', 'legacy'),
    ('modified_huber', 'pipeline'), ('modified_huber', 'legacy'), ('subclass', 'pipeline')])
def test_scoring_compatible_custom_bundle_cannot_claim_log_odds(
        tmp_path, monkeypatch, invented_rows, kind, layout):
    raw, labels = invented_rows
    if kind == 'svc':
        estimator = SVC(kernel='linear', probability=True, random_state=42)
    elif kind == 'modified_huber':
        estimator = SGDClassifier(loss='modified_huber', alpha=.1, max_iter=2000, tol=1e-4, random_state=42)
    else:
        estimator = CustomProbabilityLogistic(solver='liblinear', random_state=42)
    model = save_pipeline(tmp_path, raw, labels, estimator)
    if layout == 'legacy':
        # Select an explicit custom artifact through the documented override path.
        select_legacy_artifact(tmp_path, model, monkeypatch)
    bundle = load_bundle(tmp_path)
    row = raw.iloc[6].tolist()
    label, probability = bundle.score(row)
    assert label == model.predict(raw.iloc[[6]])[0]
    assert probability == pytest.approx(model.predict_proba(raw.iloc[[6]])[0, 1])
    with pytest.raises(ValueError, match='Linear log-odds'):
        bundle.explain(row)


@pytest.mark.parametrize('settings', [{'solver': 'liblinear'},
    {'solver': 'lbfgs', 'multi_class': 'ovr'}, {'solver': 'lbfgs', 'multi_class': 'auto'}])
def test_supported_binary_logistic_pipeline_preserves_probability_link(tmp_path, invented_rows, settings):
    raw, labels = invented_rows
    model = save_pipeline(tmp_path, raw, labels, LogisticRegression(random_state=42, **settings))
    bundle = load_bundle(tmp_path)
    for index in [0, 6]:
        row = raw.iloc[index].tolist()
        label, probability = bundle.score(row)
        values, base, method = bundle.explain(row)
        assert label == model.predict(raw.iloc[[index]])[0]
        assert method == 'linear_log_odds_zero_reference'
        assert expit(sum(values) + base) == pytest.approx(probability, abs=1e-12)


@pytest.mark.parametrize('layout', ['pipeline', 'legacy'])
@pytest.mark.parametrize('reverse', [False, True])
def test_binary_multinomial_explanation_reconstructs_probability(
        tmp_path, monkeypatch, invented_rows, layout, reverse):
    raw, labels = invented_rows
    model = save_pipeline(tmp_path, raw, labels,
        LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42), reverse=reverse)
    if layout == 'legacy':
        select_legacy_artifact(tmp_path, model, monkeypatch)
    bundle = load_bundle(tmp_path)
    for index in [0, 6]:
        row = raw.iloc[index].tolist()
        _, probability = bundle.score(row)
        values, base, method = bundle.explain(row)
        assert method == 'linear_log_odds_zero_reference'
        assert expit(sum(values) + base) == pytest.approx(probability, abs=1e-12)


@pytest.mark.parametrize('kind', ['logistic_cv', 'logistic_cv_multinomial', 'sgd_log_loss', 'lda'])
def test_other_known_binary_logistic_links_are_preserved(tmp_path, invented_rows, kind):
    raw, labels = invented_rows
    if kind.startswith('logistic_cv'):
        settings = {'multi_class': 'multinomial'} if kind.endswith('multinomial') else {}
        estimator = LogisticRegressionCV(Cs=[.1, 1], cv=2, random_state=42, **settings)
    elif kind == 'sgd_log_loss':
        estimator = SGDClassifier(loss='log_loss', alpha=.1, max_iter=2000, tol=1e-4, random_state=42)
    else:
        estimator = LinearDiscriminantAnalysis()
    save_pipeline(tmp_path, raw, labels, estimator)
    bundle = load_bundle(tmp_path)
    for index in [0, 6]:
        row = raw.iloc[index].tolist()
        _, probability = bundle.score(row)
        values, base, method = bundle.explain(row)
        assert method == 'linear_log_odds_zero_reference'
        assert expit(sum(values) + base) == pytest.approx(probability, abs=1e-12)


def test_packaged_logistic_retains_its_explanation_link_on_invented_inputs(invented_rows):
    raw, _ = invented_rows
    bundle = load_bundle(Path(__file__).resolve().parents[1] / 'models')
    for row in raw.to_numpy():
        _, probability = bundle.score(row)
        values, base, method = bundle.explain(row)
        assert method == 'linear_log_odds_zero_reference'
        assert expit(sum(values) + base) == pytest.approx(probability, abs=1e-12)


def test_trained_xgboost_stays_on_the_tree_explanation_path(tmp_path, invented_rows):
    from xgboost import XGBClassifier

    raw, labels = invented_rows
    save_pipeline(tmp_path, raw, labels, XGBClassifier(n_estimators=2, max_depth=1,
        min_child_weight=0, learning_rate=.2, n_jobs=1, random_state=42))
    bundle = load_bundle(tmp_path)
    row = raw.iloc[6].tolist()
    _, probability = bundle.score(row)
    values, base, method = bundle.explain(row)
    assert method == 'tree_shap_raw_output'
    assert len(values) == len(FEATURE_NAMES)
    assert any(abs(value) > 1e-8 for value in values)
    assert expit(sum(values) + base) == pytest.approx(probability, abs=1e-6)
