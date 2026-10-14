"""API regression tests using synthetic fixtures and an isolated SQLite database.

These tests do not measure fraud detection accuracy or prove Redis/Postgres health.
"""
import importlib
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def synthetic_bundle(tmp_path):
    from api.inference import load_bundle
    columns = ["Time", "V1", "Amount"]
    raw = pd.DataFrame([[0., -2., 1.], [1., -1., 2.], [2., 1., 4.], [3., 2., 5.]], columns=columns)
    scaler = StandardScaler().fit(raw)
    model = LogisticRegression(solver="liblinear").fit(scaler.transform(raw), [0, 0, 1, 1])
    # Fix known coefficients after fitting to obtain hand-checkable synthetic log odds.
    model.coef_ = np.array([[0., 1., 0.]])
    model.intercept_ = np.array([0.])
    joblib.dump(model, tmp_path / "logistic_model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(columns, tmp_path / "columns.joblib")
    return load_bundle(tmp_path)


@pytest.fixture
def api_client(monkeypatch, synthetic_bundle):
    module = importlib.import_module("api.app")
    from db.models import Explanation
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Explanation.__table__.create(engine)
    monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(module, "BUNDLE", synthetic_bundle)
    monkeypatch.setattr(module, "check_broker", lambda: True)
    monkeypatch.setattr(module.celery_app, "send_task", lambda *a, **k: SimpleNamespace(id="synthetic-task"))
    # Startup model load is tested separately; this fixture isolates request handling.
    with TestClient(module.app, raise_server_exceptions=False) as client:
        yield client, module, engine
    engine.dispose()


def test_liveness_remains_up_when_broker_fails(api_client, monkeypatch):
    client, module, _ = api_client
    monkeypatch.setattr(module, "check_broker", lambda: False)
    assert client.get("/health").status_code == 200
    assert client.get("/status").json() == {"status": "UP"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["dependencies"]["redis_broker"] == "DOWN"


def test_prediction_returns_real_positive_probability_and_pending_state(api_client):
    client, _, _ = api_client
    response = client.post("/predict", json={"features": [1., -1., 2.]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] == 0
    assert result["score"] == pytest.approx(1 / (1 + np.exp(1 / np.sqrt(2.5))))
    assert result["explanation_status"] == "Calculation queued"
    assert response.headers["X-Correlation-ID"] == result["correlation_id"]
    pending = client.get(f"/explain/{result['transaction_id']}")
    assert pending.status_code == 202
    assert pending.json()["status"] == "PENDING"


@pytest.mark.parametrize("features", [[1, 2], [1, "2", 3], [1, True, 3], [1, None, 3], [1, {}, 3]])
def test_invalid_features_rejected_before_queue(api_client, features):
    client, _, _ = api_client
    assert client.post("/predict", json={"features": features}).status_code == 422


def test_nonfinite_input_rejected(api_client):
    client, _, _ = api_client
    response = client.post("/predict", content='{"features": [1, NaN, 3]}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_queue_failure_is_503_and_persisted_failed(api_client, monkeypatch):
    client, module, _ = api_client
    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic broker unavailable")
    monkeypatch.setattr(module.celery_app, "send_task", unavailable)
    response = client.post("/predict", json={"transaction_id": "synthetic-unavailable", "features": [1, -1, 2]})
    assert response.status_code == 503
    assert client.get("/explain/synthetic-unavailable").json()["detail"]["status"] == "FAILED"


def test_duplicate_id_does_not_overwrite_original(api_client):
    client, _, _ = api_client
    payload = {"transaction_id": "synthetic-duplicate", "features": [1, -1, 2]}
    assert client.post("/predict", json=payload).status_code == 200
    assert client.post("/predict", json=payload).status_code == 409


def test_database_failure_does_not_claim_queued(api_client, monkeypatch):
    client, module, _ = api_client
    class Unavailable:
        def begin(self):
            raise ConnectionError("synthetic database unavailable")
    monkeypatch.setattr(module, "engine", Unavailable())
    assert client.post("/predict", json={"features": [1, -1, 2]}).status_code == 503


def test_model_info_exposes_saved_feature_order(api_client):
    client, _, _ = api_client
    info = client.get("/model-info").json()
    assert info["feature_names"] == ["Time", "V1", "Amount"]
    assert info["feature_count"] == 3


def test_bundle_resolves_explicit_directory_after_cwd_change(tmp_path, synthetic_bundle, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert synthetic_bundle.score([1, -1, 2])[0] == 0


def test_probability_preserves_scientific_notation(synthetic_bundle):
    # At this synthetic extreme, probability is tiny; string parsing loses its exponent.
    prediction, score = synthetic_bundle.score([-50., -200., -200.])
    assert prediction == 0
    assert 0 < score < 1e-30


def test_worker_persists_real_explanation_in_same_table(api_client, monkeypatch, synthetic_bundle):
    from api import worker
    from db.models import Explanation
    from sqlalchemy import select
    client, _, engine = api_client
    response = client.post("/predict", json={"transaction_id": "synthetic-worker", "features": [1, -1, 2]})
    assert response.status_code == 200
    monkeypatch.setattr(worker, "engine", engine)
    monkeypatch.setattr(worker, "get_bundle", lambda: synthetic_bundle)
    worker.compute_shap.run("synthetic-worker", [1, -1, 2], "synthetic-correlation")
    explanation = client.get("/explain/synthetic-worker")
    assert explanation.status_code == 200
    data = explanation.json()
    assert data["status"] == "COMPLETED"
    assert data["feature_names"] == ["Time", "V1", "Amount"]
    # For logistic regression these are exact log-odds contributions from zero scaled reference.
    log_odds = sum(data["shap_values"]) + data["base_value"]
    probability = 1 / (1 + np.exp(-log_odds))
    assert probability == pytest.approx(response.json()["score"])
    assert data["explanation_method"] == "linear_log_odds_zero_reference"
    # A duplicate Celery delivery leaves a completed explanation intact.
    worker.compute_shap.run("synthetic-worker", [100, 100, 100], "synthetic-duplicate-delivery")
    assert client.get("/explain/synthetic-worker").json()["shap_values"] == data["shap_values"]


def test_worker_final_failure_is_visible(api_client, monkeypatch):
    from api import worker
    client, _, engine = api_client
    client.post("/predict", json={"transaction_id": "synthetic-worker-failed", "features": [1, -1, 2]})
    monkeypatch.setattr(worker, "engine", engine)
    def failed_bundle():
        raise ValueError("synthetic artifact failure")
    monkeypatch.setattr(worker, "get_bundle", failed_bundle)
    worker.compute_shap.push_request(retries=worker.compute_shap.max_retries)
    try:
        with pytest.raises(ValueError):
            worker.compute_shap.run("synthetic-worker-failed", [1, -1, 2], "synthetic-correlation")
    finally:
        worker.compute_shap.pop_request()
    assert client.get("/explain/synthetic-worker-failed").json()["detail"]["status"] == "FAILED"


def test_demo_serves_only_public_assets(api_client):
    client, _, _ = api_client
    assert client.get("/demo/").status_code == 200
    asset = client.get("/demo/app.mjs")
    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]
    assert client.get("/demo/models/logistic_model.joblib").status_code == 404


def test_overflowing_integer_input_is_validation_error(api_client):
    client, _, _ = api_client
    response = client.post("/predict", json={"features": [1, 10 ** 400, 3]})
    assert response.status_code == 422


def test_readiness_requires_schema_not_just_connection(api_client):
    from db.models import Explanation
    client, _, engine = api_client
    Explanation.__table__.drop(engine)
    assert client.get("/ready").status_code == 503


def test_missing_model_keeps_liveness_but_rejects_scoring(api_client, monkeypatch):
    client, module, _ = api_client
    monkeypatch.setattr(module, "BUNDLE", None)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"features": [1, -1, 2]}).status_code == 503


def test_remote_standalone_estimator_rejected_instead_of_scoring_raw(tmp_path, synthetic_bundle, monkeypatch):
    from api.inference import load_bundle
    import mlflow.sklearn
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/synthetic-model@production")
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: synthetic_bundle.model)
    with pytest.raises(ValueError, match="pipeline"):
        load_bundle(tmp_path)


def test_lost_dispatch_ack_cannot_overwrite_completed_explanation(api_client, monkeypatch):
    from db.models import Explanation
    from sqlalchemy import update
    client, module, engine = api_client
    def completed_then_unconfirmed(*args, **kwargs):
        with engine.begin() as connection:
            connection.execute(update(Explanation).where(Explanation.transaction_id == "synthetic-lost-ack")
                               .values(status="COMPLETED", shap_values=[0., 0., 0.],
                                       feature_names=["Time", "V1", "Amount"]))
        raise ConnectionError("synthetic acknowledgement lost after completion")
    monkeypatch.setattr(module.celery_app, "send_task", completed_then_unconfirmed)
    response = client.post("/predict", json={"transaction_id": "synthetic-lost-ack", "features": [1, -1, 2]})
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "COMPLETED"
    assert client.get("/explain/synthetic-lost-ack").status_code == 200


def test_worker_retry_publish_failure_marks_failed(api_client, monkeypatch):
    from api import worker
    from celery.exceptions import Reject
    client, _, engine = api_client
    client.post("/predict", json={"transaction_id": "synthetic-retry-lost", "features": [1, -1, 2]})
    monkeypatch.setattr(worker, "engine", engine)
    def failed_bundle():
        raise ValueError("synthetic artifact failure")
    def failed_retry(**kwargs):
        raise Reject("synthetic retry broker unavailable", requeue=False)
    monkeypatch.setattr(worker, "get_bundle", failed_bundle)
    monkeypatch.setattr(worker.compute_shap, "retry", failed_retry)
    with pytest.raises(Reject):
        worker.compute_shap.run("synthetic-retry-lost", [1, -1, 2], "synthetic-correlation")
    assert client.get("/explain/synthetic-retry-lost").json()["detail"]["status"] == "FAILED"


def test_worker_successfully_scheduled_retry_stays_pending(api_client, monkeypatch):
    from api import worker
    from celery.exceptions import Retry
    client, _, engine = api_client
    client.post("/predict", json={"transaction_id": "synthetic-retry-ok", "features": [1, -1, 2]})
    monkeypatch.setattr(worker, "engine", engine)
    def failed_bundle():
        raise ValueError("synthetic temporary artifact failure")
    def scheduled_retry(**kwargs):
        raise Retry()
    monkeypatch.setattr(worker, "get_bundle", failed_bundle)
    monkeypatch.setattr(worker.compute_shap, "retry", scheduled_retry)
    with pytest.raises(Retry):
        worker.compute_shap.run("synthetic-retry-ok", [1, -1, 2], "synthetic-correlation")
    assert client.get("/explain/synthetic-retry-ok").status_code == 202


def test_readiness_rejects_historical_partial_schema(api_client):
    from db.models import Explanation
    from sqlalchemy import text
    client, _, engine = api_client
    Explanation.__table__.drop(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE shap_explanations (transaction_id VARCHAR(128) PRIMARY KEY, shap_values JSON)"))
    assert client.get("/ready").status_code == 503
