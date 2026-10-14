"""Canonical Celery task: identical preprocessing and shared explanation persistence."""
from functools import lru_cache
import logging
import os

from celery.signals import worker_ready
from celery.exceptions import Retry
from prometheus_client import Counter, Histogram, start_http_server
from sqlalchemy import select, update

from api.inference import load_bundle
from api.queue import celery_app
from db.db import engine
from db.models import Explanation

app = celery_app
logger = logging.getLogger(__name__)
task_duration = Histogram("xai_task_duration_seconds", "Explanation task duration")
task_success = Counter("xai_task_success_total", "Completed explanation tasks")
task_failure = Counter("xai_task_failures_total", "Failed explanation task attempts")


@lru_cache(maxsize=1)
def get_bundle():
    return load_bundle()


@worker_ready.connect
def start_metrics(**kwargs):
    # API imports only api.queue; starting the API must never bind worker ports.
    try:
        start_http_server(int(os.getenv("WORKER_METRICS_PORT", "8001")))
    except OSError:
        logger.warning("Worker metrics port unavailable")


def persist_failure(transaction_id):
    try:
        with engine.begin() as connection:
            connection.execute(update(Explanation).where(
                Explanation.transaction_id == transaction_id, Explanation.status != "COMPLETED")
                .values(status="FAILED"))
    except Exception:
        logger.error("Could not persist explanation failure for %s", transaction_id)


@app.task(bind=True, name="xai_tasks.compute_shap", max_retries=3, acks_late=True,
          reject_on_worker_lost=True)
def compute_shap(self, transaction_id: str, features: list, correlation_id: str | None = None):
    try:
        with engine.connect() as connection:
            row = connection.execute(select(Explanation.status).where(
                Explanation.transaction_id == transaction_id)).first()
        if row is None:
            raise ValueError("No persisted transaction exists for explanation task")
        if row.status == "COMPLETED":
            return {"transaction_id": transaction_id, "status": "COMPLETED"}
        with task_duration.time():
            bundle = get_bundle()
            values, base_value, method = bundle.explain(features)
            with engine.begin() as connection:
                connection.execute(update(Explanation).where(Explanation.transaction_id == transaction_id)
                                   .values(status="COMPLETED", shap_values=values, feature_names=bundle.explanation_columns,
                                           base_value=base_value, explanation_method=method))
        task_success.inc()
        return {"transaction_id": transaction_id, "status": "COMPLETED"}
    except Exception as exc:
        task_failure.inc()
        logger.warning("Explanation attempt failed for %s (%s)", transaction_id, type(exc).__name__)
        if self.request.retries < self.max_retries:
            try:
                raise self.retry(exc=exc, countdown=min(60, 2 ** self.request.retries))
            except Retry:
                raise
            except Exception:
                # Failed retry publication can reject the original task without requeueing.
                persist_failure(transaction_id)
                raise
        persist_failure(transaction_id)
        # Preserve Celery's failure state, rather than returning a success-shaped failed result.
        raise
