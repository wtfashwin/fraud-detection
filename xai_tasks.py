import os
import logging
import json
import contextvars
import uuid as _uuid
import numpy as np
from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from db.db import SessionLocal
from db.models import TransactionResult, StatusEnum

from api.utils import load_model_and_features 

logging.basicConfig(level=logging.INFO)
logger = get_task_logger(__name__)

# Structured logging for worker
from pythonjsonlogger import jsonlogger

# OpenTelemetry + Prometheus setup for Celery worker
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from prometheus_client import start_http_server, Histogram, Counter

try:
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")
    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "xai-worker")})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    try:
        CeleryInstrumentor().instrument()
    except Exception:
        logger.warning("Failed to instrument Celery for OpenTelemetry")
except Exception:
    logger.exception("Failed to configure OpenTelemetry for worker")

# Prometheus metrics for the worker
task_duration = Histogram('xai_task_duration_seconds', 'XAI task duration seconds')
task_success = Counter('xai_task_success_total', 'Number of successful XAI tasks')
task_failure = Counter('xai_task_failures_total', 'Number of failed XAI tasks')

try:
    start_http_server(8001)
    logger.info("Prometheus metrics HTTP server started on :8001")
except Exception:
    logger.exception("Failed to start Prometheus HTTP server for worker")

# --- Celery App Initialization ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "sentinel://redis-master:26379/0")
# Add Redis result backend for DLQ implementation
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-master:6379/2")

celery_app = Celery("xai_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Configure Celery with retry and DLQ settings
celery_app.conf.update(
    task_routes={
        'xai_tasks.compute_shap': {
            'queue': 'xai_tasks',
            'routing_key': 'xai_tasks.compute_shap'
        }
    },
    task_default_queue='xai_tasks',
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
)

# Define PII fields that should be masked
PII_FIELDS = {
    'user_id', 'customer_id', 'account_number', 'cc_number', 'credit_card',
    'ssn', 'social_security', 'phone', 'email', 'address'
}

# Add environment variable for SHAP background size
SHAP_BG_SIZE = int(os.getenv('SHAP_BG_SIZE', 100))

@celery_app.task(
    bind=True, 
    autoretry_for=(Exception,), 
    retry_backoff=True, 
    max_retries=3, 
    retry_jitter=True,
    acks_late=True
)
def compute_shap(self, transaction_id: str, input_data: dict, correlation_id: str | None = None):
    """
    Celery task that computes prediction and SHAP-like attributions and writes results to Postgres.
    
    This function implements Task A3 (Asynchronous Decoupling) and A4 (PostgreSQL Integration).
    The 'bind=True' allows access to 'self' for retry logic.
    """
    
    logger.info("Starting SHAP computation for transaction: %s (Attempt %s)", 
                transaction_id, self.request.retries + 1)
    
    session: Session = SessionLocal()

    try:
        # 1. Load Model and Prepare Data
        # Handles model loading errors gracefully within the worker context
        model, feature_names = load_model_and_features(
            os.getenv("MODEL_PATH"), os.getenv("FEATURE_NAMES_PATH")
        )

        # Map input data to the model's expected feature order
        # Fallback to dictionary order if feature names are unavailable (less robust)
        if feature_names:
            x = [float(input_data.get(f, 0.0)) for f in feature_names]
        else:
            x = [float(v) for _, v in sorted(input_data.items())]
            feature_names = [k for k, _ in sorted(input_data.items())]
        
        x_arr = np.array([x])

        # 2. Prediction Calculation
        try:
            # Use predict_proba for probability score (common for fraud detection)
            pred_proba = float(model.predict_proba(x_arr)[0, 1])
        except AttributeError:
            # Fallback for models without predict_proba (e.g., some simple linear models)
            pred_proba = float(model.predict(x_arr)[0])

        # 3. Attribution Calculation (SHAP-like for Linear Models)
        shap_values = {}
        try:
            # For linear models (LogisticRegression), attribution is coef * feature_value
            coefs = np.asarray(getattr(model, "coef_")).ravel()
            attributions = coefs * x_arr[0]
            
            for fname, attr in zip(feature_names, attributions.tolist()):
                shap_values[fname] = float(attr)

        except Exception as e:
            logger.warning("Model lacks coefficients for simple XAI: %s. Using simple feature values.", e)
            for k, v in input_data.items():
                 shap_values[k] = float(v) 
        
        # 4. PII Redaction
        # Mask sensitive fields before storing
        masked_shap_values = {}
        for k, v in shap_values.items():
            if k.lower() in PII_FIELDS:
                masked_shap_values[k] = '***'  # Mask PII fields
            else:
                masked_shap_values[k] = v
        
        # Convert to JSON string for storage
        shap_json = json.dumps(masked_shap_values)
        
        # 5. Store results with SERIALIZABLE isolation level
        # Begin a transaction with SERIALIZABLE isolation level
        session.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
        
        rec = session.get(TransactionResult, _uuid.UUID(transaction_id))

        if not rec:
            rec = TransactionResult(
                id=_uuid.UUID(transaction_id),
                input_data=input_data,
                status=StatusEnum.COMPLETED,
            )
            session.add(rec)
            logger.warning("Record not found, creating new COMPLETED record for %s", transaction_id)
        
        rec.shap_values = masked_shap_values
        rec.prediction_score = pred_proba
        rec.status = StatusEnum.COMPLETED

        session.commit()
        logger.info("SHAP task completed", extra={"transaction_id": transaction_id, "correlation_id": correlation_id})
        return {"transaction_id": transaction_id, "status": "COMPLETED"}

    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Database (SQLAlchemy) error for %s: %s", transaction_id, exc)
        
        raise self.retry(exc=exc, countdown=5)
    
    except Exception as exc:
        session.rollback()
        logger.exception("Unrecoverable processing error for %s: %s", transaction_id, exc)
        
        # Update status to FAILED and log max retries
        try:
            # Update the status to FAILED before giving up
            rec = session.get(TransactionResult, _uuid.UUID(transaction_id))
            if rec:
                rec.status = StatusEnum.FAILED
                session.commit()
        except SQLAlchemyError:
            # Ignore errors while attempting to mark as FAILED
            session.rollback()
        
        # Raise error to retry, which should eventually hit max_retries
        try:
            raise self.retry(exc=exc, countdown=10)
        except self.MaxRetriesExceededError:
            logger.error("Max retries exceeded for %s. Final status: FAILED.", transaction_id)
            # Send to DLQ by storing in a separate table or Redis list
            try:
                # Store failed task in Redis DLQ
                dlq_key = "dlq:failed_shap_tasks"
                failed_task_data = {
                    "transaction_id": transaction_id,
                    "input_data": input_data,
                    "correlation_id": correlation_id,
                    "error": str(exc),
                    "timestamp": _uuid.uuid1().time
                }
                celery_app.backend.lpush(dlq_key, json.dumps(failed_task_data))
                logger.info("Task sent to DLQ: %s", transaction_id)
            except Exception as dlq_error:
                logger.error("Failed to send task to DLQ: %s", dlq_error)
            
            return {"transaction_id": transaction_id, "status": "FAILED"}

    finally:
        # Crucial for Celery workers to explicitly close the session
        session.close()