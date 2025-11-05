# api/worker.py

from celery import Celery
import os
import joblib
import pandas as pd
import shap
import uuid
import json
from sqlalchemy import create_engine, text
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from prometheus_client import start_http_server, Histogram, Counter

# 1. Setup Celery App
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app = Celery('xai_tasks', broker=CELERY_BROKER_URL)

# Initialize OpenTelemetry for worker
try:
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")
    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "xai-worker")})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    # Instrument Celery
    try:
        CeleryInstrumentor().instrument()
    except Exception:
        pass
    # Instrument SQLAlchemy engine later (after engine exists)
except Exception:
    pass

# 2. Setup Database Engine
# NOTE: In production, use connection pooling (e.g., SQLAlchemy Pool)
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# 3. Model Loading (Global for performance)
MODEL = joblib.load('models/logistic_model.joblib') 
SCALER = joblib.load('models/scaler.joblib')
COLUMN_NAMES = joblib.load('models/columns.joblib')
FEATURE_COUNT = len(COLUMN_NAMES)

X_sample = pd.DataFrame([[0] * FEATURE_COUNT], columns=COLUMN_NAMES)
EXPL = shap.LinearExplainer(MODEL, X_sample)
# Prometheus metrics for worker
task_duration = Histogram('xai_task_duration_seconds', 'XAI task duration seconds')
task_success = Counter('xai_task_success_total', 'Number of successful XAI tasks')
task_failure = Counter('xai_task_failures_total', 'Number of failed XAI tasks')

# Start Prometheus HTTP server for worker metrics (scrapable by Prometheus)
try:
    start_http_server(8001)
except Exception:
    pass
# 4. Define the Asynchronous Task
@app.task(name='xai_tasks.compute_shap')
def compute_shap(transaction_id: str, features: list, correlation_id: str):
    print(f"[{correlation_id}] Worker started for TX ID: {transaction_id}")
    
    # 4a. Format Features
    X = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(len(features))])
    
    # 4b. Calculate SHAP Values (the slow part)
    try:
        with task_duration.time():
            shap_values = EXPL.shap_values(X)
        task_success.inc()
    except Exception as e:
        task_failure.inc()
        raise
    
    # 4c. Format for Storage
    shap_dict = {
        'transaction_id': transaction_id,
        'correlation_id': correlation_id,
        'shap_values': json.dumps(shap_values[0].tolist()), # Convert numpy array to JSON string
        'feature_names': json.dumps(list(X.columns)),
    }
    
    # 4d. Store Results in Postgres
    insert_sql = text("""
        INSERT INTO shap_explanations (transaction_id, correlation_id, shap_values, feature_names)
        VALUES (:transaction_id, :correlation_id, :shap_values, :feature_names)
        ON CONFLICT (transaction_id) DO UPDATE
        SET shap_values = EXCLUDED.shap_values;
    """)
    
    with engine.connect() as connection:
        connection.execute(insert_sql, shap_dict)
        connection.commit()
    
    print(f"[{correlation_id}] SHAP results saved successfully for {transaction_id}")
    return True