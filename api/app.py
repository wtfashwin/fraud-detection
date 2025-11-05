from fastapi import FastAPI, HTTPException, Request, Depends, status
from pydantic import BaseModel
import joblib
import uuid
import pandas as pd
import json
import os
import logging
from sqlalchemy import create_engine, text
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from .worker import compute_shap, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL = joblib.load('models/logistic_model.joblib') 
SCALER = joblib.load('models/scaler.joblib')
COLUMN_NAMES = joblib.load('models/columns.joblib')
# --- 2. DATABASE & SCHEMA SETUP (Executed on Startup) ---

def create_db_table():
    """Ensures the table for SHAP results exists."""
    SQL = """
    CREATE TABLE IF NOT EXISTS shap_explanations (
        transaction_id VARCHAR(255) PRIMARY KEY,
        correlation_id VARCHAR(255),
        shap_values JSONB NOT NULL,
        feature_names JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.connect() as connection:
        connection.execute(text(SQL))
        connection.commit()
    logger.info("Database table 'shap_explanations' ensured.")

@app.on_event("startup")
async def startup_event():
    # Run the table creation on startup (best-effort)
    try:
        create_db_table()
    except Exception as e:
        logger.error(f"Failed to connect or create table: {e}")
        # In production, this would be a fatal error, but we log and proceed for local demo

    # OpenTelemetry tracing setup (P2.2)
    try:
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")
        resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "fraud-api")})
        provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(provider)

        # Instrument FastAPI and SQLAlchemy engine
        FastAPIInstrumentor.instrument_app(app)
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        except Exception:
            logger.warning("SQLAlchemyInstrumentor could not instrument engine")

        logger.info("OpenTelemetry instrumentation configured (exporter=%s)", otel_endpoint)
    except Exception:
        logger.exception("Failed to configure OpenTelemetry")

# --- 3. Pydantic Schemas ---

class TransactionIn(BaseModel):
    transaction_id: str = Depends(lambda: str(uuid.uuid4()))
    features: list

class PredictionOut(BaseModel):
    transaction_id: str
    prediction: int
    score: float
    correlation_id: str
    explanation_status: str

# --- 4. MIDDLEWARE (Task P2.3: Correlation ID) ---

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Generates and logs a unique ID for request tracing."""
    request.state.correlation_id = str(uuid.uuid4())
    logger.info(f"[{request.state.correlation_id}] Request received: {request.url}")
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = request.state.correlation_id
    return response

# --- 5. ENDPOINTS (P2.4 Health Check & Core Logic) ---

@app.get("/status", tags=["Health"])
def get_status():
    """Liveness check: Is the service running?"""
    return {"status": "UP"}

@app.get("/health", tags=["Health"])
def get_health(request: Request):
    """Readiness check: Checks database and queue connectivity."""
    health_status = {"status": "OK", "dependencies": {}}
    
    # Check Postgres connection
    try:
        with engine.connect():
            health_status["dependencies"]["postgres"] = "UP"
    except Exception:
        health_status["dependencies"]["postgres"] = "DOWN"
        health_status["status"] = "DEGRADED"

    # Check Redis/Celery Broker connection (Celery connection is usually implicit)
    # A simple way: Try sending a dummy task or checking Celery's connection status
    # For simplicity here, we assume if Postgres is up, the stack is mostly ready.
    health_status["dependencies"]["redis_broker"] = "UP" # Simplification for demo

    if health_status["status"] == "DEGRADED":
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status)
    return health_status

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(transaction: TransactionIn, request: Request):
    """Synchronous scoring and asynchronous SHAP calculation."""
    corr_id = request.state.correlation_id
    
    # 5a. Synchronous Prediction (FAST)
    X = pd.DataFrame([transaction.features])
    prediction = int(MODEL.predict(X)[0])
    score = float(MODEL.predict_proba(X)[:, 1][0])

    # 5b. Asynchronous SHAP Task (P2.2 Decoupling)
    # THIS LINE WAS THE SOURCE OF THE SYNTAX ERROR! It must be INSIDE a block.
    try:
        compute_shap.apply_async(
            args=[transaction.transaction_id, transaction.features, corr_id],
            countdown=0
        )
        explanation_status = "Calculation queued"
    except Exception as e:
        logger.error(f"[{corr_id}] Failed to queue SHAP task: {e}")
        explanation_status = "Queue failed"
        # Since this failed, we should still return the fast prediction
    
    logger.info(f"[{corr_id}] Prediction done: {prediction}, SHAP status: {explanation_status}")

    return PredictionOut(
        transaction_id=transaction.transaction_id,
        prediction=prediction,
        score=score,
        correlation_id=corr_id,
        explanation_status=explanation_status
    )

@app.get("/explain/{transaction_id}", tags=["Explanation"])
def get_shap_explanation(transaction_id: str):
    """Retrieves stored SHAP results from Postgres."""
    SQL = text("SELECT shap_values, feature_names, created_at FROM shap_explanations WHERE transaction_id = :tx_id")
    
    with engine.connect() as connection:
        result = connection.execute(SQL, {"tx_id": transaction_id}).fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="SHAP explanation not found. Calculation may still be pending.")
    
    # The JSON data is retrieved as a string/dict from the JSONB column
    return {
        "transaction_id": transaction_id,
        "created_at": result.created_at,
        "shap_values": result.shap_values,
        "feature_names": result.feature_names
    }

# --- 6. PROMETHEUS INSTRUMENTATION (Task P2.1) ---

# Instrument the app globally for standard metrics (latency, request count)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")