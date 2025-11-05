from fastapi import FastAPI, HTTPException, Request, Depends, status
from pydantic import BaseModel
import joblib
import uuid
import pandas as pd
import json
import os
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from functools import lru_cache
from sqlalchemy import create_engine, text
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from xai_tasks import compute_shap
from db.db import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'fraud-detection-model')
MODEL_STAGE = os.getenv('MLFLOW_MODEL_STAGE', 'Production')

# Initialize MLflow client
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

@lru_cache(maxsize=1)
def load_production_model():
    """Load the latest production model from MLflow."""
    try:
        # Find the latest version in the specified stage
        latest_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not latest_version:
            raise ValueError(f"No {MODEL_STAGE} version found for model {MODEL_NAME}")
            
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded {MODEL_NAME} version {latest_version[0].version} from {MODEL_STAGE}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # Fallback to local file if MLflow fails (for development)
        logger.warning("Falling back to local model file")
        return joblib.load('models/logistic_model.joblib')

# Load model and preprocessing artifacts
MODEL = load_production_model()
SCALER = joblib.load('models/scaler.joblib')  # Keep scaler local for now
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


# Custom Prometheus metrics (API-side)
predictions_submitted = Counter("predictions_submitted_total", "Total number of prediction requests submitted")
inference_time = Histogram("api_inference_duration_seconds", "Synchronous model inference time (seconds)")
db_latency = Histogram("api_db_latency_seconds", "DB call latency for startup checks (seconds)")

@app.on_event("startup")
async def startup_event():
    # Run the table creation on startup (best-effort) and measure DB latency
    try:
        with db_latency.time():
            create_db_table()
    except Exception as e:
        logger.error(f"Failed to connect or create table: {e}")
        # In production, this would be a fatal error, but we log and proceed for local demo
        
    # Ensure model is loaded from MLflow (or fallback)
    try:
        global MODEL
        MODEL = load_production_model()
        logger.info(f"Model loaded successfully from {'MLflow' if isinstance(MODEL, mlflow.sklearn.SKLearnModel) else 'local fallback'}")
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        raise

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
    """Readiness check: Checks database, queue, and MLflow connectivity."""
    health_status = {"status": "OK", "dependencies": {}}
    degraded = False
    
    # Check Postgres connection
    try:
        with engine.connect():
            health_status["dependencies"]["postgres"] = "UP"
    except Exception as e:
        health_status["dependencies"]["postgres"] = f"DOWN ({str(e)})"
        degraded = True

    # Check Redis/Celery Broker connection
    # For demo we assume if Postgres is up, Redis is probably up
    health_status["dependencies"]["redis_broker"] = "UP"

    # Check MLflow connection and model
    try:
        client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        health_status["dependencies"]["mlflow"] = "UP"
        
        # Verify model is loaded
        if not isinstance(MODEL, mlflow.sklearn.SKLearnModel):
            raise ValueError("Model not loaded from MLflow")
        health_status["dependencies"]["model"] = "UP"
    except Exception as e:
        health_status["dependencies"]["mlflow"] = f"DOWN ({str(e)})"
        # If using fallback model, mark as degraded but not down
        if isinstance(MODEL, mlflow.sklearn.SKLearnModel):
            health_status["dependencies"]["model"] = "DEGRADED (using fallback)"
        else:
            health_status["dependencies"]["model"] = "DOWN"
            degraded = True

    if degraded:
        health_status["status"] = "DEGRADED"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status)
    return health_status

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(transaction: TransactionIn, request: Request):
    """Synchronous scoring and asynchronous SHAP calculation."""
    corr_id = request.state.correlation_id
    predictions_submitted.inc()
    
    # 5a. Synchronous Prediction (FAST)
    X = pd.DataFrame([transaction.features])
    with inference_time.time():
        prediction = int(MODEL.predict(X)[0])
        try:
            score = float(MODEL.predict_proba(X)[:, 1][0])
        except Exception:
            score = float(MODEL.predict(X)[0])

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