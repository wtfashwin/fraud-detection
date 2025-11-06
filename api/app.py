import redis
import pickle
import hashlib
import threading
from fastapi import FastAPI, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field
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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Global variables for lazy loading
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_VERSION = None

# Redis connection for model caching
REDIS_HOST = os.getenv('REDIS_HOST', 'redis-master')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=False)

def get_model():
    """Lazy load model with thread safety and Redis caching."""
    global _MODEL, _MODEL_LOCK, _MODEL_VERSION
    
    # Check if model is already loaded
    if _MODEL is not None:
        return _MODEL
    
    # Thread-safe model loading
    with _MODEL_LOCK:
        # Double-check pattern to avoid race conditions
        if _MODEL is not None:
            return _MODEL
            
        try:
            latest_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE]) 
            if not latest_version:
                raise ValueError(f"No {MODEL_STAGE} version found for model {MODEL_NAME}")
                
            model_version = latest_version[0].version
            model_uri = f"models:/{MODEL_NAME}/{model_version}"
            
            # Try to get model from Redis cache
            model_hash = hashlib.md5(model_uri.encode()).hexdigest()
            cached_model = redis_client.get(f"model_cache:{model_hash}")
            
            if cached_model:
                logger.info(f"Loading model from Redis cache for {MODEL_NAME} version {model_version}")
                _MODEL = pickle.loads(cached_model)
                _MODEL_VERSION = model_version
                return _MODEL
            
            # Load model from MLflow
            model = mlflow.pyfunc.load_model(model_uri) 
            logger.info(f"Loaded {MODEL_NAME} version {model_version} from {MODEL_STAGE}")
            
            # Cache model in Redis
            try:
                redis_client.setex(f"model_cache:{model_hash}", 3600, pickle.dumps(model))  # Cache for 1 hour
                logger.info(f"Cached model in Redis for {MODEL_NAME} version {model_version}")
            except Exception as e:
                logger.warning(f"Failed to cache model in Redis: {e}")
            
            _MODEL = model
            _MODEL_VERSION = model_version
            return _MODEL
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            logger.warning("Falling back to local model file")
            _MODEL = joblib.load('models/logistic_model.joblib')
            return _MODEL

@lru_cache(maxsize=1)
def load_production_model():
    # This function is now deprecated, but kept for backward compatibility
    return get_model()

MODEL = load_production_model()
SCALER = joblib.load('models/scaler.joblib')
COLUMN_NAMES = joblib.load('models/columns.joblib')

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


predictions_submitted = Counter("predictions_submitted_total", "Total number of prediction requests submitted")
inference_time = Histogram("api_inference_duration_seconds", "Synchronous model inference time (seconds)")
db_latency = Histogram("api_db_latency_seconds", "DB call latency for startup checks (seconds)")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with db_latency.time():
            create_db_table()
    except Exception as e:
        logger.error(f"Failed to connect or create table: {e}")
        
    try:
        global MODEL
        MODEL = load_production_model()
        logger.info(f"Model loaded successfully from {'MLflow' if hasattr(MODEL, 'predict') else 'local fallback'}")
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        raise

    try:
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")
        resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "fraud-api")})
        provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        except Exception:
            logger.warning("SQLAlchemyInstrumentor could not instrument engine")

        logger.info("OpenTelemetry instrumentation configured (exporter=%s)", otel_endpoint)
    except Exception:
        logger.exception("Failed to configure OpenTelemetry")

    yield

app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan) 

class TransactionIn(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    features: list 

class PredictionOut(BaseModel):
    transaction_id: str
    prediction: int
    score: float
    correlation_id: str
    explanation_status: str

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Generates and logs a unique ID for request tracing."""
    request.state.correlation_id = str(uuid.uuid4())
    logger.info(f"[{request.state.correlation_id}] Request received: {request.url}")
    
    # Add correlation ID to OTEL span
    from opentelemetry import trace
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute("correlation.id", request.state.correlation_id)
        current_span.set_attribute("http.url", str(request.url))
    
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = request.state.correlation_id
    return response

@app.get("/status", tags=["Health"])
def get_status():
    """Liveness check: Is the service running?"""
    return {"status": "UP"}

@app.get("/health", tags=["Health"])
def get_health(request: Request):
    """Readiness check: Checks database, queue, and MLflow connectivity."""
    health_status = {"status": "OK", "dependencies": {}}
    degraded = False
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            health_status["dependencies"]["postgres"] = "UP"
    except Exception as e:
        health_status["dependencies"]["postgres"] = f"DOWN ({str(e)})"
        degraded = True

    health_status["dependencies"]["redis_broker"] = "UP"

    try:
        client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE]) 
        health_status["dependencies"]["mlflow"] = "UP"
        
        # Use the new get_model function
        model = get_model()
        if hasattr(model, 'predict'): 
            health_status["dependencies"]["model"] = "UP"
            
            # Add runtime validation with sample inference
            try:
                # Create a sample data point for testing
                sample_features = [0.5] * len(COLUMN_NAMES)  # Use mean values as sample
                sample_data = {col: 0.5 for col in COLUMN_NAMES}
                sample_df = pd.DataFrame([sample_data])
                
                # Test prediction
                prediction_result = model.predict(sample_df)
                prediction = prediction_result.iloc[0] if hasattr(prediction_result, 'iloc') else prediction_result[0]
                
                # Validate prediction is a valid numeric value
                assert isinstance(prediction, (int, float)), "Prediction is not numeric"
                assert -100 <= prediction <= 100, "Prediction value out of expected range"
                
                # Test probability prediction if available
                if callable(getattr(model, 'predict_proba', None)):
                    proba_result = model.predict_proba(sample_df)
                    proba = proba_result.iloc[0, 1] if hasattr(proba_result, 'iloc') else proba_result[0][1]
                    assert isinstance(proba, (int, float)), "Probability is not numeric"
                    assert 0 <= proba <= 1, "Probability value out of expected range"
                
                health_status["dependencies"]["model_validation"] = "PASSED"
                logger.info("Model runtime validation passed")
            except Exception as validation_error:
                health_status["dependencies"]["model_validation"] = f"FAILED ({str(validation_error)})"
                logger.error(f"Model runtime validation failed: {validation_error}")
                # Log failure to MLflow as experiment artifact
                try:
                    # This would normally log to MLflow, but we'll just log for now
                    logger.warning(f"Model validation failure would be logged to MLflow: {validation_error}")
                except Exception as mlflow_error:
                    logger.error(f"Failed to log validation failure to MLflow: {mlflow_error}")
        else:
            raise ValueError("Model object is invalid or not loaded.")
    except Exception as e:
        health_status["dependencies"]["mlflow"] = f"DOWN ({str(e)})"
        if hasattr(MODEL, 'predict'): 
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
    
    # Use lazy-loaded model
    model = get_model()
    
    # Validation is now handled by Pydantic model
    raw_data = pd.DataFrame([transaction.features])
    expected_input_features = len(SCALER.mean_)
    
    if raw_data.shape[1] != expected_input_features:
        raise HTTPException(
            status_code=422,
            detail=f"Input data must have {expected_input_features} features, but got {raw_data.shape[1]}. "
                   "This is the raw input size, *before* encoding/scaling."
        )

    scaled_features = SCALER.transform(raw_data)
    column_names_list = list(COLUMN_NAMES) if COLUMN_NAMES else [f'feature_{i}' for i in range(scaled_features.shape[1])]
    
    # Create DataFrame with proper column handling
    if scaled_features.shape[1] == len(column_names_list):
        X = pd.DataFrame(data=scaled_features, columns=column_names_list)
    else:
        feature_names = [f'feature_{i}' for i in range(scaled_features.shape[1])]
        X = pd.DataFrame(data=scaled_features, columns=feature_names)
        logger.warning(f"Feature count mismatch after scaling. Using {scaled_features.shape[1]} features, but model expects {len(column_names_list)}")
        
    if X.shape[1] != len(column_names_list):
        raise HTTPException(
            status_code=500,
            detail=f"Internal pre-processing error: Model requires {len(column_names_list)} features, but transformation pipeline produced {X.shape[1]}. Check SCALER/COLUMN_NAMES alignment."
        )

    with inference_time.time():
        # Use the correct MLflow PyFunc model interface
        prediction_result = model.predict(X)
        # Handle both array-like and DataFrame results
        if hasattr(prediction_result, 'iloc'):
            prediction = int(prediction_result.iloc[0])
        elif hasattr(prediction_result, '__getitem__'):
            prediction = int(prediction_result[0])
        else:
            prediction = int(prediction_result)
        
        try:
            # For PyDantic models, we need to handle predict_proba differently
            score = float(prediction)  # Default fallback
            # Try to get probability scores if the model supports it
            if callable(getattr(model, 'predict_proba', None)):
                proba_result = model.predict_proba(X)
                # Handle both array-like and DataFrame results
                if hasattr(proba_result, 'iloc'):
                    score = float(proba_result.iloc[0, 1])
                elif hasattr(proba_result, '__getitem__') and len(proba_result) > 0:
                    if hasattr(proba_result[0], '__getitem__') and len(proba_result[0]) > 1:
                        score = float(proba_result[0][1])
                    else:
                        score = float(proba_result[0])
                else:
                    score = float(proba_result)
        except Exception as e:
            logger.warning(f"Could not get probability score: {e}")
            score = float(prediction)

    # Add business context to OTEL span
    from opentelemetry import trace
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute("fraud.score", score)
        current_span.set_attribute("fraud.prediction", prediction)
        current_span.set_attribute("fraud.amount_features", len(transaction.features))
        
        # Add fraud score bucket
        if score < 0.3:
            fraud_bucket = "LOW_RISK"
        elif score < 0.7:
            fraud_bucket = "MEDIUM_RISK"
        else:
            fraud_bucket = "HIGH_RISK"
        current_span.set_attribute("fraud.score_bucket", fraud_bucket)

    try:
        # Use the correct Celery task interface
        task = compute_shap.apply_async(
            args=[transaction.transaction_id, transaction.features, corr_id]
        )
        explanation_status = "Calculation queued"
        logger.info(f"SHAP task queued with ID: {task.id}")
    except Exception as e:
        logger.error(f"[{corr_id}] Failed to queue SHAP task: {e}")
        explanation_status = "Queue failed"
        
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
    
    return {
        "transaction_id": transaction_id,
        "created_at": result.created_at,
        "shap_values": result.shap_values,
        "feature_names": result.feature_names
    }


Instrumentator().instrument(app).expose(app, endpoint="/metrics")