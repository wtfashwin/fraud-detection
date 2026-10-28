"""Synchronous fraud scoring with persisted, asynchronous model explanations."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictFloat, StrictInt, field_validator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, REGISTRY
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from api.inference import load_bundle, InvalidFeatures
from api.drift import DriftMonitor, DriftCollector
from api.queue import celery_app, check_broker
from api.middleware import PredictionBodyLimitMiddleware
from db.db import engine
from db.models import Explanation

logger = logging.getLogger(__name__)
BUNDLE = None
DRIFT_MONITOR = None
DRIFT_MONITOR_KEY = None
predictions_submitted = Counter("predictions_submitted_total", "Prediction requests submitted")
inference_time = Histogram("api_inference_duration_seconds", "Synchronous inference duration")


@asynccontextmanager
async def lifespan(app):
    global BUNDLE
    if BUNDLE is None:
        try:
            BUNDLE = load_bundle()
        except Exception as exc:
            logger.error("Model startup failed (%s); readiness will remain down", type(exc).__name__)
    yield


app = FastAPI(title="Fraud Detection API", version="1.1.0", lifespan=lifespan)
# Correlation middleware is registered below, so it also wraps body-limit errors.
app.add_middleware(PredictionBodyLimitMiddleware)


class TransactionIn(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1,
                                max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    features: Annotated[list[StrictInt | StrictFloat], Field(min_length=1, max_length=1024)]

    @field_validator("features")
    @classmethod
    def finite_features(cls, values):
        import math
        try:
            finite = all(math.isfinite(value) for value in values)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Features must be finite numbers within floating-point range")
        return values


class PredictionOut(BaseModel):
    transaction_id: str
    prediction: int
    score: float
    correlation_id: str
    explanation_status: str


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    request.state.correlation_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = request.state.correlation_id
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Nonfinite submitted values are not JSON serializable; omit rejected input values.
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # Starlette's outer server-error response bypasses the middleware return path.
    correlation = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())
    return PlainTextResponse("Internal Server Error", status_code=500,
                             headers={"X-Correlation-ID": correlation})


def get_drift_monitor():
    global DRIFT_MONITOR, DRIFT_MONITOR_KEY
    config = (os.getenv("DRIFT_MIN_SAMPLES", "20"), os.getenv("DRIFT_MAX_SAMPLES", "512"),
              os.getenv("DRIFT_KS_THRESHOLD", "0.05"))
    key = (id(BUNDLE), config)
    if DRIFT_MONITOR is None or DRIFT_MONITOR_KEY != key:
        try:
            DRIFT_MONITOR = DriftMonitor(getattr(BUNDLE, "drift_baseline", None),
                min_samples=int(config[0]), max_samples=int(config[1]), threshold=float(config[2]),
                unavailable_reason=getattr(BUNDLE, "baseline_error", None))
        except (ValueError, TypeError):
            DRIFT_MONITOR = DriftMonitor(None, unavailable_reason="Drift monitoring configuration is invalid")
        DRIFT_MONITOR_KEY = key
    return DRIFT_MONITOR


@app.get("/status", tags=["Health"])
def get_status():
    return {"status": "UP"}


@app.get("/health", tags=["Health"])
def get_health():
    # Distribution drift does not mean the process is dead or prove an AUC decrease.
    return {"status": "UP", "drift": get_drift_monitor().report()}


@app.get("/ready", tags=["Health"])
def get_readiness():
    dependencies = {"model": "UP" if BUNDLE is not None else "DOWN"}
    try:
        with engine.connect() as connection:
            # A connected database without migrations is not ready to persist requests.
            connection.execute(select(Explanation.__table__).limit(1))
        dependencies["postgres"] = "UP"
    except Exception:
        dependencies["postgres"] = "DOWN"
    dependencies["redis_broker"] = "UP" if check_broker() else "DOWN"
    ready = all(value == "UP" for value in dependencies.values())
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "OK" if ready else "DEGRADED", "dependencies": dependencies})


def require_bundle():
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return BUNDLE


@app.get("/model-info", tags=["Model"])
def model_info():
    bundle = require_bundle()
    return {"feature_names": bundle.columns, "feature_count": len(bundle.columns), "model_source": bundle.source,
            "explanation_feature_names": bundle.explanation_columns}


@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(transaction: TransactionIn, request: Request):
    bundle = require_bundle()
    if len(transaction.features) != len(bundle.columns):
        raise HTTPException(status_code=422, detail=f"Input data must have {len(bundle.columns)} features")
    predictions_submitted.inc()
    try:
        with inference_time.time():
            prediction, score = bundle.score(transaction.features)
    except InvalidFeatures as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Inference failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Model could not score this request") from exc
    corr_id = request.state.correlation_id
    try:
        with engine.begin() as connection:
            connection.execute(insert(Explanation).values(transaction_id=transaction.transaction_id,
                               correlation_id=corr_id, status="PENDING"))
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Transaction ID already exists; use a new ID") from exc
    except Exception as exc:
        logger.error("Prediction persistence failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Explanation storage is unavailable") from exc
    try:
        celery_app.send_task("xai_tasks.compute_shap",
                             args=[transaction.transaction_id, list(transaction.features), corr_id], retry=False)
    except Exception as exc:
        logger.error("Explanation dispatch failed (%s)", type(exc).__name__)
        dispatch_status = "DISPATCH_UNCONFIRMED"
        try:
            with engine.begin() as connection:
                connection.execute(update(Explanation).where(
                    Explanation.transaction_id == transaction.transaction_id, Explanation.status == "PENDING")
                    .values(status="FAILED"))
                dispatch_status = connection.execute(select(Explanation.status).where(
                    Explanation.transaction_id == transaction.transaction_id)).scalar_one_or_none() or dispatch_status
        except Exception:
            logger.error("Could not confirm saved dispatch status")
        if dispatch_status == "COMPLETED":
            message = "Explanation completed despite a lost dispatch acknowledgement; read the saved explanation"
        elif dispatch_status == "FAILED":
            message = "Explanation queue is unavailable"
        else:
            message = "Explanation dispatch is unconfirmed; check saved status before retrying"
        raise HTTPException(status_code=503, detail={"transaction_id": transaction.transaction_id,
                            "status": dispatch_status, "message": message}) from exc
    # Record only successfully accepted scoring requests; Redis monitoring failure is nonfatal.
    try:
        get_drift_monitor().observe(transaction.features)
    except Exception as exc:
        logger.warning("Drift observation failed (%s)", type(exc).__name__)
    return PredictionOut(transaction_id=transaction.transaction_id, prediction=prediction, score=score,
                         correlation_id=corr_id, explanation_status="Calculation queued")


@app.get("/explain/{transaction_id}", tags=["Explanation"])
def get_explanation(transaction_id: str):
    try:
        with engine.connect() as connection:
            row = connection.execute(select(Explanation.__table__).where(
                Explanation.transaction_id == transaction_id)).mappings().first()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Explanation storage is unavailable") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="SHAP explanation not found")
    if row["status"] == "PENDING":
        return JSONResponse(status_code=202, content={"transaction_id": transaction_id, "status": "PENDING"})
    if row["status"] == "FAILED":
        raise HTTPException(status_code=503, detail={"transaction_id": transaction_id, "status": "FAILED",
                                                  "message": "Explanation calculation failed"})
    if row["status"] != "COMPLETED":
        logger.error("Saved explanation has an unrecognized status")
        raise HTTPException(status_code=503, detail={"transaction_id": transaction_id, "status": "UNAVAILABLE",
                                                  "message": "Saved explanation status is unavailable"})
    return {key: row[key] for key in ("transaction_id", "status", "created_at", "shap_values", "feature_names",
                                      "base_value", "explanation_method")}


REGISTRY.register(DriftCollector(lambda: get_drift_monitor()))
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
demo_directory = Path(__file__).resolve().parents[1] / "web"
if demo_directory.is_dir():
    app.mount("/demo", StaticFiles(directory=demo_directory, html=True), name="demo")
