# Fraud Detection

Fraud Detection is a Python prototype for credit-card transaction classification, with an API, background attribution jobs, and deployment configuration.

## What it contains

- XGBoost training with scaling, stratified splits, SMOTE, and MLflow logging/registration.
- A FastAPI prediction service with `/predict`, `/explain/{transaction_id}`, health routes, and Prometheus metrics.
- Celery attribution workers and PostgreSQL persistence code.
- Docker Compose, Helm/Kubernetes files, and monitoring configuration for Prometheus, Grafana, Jaeger, and OpenTelemetry.

## Local training

The Dockerfile uses Python 3.11. For local exploration:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`train_model.py` reads a CSV with a `Class` target and numeric feature columns. Its default path is `data/creditcard.csv`, while the included data is named `data/credit_card.csv`. Select the included path explicitly:

```bash
DATA_CSV=data/credit_card.csv MLFLOW_TRACKING_URI=http://localhost:5000 python train_model.py
```

Alternatively, create the synthetic development dataset first:

```bash
python scripts/generate_synthetic_data.py
```

The generator writes `data/creditcard.csv`; its second generated dataset overwrites its first. Training writes `models/xgb_model.joblib` and `models/scaler.joblib`. MLflow errors are caught after local artifacts are saved. Registration depends on the configured AUC threshold; a synthetic-data run does not establish real fraud-detection performance.

## API and service setup

The full stack is configured in `docker-compose.yml`:

```bash
docker compose up --build -d
```

This setup needs repair before it is a working quick start:

- The API broker URL points to `redis-master:26379`, while the Sentinel service is named `redis-sentinel`.
- The Sentinel command expects Bash in a Redis Alpine image, and the API healthcheck expects curl although the Dockerfile does not install it.
- `run_migrations.sh` skips migrations when Alembic is absent; Alembic is not declared in `requirements.txt`.
- API fallback and workers load logistic-model artifacts while training writes an XGBoost model and replaces the shared scaler.
- Worker attribution inputs and stored results need alignment with the API's feature mapping and explanation lookup.

Review `DATABASE_URL`, `CELERY_BROKER_URL`, `MLFLOW_TRACKING_URI`, model settings, and deployment credentials before launching shared services. A tracked `.env` is present; review it for private values before reuse or sharing.

Once the service integration is repaired, API docs are configured at `http://localhost:8000/docs`. Send an ordered numeric `features` array to `/predict`; the order must match `models/feature_names.json`. Predictions and queued explanations are separate operations. The current worker calculates coefficient-based attributions with a feature-value fallback, not validated SHAP explanations.

Configured UIs use ports 5000 for MLflow, 3000 for Grafana, 9090 for Prometheus, and 16686 for Jaeger.

## Project structure

- `train_model.py`, `preprocess.py`, `evaluate_model.py` — different training/evaluation paths.
- `api/`, `xai_tasks.py`, `db/`, `alembic/` — prediction, workers, storage, and migrations.
- `models/` — checked-in logistic artifacts; training also produces XGBoost artifacts.
- `docker-compose.yml`, `charts/`, `k8s/`, `monitoring/` — service/deployment configuration.
- `tests/test_api.py`, `docs/` — API test source and recovery/runbook notes.

## Checks

With development dependencies and required services configured:

```bash
python -m pytest tests/
alembic upgrade head
```

Pytest and Alembic are not declared in `requirements.txt`. Tests import the API and its service/model dependencies, so they are not isolated checks. Kubernetes manifests and monitoring files are configuration, not evidence of a successful deployment.

## License

The project states MIT, but no license file is included.
