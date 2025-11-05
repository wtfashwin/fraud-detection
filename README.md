# AI-Powered Fraud Detection System

This project implements a production-grade fraud detection system with asynchronous explainability, MLOps instrumentation, and observability.

## Features
- FastAPI-based prediction API with correlation IDs
- Async XAI calculations via Celery workers
- MLflow experiment tracking and model registry
- Prometheus metrics and Grafana dashboards
- OpenTelemetry distributed tracing
- Docker Compose stack for local development
- Kubernetes deployment manifests and KEDA autoscaling
- Synthetic data generation for testing

## Project Structure
```
├── api/              # FastAPI application
├── models/           # Trained models and scalers
├── scripts/          # Utility scripts
├── monitoring/       # Prometheus, Grafana, OTEL configs
├── alembic/          # Database migrations
├── k8s/              # Kubernetes manifests
└── docker-compose.yml
```

## Quick Start

### Local Development
1. Start the full stack:
```bash
docker compose up -d
```

2. Generate synthetic test data:
```bash
python scripts/generate_synthetic_data.py
```

3. Train and register model:
```bash
# Set environment variables for MLflow (optional)
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT=fraud-detection
export MLFLOW_MODEL_NAME=fraud-detection-model
export MLFLOW_AUC_THRESHOLD=0.95

# Train using real or synthetic data
python train_model.py
```

4. Access services:
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000
- Jaeger UI: http://localhost:16686

### Model Training & MLflow
- The `train_model.py` script:
  - Preprocesses data with SMOTE
  - Trains a LogisticRegression model
  - Computes test AUC score
  - Logs metrics and artifacts to MLflow
  - Registers model if AUC threshold is met

Environment variables:
- `DATA_CSV`: Path to training data CSV
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MLFLOW_EXPERIMENT`: Experiment name
- `MLFLOW_MODEL_NAME`: Name for model registry
- `MLFLOW_AUC_THRESHOLD`: Min AUC to register model

### API & Worker
The system consists of:
1. FastAPI application (/predict endpoint)
2. Celery worker for async XAI
3. PostgreSQL for result persistence
4. Redis for task queue
5. Observability stack (Prometheus, Grafana, Jaeger)

### CI/CD
The GitHub Actions workflow:
1. Runs tests
2. Generates synthetic data
3. Trains model on CI dataset
4. Builds and scans Docker image
5. Uploads model artifacts

### Production Deployment
Kubernetes manifests provided:
- API and worker deployments
- KEDA ScaledObject for autoscaling
- HorizontalPodAutoscaler configs
- Example Prometheus alert rules

## Development

### Running Tests
```bash
pytest
```

### Database Migrations
```bash
alembic upgrade head
```

### Synthetic Data Generation
For testing without production data:
```bash
python scripts/generate_synthetic_data.py
```

## Monitoring
- Prometheus metrics at /metrics
- Grafana dashboards for model/system metrics
- Jaeger for distributed tracing
- Structured JSON logging with correlation IDs

## Contributing
1. Branch from main
2. Make changes
3. Run tests and lint
4. Submit PR

## License
MIT