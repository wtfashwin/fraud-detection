# Fraud Detection API

A small FastAPI demo for scoring 30 ordered credit-card features, with asynchronous explanations persisted by a Celery worker. PostgreSQL stores explanation state; Redis carries tasks. The browser demo uses the real API.

The bundled model is a demonstration artifact. Its training data and held-out fraud performance have not been established. The checked-in `archive/data/credit_approval.csv` is a different credit-approval dataset and is not suitable for evaluating this fraud model. Synthetic data checks software behavior, not fraud accuracy.

## Repository layout

```text
api/         HTTP endpoints, inference, queue, and worker
db/         Shared database models and connections
training/    Model training, preprocessing, evaluation, single-row prediction
web/         Browser demo and JavaScript tests
scripts/     Data generation, migrations, validation, and HTTP smoke checks
tests/       Python regression tests
models/      Canonical bundled demo artifacts; trained outputs are separate
charts/      Optional Helm deployment example and render checks
monitoring/  Optional Prometheus configuration and monitoring references
docs/        Current operating notes
archive/     Earlier experiments, raw manifests, and incompatible sample data
```

`docker-compose.yml`, `Dockerfile`, dependency files, and project instructions stay at the root. `xai_tasks.py` is a small compatibility entry point; new workers use `api.worker.app`.

## Run locally

```sh
docker compose --env-file .env.example up --build -d --wait
python3 scripts/smoke_test.py
```

Open [the demo](http://127.0.0.1:8000/demo/) or [API docs](http://127.0.0.1:8000/docs). The core stack has four services: API, worker, PostgreSQL, and Redis. Ports bind to loopback. `.env.example` contains local demonstration credentials; use separate secrets and access controls before external deployment.

```sh
docker compose --env-file .env.example ps
docker compose --env-file .env.example logs --tail 100 api xai-worker
docker compose --env-file .env.example down
```

`down` keeps local data volumes. Add `-v` only when deliberately discarding local demonstration records.

## Pull a published image

The existing package is at [GHCR](https://github.com/wtfashwin/fraud-detection/pkgs/container/fraud-api). The historical image below predates the reliability fixes in this branch:

```sh
docker pull ghcr.io/wtfashwin/fraud-api:8192bd3f4d2d5ac053ccf79fb90fc6c706e2ee86
```

After an enhanced version is published, select its immutable commit tag and use the matching Compose file:

```sh
FRAUD_IMAGE=ghcr.io/wtfashwin/fraud-api:<published-commit-sha> \
  docker compose --env-file .env.example up -d --no-build --wait
```

Pulling an image alone does not start PostgreSQL or Redis. Avoid pairing historical images with this branch's worker command and schema. Release images include OCI source/documentation/revision metadata. The workflow tests both Linux amd64 and arm64 before publishing those architectures.

## API contract

| Route | Behavior |
| --- | --- |
| `GET /health`, `GET /status` | Process liveness. |
| `GET /ready` | 200 only when the loaded model, migrated explanation table, and broker are usable; otherwise 503. |
| `GET /model-info` | Ordered feature names and model source. |
| `POST /predict` | Validated raw numeric features; actual class-1 probability; a persisted explanation ID and successful queue dispatch. |
| `GET /explain/{transaction_id}` | 202 while pending, 200 when complete, 503 when failed, 404 for an unknown ID. |
| `GET /metrics` | API Prometheus metrics. |

Features must be finite numbers in this order: `Time`, `V1` through `V28`, `Amount`. `transaction_id` is optional; a server-generated UUID is the default. Reusing an ID returns 409 and does not overwrite the previous explanation. Queue/storage failure returns 503. A successful response means the task was dispatched; use `/explain` to confirm completion.

Linear explanations use exact feature contributions in **log-odds relative to a zero reference in scaled feature space**. They are not percentage contributions or causal evidence. Tree explanations identify their method in the returned payload. The API and worker use the same saved preprocessing bundle.

## Development and evaluation

Python 3.11 is the serving target. The default image installs `requirements-runtime.txt`; training and tests use the full requirements file:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
node --test web/tests/*.test.mjs
```

For a software-only training smoke test:

```sh
CI_SYNTHETIC_SAMPLES=500 python scripts/generate_synthetic_data.py
DATA_CSV=data/creditcard.csv MODEL_OUTPUT_DIR=models/trained python -m training.train
```

The generator writes metadata identifying the data as synthetic. Training splits raw data before fitting the scaler and applies preprocessing/SMOTE within each cross-validation training fold. It reports ROC AUC, average precision, and threshold-based metrics. `DECISION_THRESHOLD` changes evaluation metrics only; API classification uses the saved estimator’s `predict` method. Synthetic labels do not establish a useful real-world model.

Training writes a separate complete bundle to `models/trained/`; it does not replace the packaged logistic model or its scaler. Evaluate compatible held-out data explicitly. Do not train on the evaluation dataset:

```sh
EVAL_DATA_CSV=/path/to/heldout.csv MODEL_DIR=models/trained python -m training.evaluate
python scripts/validate_auc.py --help
```

MLflow logging is optional. Set `MLFLOW_TRACKING_URI` to enable tracking and `MLFLOW_REGISTER_MODEL=1` only for deliberate registration after evaluation. For MLflow/XGBoost serving, build with `--target training`, mount the complete bundle read-only, and set `MODEL_DIR` to its path. `MLFLOW_MODEL_URI` opts into loading a registered raw-input pipeline; the default API does not contact a registry at startup.

## CI and packaging

Pull requests run Python regression tests, JavaScript tests, synthetic training smoke, and real container smoke on Linux amd64 and arm64. The container smoke requires a prediction and a completed asynchronous explanation, and checks invalid input and duplicate IDs. A failed check stops publication.

After checks pass on `main`, the package job publishes immutable commit and `latest` tags with provenance and an SBOM. Pushes that change only Markdown or files under `docs/` skip this workflow; pull requests still run the full checks. A synthetic training artifact is uploaded separately and never substituted for the packaged model. The workflow publishes a package; it does not deploy a production system.

The Helm chart under `charts/fraud-detection/` is an optional deployment example with an explicit image tag. Historical raw manifests under `archive/k8s/`, older observability files, and worker-recovery notes are retained as references, not verified release paths. Kubernetes rollout, failover, monitoring dashboards, authentication, production load, and real fraud-model accuracy require separate validation.

## License

An earlier README stated MIT, but the repository contains no license file.
