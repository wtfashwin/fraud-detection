# Contributing

Keep changes focused and preserve real-data and model provenance. Use Python 3.11 for the serving target. See the README for local environment setup.

```sh
python -m pytest -q
node --test web/tests/*.test.mjs
python charts/fraud-detection/tests/check_render.py --helm helm
docker compose --env-file .env.example up --build -d --wait
python scripts/smoke_test.py
```

The PR workflow repeats these checks, builds the optional CPU training image, and verifies real prediction/explanation requests on amd64 and arm64. Describe what changed, what was tested, and any remaining limits. Keep synthetic data clearly labeled; do not claim fraud quality from smoke fixtures.

Store current model code in `training/`, HTTP/worker code in `api/`, and the dependency-free browser demo in `web/`. Earlier experiments belong in `archive/` and are outside current runtime checks. Commit no credentials, real transaction data, generated model bundles, or local environment files.

The optional demo gate is covered by `tests/test_demo_access.py`, `tests/test_demo_probe.py`, and `tests/test_demo_deployment.py`, plus secured chart cases in `check_render.py`. Probe tests briefly bind loopback port 8000 with invented HTTP responses; Compose expansion needs its CLI but starts no services. Keep real credentials out of fixtures and logs. A real same-origin browser check and authenticated API/worker integration remain distinct from ASGI and manifest checks.

The hosted container workflow adds an authenticated stack check on both existing Linux runners. It reuses the built image in a separate Compose project on loopback port 18001, with its own database, Redis and worker. Synthetic credentials exist only in that step and its child environments. Its cleanup targets only the temporary project. The original core and monitoring stacks keep their configuration.

`scripts/smoke_demo_auth.py` is an HTTP client with no Docker operations. It reads credentials from the environment and checks denial, admitted demo assets, readiness, synthetic scoring and a saved queued explanation reaching exact `COMPLETED`. Local tests use invented loopback responses and fake Docker commands to exercise workflow isolation and cleanup. Those tests do not establish actual container or worker completion. The hosted receipt establishes that only after a successful workflow run. Browser authentication prompts and browser credential reuse require a separate browser check.
