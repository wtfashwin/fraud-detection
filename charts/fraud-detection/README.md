# Fraud detection Helm chart

This chart installs the API, explanation worker, PostgreSQL, and **standalone Redis**. It is a local/demo deployment baseline, not a tested high-availability setup. Redis authentication is disabled and its service stays inside the cluster; do not expose it publicly. Authenticated Redis or Sentinel require different broker and KEDA credentials/configuration and are rejected by this baseline.

Select an image built from the updated API/worker code using `image.tag`; there is deliberately no invented default release. API, worker, and init containers use this one artifact. The image must contain `/app/models`, the migration entrypoint, and `api.worker.app`. `imagePullSecrets` is optional and applies to both deployments.

Set `postgresql.auth.password` in a private values file. Keep the same password when reusing PostgreSQL storage. The API waits for database connectivity and runs Alembic through its image entrypoint. The worker sets `RUN_MIGRATIONS=0`, waits for the explanation schema, and then starts Celery. One API replica is the default migration owner; coordinate migrations separately before increasing replicas during a schema upgrade.

By default, API liveness uses `/health`; readiness uses `/ready` to check the loaded model, migrated database, and broker. Worker readiness pings only that pod's Celery node. Shutdown uses Celery's normal TERM handling and never broadcasts a shutdown to other workers.

## Render and inspect

Use Helm 3.7 or newer. Dependencies are already packaged in `charts/` at the versions in `Chart.lock`; no dependency download is needed to lint or render.

```sh
helm lint charts/fraud-detection --set image.tag=local-verification --set postgresql.auth.password=chart-test-only
helm template fraud charts/fraud-detection --set image.tag=local-verification --set postgresql.auth.password=chart-test-only
python charts/fraud-detection/tests/check_render.py --helm helm
```

`local-verification` and `chart-test-only` are render-test values, not a published image or usable deployment credentials. Replace them with the actual built image tag and private password when installing.

The API Service defaults to ClusterIP port 8000. After installation and readiness, forward the Service's port and open `/demo/` or `/docs`. This chart does not create an ingress or publish the demo.

## Optional components

- `worker.autoscaling.enabled=true` renders a KEDA ScaledObject targeting the actual worker Deployment and Redis queue. Install KEDA and its CRDs separately first. The default is disabled, with one worker replica.
- `mlflowEnabled=true` enables the packaged MLflow dependency and supplies its tracking URL. This is separate from `mlflow.*` because the vendored chart rejects an `enabled` property. Configure MLflow's durable backend and artifact storage yourself. Enabling tracking does not register a model or prove registry loading.
- `otel.enabled=true` requires `otel.endpoint` pointing at an existing collector. No collector is installed.
- `monitoring.enabled=true` adds API Prometheus scrape annotations. No Prometheus or Grafana deployment is installed.

## Upgrade and verification limits

Version 0.2.0 scopes selectors to the Helm release so separate releases cannot select each other's pods. Kubernetes Deployment selectors are immutable: an existing 0.1.0 Deployment requires an explicitly planned replacement, not an automatic in-place selector update. Preserve its database volumes and secrets when planning that replacement.

Lint and render checks verify manifest structure and wiring. They do not prove cluster rollout, dependency image availability, storage provisioning, KEDA operation, or live model behavior. The existing vendored dependency versions are unchanged; verify their images and storage configuration in the target cluster before deployment.

## Optional demo access gate

Set these non-secret values to enable the shared API gate:

```yaml
demoAuth:
  mode: basic
  existingSecret: fraud-demo-auth
monitoring:
  enabled: false
```

Create and manage that Secret outside this chart with `username` and `password` keys. The API reads them through `secretKeyRef`; worker and init containers receive no demo credentials. No credential value belongs in chart values or command arguments. Missing Secret keys prevent container startup; empty/invalid credentials prevent application startup. Secret environment changes require an explicit API restart/rollout to rotate every replica.

Basic mode requires an existing Secret and disables the anonymous scrape annotations; conflicting configuration fails rendering. `/metrics` remains authenticated. Configure an authenticated collector separately before claiming monitoring works. `off` mode rejects an existing Secret reference so a presumed gate is not silently ignored. Quote `"off"` in YAML values.

Secured startup/liveness probes use exact public `/status`. Readiness executes `python -m api.probe` against fixed loopback `/ready`, using API credentials without putting them into probe arguments or manifests. Dependency failures still make the pod unready. Default local probes retain their previous HTTP paths; worker probes are unchanged.

Open the demo on the API's own origin and use the browser Basic prompt. There is no per-user access model or application logout. The chart still creates no ingress or TLS endpoint. Before external access, verify HTTPS, network isolation, secret handling, budgets and retention separately. Render/probe tests do not establish a cluster rollout or worker completion.
