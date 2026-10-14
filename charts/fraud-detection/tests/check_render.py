"""Check rendered chart wiring; requires Helm and PyYAML, never a cluster."""
import argparse
from pathlib import Path
import subprocess

import yaml


CHART = Path(__file__).resolve().parents[1]


def render(helm, release="fraud", settings=()):
    args = [helm, "template", release, str(CHART), "--set", "image.tag=render-test", "--set", "postgresql.auth.password=render-test-only"]
    for setting in settings:
        args.extend(["--set", setting])
    return [item for item in yaml.safe_load_all(subprocess.check_output(args, text=True)) if item]


def workload(documents, component):
    return next(item for item in documents if item["kind"] == "Deployment" and item["metadata"].get("labels", {}).get("app.kubernetes.io/component") == component)


def env(container):
    return {item["name"]: item.get("value", item.get("valueFrom")) for item in container["env"]}


def check(helm):
    subprocess.run([helm, "lint", str(CHART), "--set", "image.tag=render-test", "--set", "postgresql.auth.password=render-test-only"], check=True)
    default = render(helm)
    api, worker = (workload(default, component) for component in ["api", "worker"])
    pods = [item["spec"]["template"]["spec"] for item in [api, worker]]
    api_container, worker_container = [pod["containers"][0] for pod in pods]
    assert all(container["image"] == "ghcr.io/wtfashwin/fraud-api:render-test" for pod in pods for container in pod["containers"] + pod["initContainers"])
    service = next(item for item in default if item["kind"] == "Service" and item["metadata"]["name"] == api["metadata"]["name"])
    assert service["spec"]["selector"] == api["spec"]["selector"]["matchLabels"]
    for container in [api_container, worker_container]:
        security = container["securityContext"]
        assert security["runAsNonRoot"] and security["runAsUser"] == security["runAsGroup"] == 10001
    assert api_container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert api_container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert env(api_container)["RUN_MIGRATIONS"] == "1"
    assert env(worker_container)["RUN_MIGRATIONS"] == "0"
    assert worker_container["args"][2] == "api.worker.app"
    assert "command" not in worker_container  # Preserve migration entrypoint.
    assert pods[1]["initContainers"][0]["name"] == "wait-for-schema"
    assert "shap_explanations" in pods[1]["initContainers"][0]["args"][0]
    probe = worker_container["readinessProbe"]["exec"]["command"]
    assert probe[:2] == ["/bin/sh", "-ec"] and '"celery@$HOSTNAME"' in probe[2]
    assert "lifecycle" not in worker_container
    assert env(worker_container)["CELERY_BROKER_URL"] == "redis://fraud-redis-master:6379/0"
    assert "MLFLOW_TRACKING_URI" not in env(api_container)
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env(api_container)
    assert not any(item["kind"] == "ScaledObject" for item in default)
    assert not any("mlflow" in item["metadata"]["name"] for item in default)
    optional = render(helm, settings=["worker.autoscaling.enabled=true", "mlflowEnabled=true", "otel.enabled=true", "otel.endpoint=http://collector:4318/v1/traces", "imagePullSecrets[0].name=registry-secret"])
    scaler = next(item for item in optional if item["kind"] == "ScaledObject")
    assert scaler["spec"]["scaleTargetRef"]["name"] == worker["metadata"]["name"]
    assert scaler["spec"]["triggers"][0]["metadata"]["address"] == "fraud-redis-master:6379"
    for component in ["api", "worker"]:
        pod = workload(optional, component)["spec"]["template"]["spec"]
        assert pod["imagePullSecrets"] == [{"name": "registry-secret"}]
        assert env(pod["containers"][0])["MLFLOW_TRACKING_URI"] == "http://fraud-mlflow:5000"
    overridden = render(helm, release="another", settings=["redis.fullnameOverride=queue", "postgresql.fullnameOverride=database"])
    assert env(workload(overridden, "api")["spec"]["template"]["spec"]["containers"][0])["CELERY_BROKER_URL"] == "redis://queue-master:6379/0"
    database_secret = next(item for item in overridden if item["kind"] == "Secret" and "DATABASE_URL" in item.get("stringData", {}))
    assert "@database:5432/fraud" in database_secret["stringData"]["DATABASE_URL"]
    assert workload(overridden, "api")["spec"]["selector"]["matchLabels"]["app.kubernetes.io/instance"] == "another"
    for settings, expected in [
        (["postgresql.auth.password=render-test-only"], "Set image.tag"),
        (["image.tag=render-test"], "Set postgresql.auth.password"),
        (["image.tag=render-test", "postgresql.auth.password=render-test-only", "redis.sentinel.enabled=true"], "standalone"),
    ]:
        args = [helm, "template", "fraud", str(CHART)]
        for setting in settings:
            args.extend(["--set", setting])
        rejected = subprocess.run(args, text=True, capture_output=True)
        assert rejected.returncode != 0 and expected in rejected.stderr
    print("Helm manifest checks passed: baseline, optional components, shared artifact, probes, schema gate, selectors, dependency host overrides.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helm", default="helm")
    check(parser.parse_args().helm)
