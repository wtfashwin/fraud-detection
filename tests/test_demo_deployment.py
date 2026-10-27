"""Inspect actual Compose expansion; no Docker daemon or services are required."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_compose_auth_settings_are_api_only_and_probe_has_no_credentials():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose CLI is unavailable; no service is started")
    version = subprocess.run([docker, "compose", "version"], text=True, capture_output=True)
    if version.returncode:
        error = (version.stdout + version.stderr).lower()
        if "compose" in error and ("unknown command" in error or "not a docker command" in error):
            pytest.skip("Docker Compose subcommand is unavailable; no service is started")
        version.check_returncode()
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "DEMO_AUTH_MODE": "basic", "DEMO_AUTH_USERNAME": "synthetic-compose",
           "DEMO_AUTH_PASSWORD": "synthetic-compose-password"}
    result = subprocess.run([docker, "compose", "--env-file", ".env.example", "config", "--format", "json"],
                            cwd=root, env=env, text=True, capture_output=True, check=True)
    services = json.loads(result.stdout)["services"]
    api = services["api"]
    assert api["environment"].get("DEMO_AUTH_MODE") == "basic"
    assert api["environment"]["DEMO_AUTH_USERNAME"] == "synthetic-compose"
    assert api["environment"]["DEMO_AUTH_PASSWORD"] == "synthetic-compose-password"
    assert api["healthcheck"]["test"] == ["CMD", "python", "-m", "api.probe"]
    assert api["ports"][0]["host_ip"] == "127.0.0.1"
    for name, service in services.items():
        if name != "api":
            assert not any(key.startswith("DEMO_AUTH_") for key in service.get("environment", {}))
