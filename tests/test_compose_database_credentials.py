"""Compose credential checks with synthetic values; no database is contacted."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def base_environment():
    allowed = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
               "SYSTEMROOT", "WINDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def compose_environment(docker_config, password, username="synthetic-user", database="synthetic-db", env_file=None,
                        override_file=None):
    executable = shutil.which("docker-compose")
    docker = shutil.which("docker")
    if executable:
        cli = [executable]
    elif docker:
        cli = [docker, "compose"]
    else:
        pytest.skip("Docker Compose CLI is not installed")
    docker_config.mkdir()
    env = dict(base_environment(), API_PORT="8000", FRAUD_IMAGE="synthetic-fraud:test",
               GRAFANA_ADMIN_PASSWORD="synthetic-grafana", DOCKER_CONFIG=str(docker_config),
               DATABASE_URL="postgresql+psycopg2://remote.invalid/unrelated")
    if env_file is None:
        env.update(POSTGRES_USER=username, POSTGRES_PASSWORD=password, POSTGRES_DB=database)
        env_file = ROOT / ".env.example"
    version = subprocess.run([*cli, "version"], cwd=ROOT, env=env,
                             text=True, capture_output=True, timeout=15)
    if version.returncode:
        pytest.skip("Docker Compose plugin is not available")
    command = [*cli, "--project-directory", str(ROOT), "--file", str(ROOT / "docker-compose.yml")]
    if override_file is not None:
        command.extend(["--file", str(override_file)])
    command.extend(["--env-file", str(env_file), "config", "--format", "json"])
    result = subprocess.run(command,
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, "Docker Compose config failed for synthetic settings"
    data = json.loads(result.stdout)
    postgres = data["services"]["postgres"]["environment"]
    controls = (("FRAUD_COMPOSE_DB_USER", "POSTGRES_USER", username),
                ("FRAUD_COMPOSE_DB_PASSWORD", "POSTGRES_PASSWORD", password),
                ("FRAUD_COMPOSE_DB_NAME", "POSTGRES_DB", database))

    def service_environment(name):
        values = dict(data["services"][name]["environment"])
        # Some Compose versions escape dollars in rendered config. Determine
        # that independently for each field from the Postgres control value.
        for app_key, postgres_key, original in controls:
            control = postgres[postgres_key]
            if control == original:
                continue
            if control != original.replace("$", "$$"):
                raise AssertionError("Unexpected rendered synthetic database setting")
            if app_key in values:
                values[app_key] = values[app_key].replace("$$", "$")
        return values

    return service_environment("api"), service_environment("xai-worker")


def inspect_api_url(environment, expected):
    child = base_environment()
    child.update(environment)
    child.update(FRAUD_TEST_EXPECTED_USER=expected[0], FRAUD_TEST_EXPECTED_PASSWORD=expected[1],
                 FRAUD_TEST_EXPECTED_DB=expected[2])
    code = '''import json, os\ntry:\n from db.db import engine\n u = engine.url\n print(json.dumps({"ok": True, "user": u.username == os.environ["FRAUD_TEST_EXPECTED_USER"],\n  "password": u.password == os.environ["FRAUD_TEST_EXPECTED_PASSWORD"],\n  "database": u.database == os.environ["FRAUD_TEST_EXPECTED_DB"], "host": u.host == "postgres", "port": u.port == 5432}))\nexcept Exception as error:\n print(json.dumps({"ok": False, "error": type(error).__name__}))\n'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, "Synthetic API URL inspection failed"
    return json.loads(result.stdout)


def clean_environment(overrides):
    child = base_environment()
    child.update(overrides)
    return child


def inspect_explicit_url(environment):
    code = '''import json\ntry:\n from db.db import engine\n u = engine.url\n print(json.dumps({"ok": True, "user": u.username, "password": u.password,\n  "database": u.database, "host": u.host, "port": u.port}))\nexcept Exception as error:\n print(json.dumps({"ok": False, "error": type(error).__name__}))\n'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env=clean_environment(environment), text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, "Synthetic explicit URL inspection failed"
    return json.loads(result.stdout)


def run_alembic_offline(environment):
    return subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
                          cwd=ROOT, env=clean_environment(environment), text=True,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize("username,password,database", [
    ("user@:/?#% space", "synthetic@password", "synthetic?db/name#%"),
    ("user@:/?#% space", "synthetic%40password", "synthetic?db/name#%"),
    ("user@:/?#% space", "!a:/?#$%+ space", "synthetic?db/name#%"),
    ("user$only", "plain-password", "plain-database"),
    ("plain-user", "plain-password", "database$only"),
    ("user$$value", "pass$$value", "database$$value"),
    ("नाम", "ताला🔒@%$", "डेटा?नाम"),
])
def test_compose_app_and_worker_keep_exact_credentials(tmp_path, username, password, database):
    expected = (username, password, database)
    api, worker = compose_environment(tmp_path / "docker-config", password, expected[0], expected[2])
    for environment in (api, worker):
        # Host-side DATABASE_URL is not automatically passed into the service.
        assert environment["DATABASE_URL"] == "fraud+compose://components"
        assert inspect_api_url(environment, expected) == {
            "ok": True, "user": True, "password": True,
            "database": True, "host": True, "port": True,
        }


def test_explicit_database_url_is_not_replaced_by_compose_fields():
    environment = {
        "DATABASE_URL": "postgresql+psycopg2://explicit:encoded%40password@chosen:5433/explicit_db",
        "FRAUD_COMPOSE_DATABASE": "1", "FRAUD_COMPOSE_DB_USER": "other-user",
        "FRAUD_COMPOSE_DB_PASSWORD": "other-password", "FRAUD_COMPOSE_DB_NAME": "other-db",
    }
    assert inspect_api_url(environment, ("explicit", "encoded@password", "explicit_db")) == {
        "ok": True, "user": True, "password": True, "database": True, "host": False, "port": False,
    }


@pytest.mark.parametrize("explicit,expected", [
    ("postgresql+psycopg2://chosen:5433", {"user": None, "password": None, "database": None}),
    ("postgresql+psycopg2://named@chosen:5433", {"user": "named", "password": None, "database": None}),
    ("postgresql+psycopg2://named:encoded%40password@chosen:5433/", {"user": "named", "password": "encoded@password", "database": ""}),
])
def test_explicit_url_omissions_do_not_inherit_compose_fields(explicit, expected):
    env = {"DATABASE_URL": explicit, "FRAUD_COMPOSE_DATABASE": "1",
           "FRAUD_COMPOSE_DB_USER": "other", "FRAUD_COMPOSE_DB_PASSWORD": "synthetic-secret",
           "FRAUD_COMPOSE_DB_NAME": "other-db"}
    assert inspect_explicit_url(env) == {"ok": True, "host": "chosen", "port": 5433, **expected}


def test_explicit_empty_url_remains_an_error_even_with_compose_fields():
    env = {"DATABASE_URL": "", "FRAUD_COMPOSE_DATABASE": "1",
           "FRAUD_COMPOSE_DB_USER": "other", "FRAUD_COMPOSE_DB_PASSWORD": "synthetic-secret",
           "FRAUD_COMPOSE_DB_NAME": "other-db"}
    assert inspect_explicit_url(env) == {"ok": False, "error": "ArgumentError"}
    assert run_alembic_offline(env).returncode != 0


def test_unrelated_postgres_environment_does_not_change_existing_default():
    env = {"POSTGRES_USER": "other", "POSTGRES_PASSWORD": "synthetic-secret", "POSTGRES_DB": "other-db"}
    assert inspect_explicit_url(env) == {"ok": True, "user": "postgres", "password": "postgres",
                                         "database": "fraud", "host": "postgres", "port": 5432}


@pytest.mark.parametrize("missing", ["FRAUD_COMPOSE_DB_USER", "FRAUD_COMPOSE_DB_PASSWORD", "FRAUD_COMPOSE_DB_NAME"])
def test_incomplete_compose_settings_fail_before_api_or_migration(missing):
    env = {"FRAUD_COMPOSE_DATABASE": "1", "FRAUD_COMPOSE_DB_USER": "other",
           "FRAUD_COMPOSE_DB_PASSWORD": "synthetic-secret", "FRAUD_COMPOSE_DB_NAME": "other-db"}
    env.pop(missing)
    assert inspect_explicit_url(env) == {"ok": False, "error": "ValueError"}
    result = run_alembic_offline(env)
    assert result.returncode != 0
    assert "other-db" not in result.stderr and "synthetic-secret" not in result.stderr


def test_alembic_offline_keeps_explicit_encoded_url_precedence():
    env = {"DATABASE_URL": "postgresql+psycopg2://explicit:encoded%40password@chosen:5433/explicit_db",
           "FRAUD_COMPOSE_DATABASE": "1", "FRAUD_COMPOSE_DB_USER": "other"}
    result = run_alembic_offline(env)
    assert result.returncode == 0
    assert "CREATE TABLE shap_explanations" in result.stdout
    assert "encoded@password" not in result.stdout + result.stderr


def test_alembic_offline_accepts_compose_fields_without_printing_password(tmp_path):
    password = "synthetic@%/?#$"
    api, _ = compose_environment(tmp_path / "docker-config", password, database="synthetic?db/name#%")
    result = run_alembic_offline(api)
    assert result.returncode == 0
    assert "CREATE TABLE shap_explanations" in result.stdout
    assert password not in result.stdout + result.stderr


def test_base_compose_overrides_an_image_baked_database_url(tmp_path):
    expected = ("synthetic-user", "synthetic@password", "synthetic-db")
    api, worker = compose_environment(tmp_path / "docker-config", expected[1], expected[0], expected[2])
    # Docker service environment takes precedence over image ENV. This models
    # that precedence; no Docker daemon or real image is used by this check.
    baked = {"DATABASE_URL": "postgresql+psycopg2://baked:synthetic@remote.invalid/baked"}
    for environment in (api, worker):
        assert inspect_api_url({**baked, **environment}, expected) == {
            "ok": True, "user": True, "password": True,
            "database": True, "host": True, "port": True,
        }


@pytest.mark.parametrize("explicit", [
    "postgresql+psycopg2://named:encoded%40password@chosen:5433/explicit_db",
    "postgresql+psycopg2://chosen:5433",
    "",
])
def test_compose_service_database_url_override_keeps_precedence(tmp_path, explicit):
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"services": {name: {"environment": {"DATABASE_URL": explicit}}
                                                 for name in ("api", "xai-worker")}}))
    api, worker = compose_environment(tmp_path / "docker-config", "synthetic@password", override_file=override)
    for environment in (api, worker):
        assert environment["DATABASE_URL"] == explicit
        observed = inspect_explicit_url(environment)
        if explicit == "":
            assert observed == {"ok": False, "error": "ArgumentError"}
        elif "named:" in explicit:
            assert observed == {"ok": True, "user": "named", "password": "encoded@password",
                                "database": "explicit_db", "host": "chosen", "port": 5433}
        else:
            assert observed == {"ok": True, "user": None, "password": None,
                                "database": None, "host": "chosen", "port": 5433}


def test_reserved_marker_is_not_interpreted_outside_compose_mode():
    assert inspect_explicit_url({"DATABASE_URL": "fraud+compose://components"}) == {
        "ok": False, "error": "NoSuchModuleError",
    }


def test_single_quoted_env_file_preserves_a_literal_dollar(tmp_path):
    password = "synthetic$literal"
    env_file = tmp_path / "synthetic.env"
    env_file.write_text("POSTGRES_USER=synthetic-user\nPOSTGRES_PASSWORD='synthetic$literal'\nPOSTGRES_DB=synthetic-db\n")
    api, worker = compose_environment(tmp_path / "docker-config", password, env_file=env_file)
    expected = ("synthetic-user", password, "synthetic-db")
    for environment in (api, worker):
        assert inspect_api_url(environment, expected) == {
            "ok": True, "user": True, "password": True,
            "database": True, "host": True, "port": True,
        }
