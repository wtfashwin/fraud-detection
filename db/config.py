"""Resolve an explicit database URL or the separate Compose credentials."""
import os

from sqlalchemy.engine import URL


DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@postgres:5432/fraud"
COMPOSE_DATABASE_MARKER = "fraud+compose://components"


def database_url():
    # Presence matters: an explicitly empty URL keeps its existing failure behavior.
    compose_enabled = os.getenv("FRAUD_COMPOSE_DATABASE") == "1"
    if "DATABASE_URL" in os.environ and not (
        compose_enabled and os.environ["DATABASE_URL"] == COMPOSE_DATABASE_MARKER
    ):
        return os.environ["DATABASE_URL"]
    if not compose_enabled:
        return DEFAULT_DATABASE_URL

    names = ("FRAUD_COMPOSE_DB_USER", "FRAUD_COMPOSE_DB_PASSWORD", "FRAUD_COMPOSE_DB_NAME")
    if any(not os.getenv(name) for name in names):
        # Do not include credential values in errors or logs.
        raise ValueError("Compose database configuration is incomplete")
    user, password, database = (os.environ[name] for name in names)
    # An object avoids parsing raw credentials and database names as URL syntax.
    return URL.create("postgresql+psycopg2", username=user, password=password,
                      host="postgres", port=5432, database=database)
