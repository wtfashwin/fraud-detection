#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL=${DATABASE_URL:-postgresql+psycopg2://postgres:postgres@postgres:5432/fraud}

if command -v alembic >/dev/null 2>&1; then
  echo "Running alembic upgrade head"
  alembic upgrade head
else
  echo "alembic not found in PATH; skipping migrations"
fi

exec "$@"
