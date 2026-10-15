#!/bin/sh
set -eu
if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
  echo "Running database migrations"
  alembic upgrade head
fi
exec "$@"
