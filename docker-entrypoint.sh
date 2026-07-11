#!/bin/sh
set -eu

# The product experience owns PostgreSQL tables. Migrate before serving traffic so a
# newly deployed API never exposes routes against an older schema.
if [ "${POSTGRES_ENABLED:-false}" = "true" ]; then
  alembic upgrade head
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
