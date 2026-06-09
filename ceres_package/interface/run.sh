#!/usr/bin/env bash
# Start the Ceres predict API (run from anywhere).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export PULL_FROM_GCP="${PULL_FROM_GCP:-false}"
exec uvicorn ceres_package.interface.api.fast:app \
  --host "${CERES_API_HOST:-127.0.0.1}" \
  --port "${CERES_API_PORT:-8000}" \
  --reload
