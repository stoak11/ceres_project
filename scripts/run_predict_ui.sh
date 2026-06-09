#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
exec streamlit run apps/predict/streamlit_app.py --server.port "${CERES_UI_PORT:-8502}" "$@"
