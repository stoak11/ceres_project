"""Settings for the team prediction API."""
from __future__ import annotations

import os
from pathlib import Path

from ceres_package.params import BUCKET_NAME, PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)

# GCS model (private bucket — needs Application Default Credentials)
GCS_BUCKET = os.environ.get("CERES_GCS_BUCKET", BUCKET_NAME or "ceres-ai-bucket")
GCS_MODEL_BLOB = os.environ.get("CERES_GCS_MODEL_BLOB", "models/ml_model.pkl")

# Same split as ceres_package.ml_logic.ml_pipeline.prepare_data
TEST_SIZE = 0.2
RANDOM_STATE = 42

YIELD_UNIT = "q/ha"
