"""Load the trained model from GCS."""
from __future__ import annotations

import pickle

from google.cloud import storage

from ceres_package.interface.config import GCS_BUCKET, GCS_MODEL_BLOB


def load_model_from_gcs():
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(GCS_MODEL_BLOB)
    return pickle.loads(blob.download_as_bytes())
