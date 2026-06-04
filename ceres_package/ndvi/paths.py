"""Project paths for NDVI artifacts."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Local export dir (gitignored); canonical copy on gs://ceres-ai-bucket/NDVI/
NDVI_DIR = PROJECT_ROOT / "data" / "NDVI"

MONTHLY_FILENAME = "ndvi_monthly_by_department_polygon.csv"
SEASONAL_FILENAME = "ndvi_season_features.csv"


def default_monthly_path() -> Path:
    override = os.environ.get("CERES_NDVI_MONTHLY")
    return Path(override) if override else NDVI_DIR / MONTHLY_FILENAME


def default_seasonal_path() -> Path:
    override = os.environ.get("CERES_NDVI_SEASONAL")
    return Path(override) if override else NDVI_DIR / SEASONAL_FILENAME


def default_monthly_source_path() -> Path:
    """Upstream build from ml-farm (full columns before Ceres slim export)."""
    override = os.environ.get("CERES_NDVI_MONTHLY_SOURCE")
    if override:
        return Path(override)
    return Path(
        "/mnt/c/Users/stani/Documents/ml-farm-recolt-forecast/data/processed/"
        "ndvi_monthly_by_department_polygon.csv"
    )
