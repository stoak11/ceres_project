"""Write Ceres NDVI products to ``data/NDVI/`` (sync to GCS separately)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ceres_package.ndvi.monthly import read_monthly_source, slim_monthly_columns
from ceres_package.ndvi.paths import (
    NDVI_DIR,
    MONTHLY_FILENAME,
    SEASONAL_FILENAME,
    default_monthly_source_path,
)
from ceres_package.ndvi.seasonal import build_seasonal_features


def export_monthly_csv(
    monthly: pd.DataFrame,
    out_dir: Path,
    *,
    filename: str = MONTHLY_FILENAME,
) -> Path:
    """Write slim monthly NDVI CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    slim_monthly_columns(monthly).to_csv(path, index=False)
    return path


def export_seasonal_csv(
    monthly: pd.DataFrame,
    out_dir: Path,
    *,
    year_min: int = 2010,
    year_max: int = 2024,
    filename: str = SEASONAL_FILENAME,
) -> Path:
    """Aggregate monthly → seasonal and write CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seasonal = build_seasonal_features(
        slim_monthly_columns(monthly),
        start_year=year_min,
        end_year=year_max,
    )
    path = out_dir / filename
    seasonal.to_csv(path, index=False)
    return path


def export_ndvi_products(
    *,
    monthly_src: str | Path | None = None,
    out_dir: str | Path | None = None,
    year_min: int = 2010,
    year_max: int = 2024,
) -> dict[str, Path]:
    """
    Regenerate both NDVI CSVs (``DEPT_ID``, slim monthly columns).

    Returns paths ``monthly`` and ``seasonal``.
    """
    src = Path(monthly_src) if monthly_src else default_monthly_source_path()
    dest = Path(out_dir) if out_dir else NDVI_DIR

    monthly = read_monthly_source(src)
    monthly_path = export_monthly_csv(monthly, dest)
    seasonal_path = export_seasonal_csv(
        monthly, dest, year_min=year_min, year_max=year_max
    )
    return {"monthly": monthly_path, "seasonal": seasonal_path}
