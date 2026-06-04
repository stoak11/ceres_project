"""Monthly polygon NDVI table preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ceres_package.ndvi.constants import DEPT_ID, MONTHLY_COLUMNS, MONTHLY_DROP_COLUMNS
from ceres_package.ndvi.dept import ensure_dept_id_column, normalize_dept_id


def slim_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep canonical monthly columns only; drop anchor/std QA fields."""
    out = ensure_dept_id_column(df)
    out = out.drop(columns=[c for c in MONTHLY_DROP_COLUMNS if c in out.columns], errors="ignore")
    missing = [c for c in MONTHLY_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Monthly NDVI missing columns: {missing}")
    out[DEPT_ID] = normalize_dept_id(out[DEPT_ID])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["harvest_year"] = out["harvest_year"].astype(int)
    return out[list(MONTHLY_COLUMNS)].sort_values([DEPT_ID, "year", "month"]).reset_index(drop=True)


def read_monthly_source(path: str | Path) -> pd.DataFrame:
    """Load upstream monthly CSV (ml-farm or Ceres export) and apply slim schema."""
    return slim_monthly_columns(pd.read_csv(path))
