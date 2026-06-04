"""Load NDVI tables and join to yield."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ceres_package.ndvi.constants import DEPT_ID
from ceres_package.ndvi.dept import normalize_dept_id
from ceres_package.ndvi.monthly import slim_monthly_columns
from ceres_package.ndvi.paths import default_monthly_path, default_seasonal_path


def load_ndvi_monthly(path: str | Path | None = None) -> pd.DataFrame:
    """Load slim monthly NDVI (enforces column contract)."""
    p = Path(path) if path else default_monthly_path()
    return slim_monthly_columns(pd.read_csv(p))


def load_ndvi_seasonal(path: str | Path | None = None) -> pd.DataFrame:
    """Load seasonal NDVI; exposes ``dept_code`` alias for ML panels."""
    p = Path(path) if path else default_seasonal_path()
    df = pd.read_csv(p)
    if DEPT_ID in df.columns:
        df = df.rename(columns={DEPT_ID: "dept_code"})
    elif "department_code" in df.columns:
        df = df.rename(columns={"department_code": "dept_code"})
    df["dept_code"] = normalize_dept_id(df["dept_code"])
    df["year"] = df["year"].astype(int)
    return df.sort_values(["dept_code", "year"]).reset_index(drop=True)


def build_ndvi_yield_panel(
    ndvi: pd.DataFrame,
    yield_df: pd.DataFrame,
    *,
    dept_col_ndvi: str = "dept_code",
    dept_col_yield: str = "dept_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """Inner-join seasonal NDVI and yield on department × harvest year."""
    panel = ndvi.merge(
        yield_df,
        left_on=[dept_col_ndvi, year_col],
        right_on=[dept_col_yield, year_col],
        how="inner",
        suffixes=("", "_y"),
    )
    drop_cols = [c for c in panel.columns if c.endswith("_y")]
    return panel.drop(columns=drop_cols).sort_values([dept_col_ndvi, year_col]).reset_index(
        drop=True
    )
