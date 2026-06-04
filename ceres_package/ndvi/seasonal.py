"""Harvest-year seasonal aggregation from monthly NDVI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceres_package.ndvi.constants import DEPT_ID, SEASON_WINDOWS, SEASONAL_FEATURE_PREFIX
from ceres_package.ndvi.dept import normalize_dept_id


def build_seasonal_features(
    monthly: pd.DataFrame,
    *,
    ndvi_col: str = "ndvi_poly_mean",
    prefix: str = SEASONAL_FEATURE_PREFIX,
    start_year: int = 2010,
    end_year: int = 2024,
    dept_col: str = DEPT_ID,
) -> pd.DataFrame:
    """
    Build dept × harvest-year seasonal medians from monthly polygon NDVI.

    Expects calendar ``year``, ``month``, ``ndvi_poly_mean``, and ``DEPT_ID``.
    """
    d = monthly.copy()
    if dept_col not in d.columns and "department_code" in d.columns:
        d = d.rename(columns={"department_code": dept_col})
    d[dept_col] = normalize_dept_id(d[dept_col])
    d["year"] = d["year"].astype(int)
    d["month"] = d["month"].astype(int)
    planting_months = SEASON_WINDOWS["planting"]

    rows: list[dict] = []
    for harvest in range(start_year, end_year + 1):
        for dept, gdept in d.groupby(dept_col):

            def _med(mask: pd.Series) -> tuple[float, int]:
                s = gdept.loc[mask, ndvi_col].dropna()
                if s.empty:
                    return np.nan, 0
                return float(s.median()), int(len(s))

            rec: dict = {dept_col: dept, "year": harvest}
            pmask = (gdept["year"] == harvest - 1) & (gdept["month"].isin(planting_months))
            med, n = _med(pmask)
            rec[f"{prefix}_planting_med"] = med
            rec[f"{prefix}_planting_n"] = n

            for season in ("winter", "spring", "summer"):
                smask = (gdept["year"] == harvest) & (gdept["month"].isin(SEASON_WINDOWS[season]))
                med, n = _med(smask)
                rec[f"{prefix}_{season}_med"] = med
                rec[f"{prefix}_{season}_n"] = n

            amask = ((gdept["year"] == harvest - 1) & (gdept["month"].isin([9, 10, 11, 12]))) | (
                (gdept["year"] == harvest) & (gdept["month"].isin(range(1, 9)))
            )
            med, n = _med(amask)
            rec[f"{prefix}_annual_med"] = med
            rec[f"{prefix}_annual_n"] = n

            spring = rec.get(f"{prefix}_spring_med", np.nan)
            planting = rec.get(f"{prefix}_planting_med", np.nan)
            summer = rec.get(f"{prefix}_summer_med", np.nan)
            if not np.isnan(spring) and not np.isnan(planting):
                rec[f"{prefix}_spring_minus_planting"] = spring - planting
            if not np.isnan(summer) and not np.isnan(spring):
                rec[f"{prefix}_summer_minus_spring"] = summer - spring
            rows.append(rec)

    out = pd.DataFrame(rows)
    out[dept_col] = normalize_dept_id(out[dept_col])
    return out.sort_values([dept_col, "year"]).reset_index(drop=True)
