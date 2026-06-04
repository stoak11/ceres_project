"""NDVI column contracts and agronomic season windows."""

from __future__ import annotations

DEPT_ID = "DEPT_ID"

# Dropped on export (anchor QA / redundant spread — not used in Ceres v1)
MONTHLY_DROP_COLUMNS = (
    "ndvi_anchor_median",
    "n_scenes_anchor",
    "ndvi_poly_std",
)

MONTHLY_COLUMNS = (
    DEPT_ID,
    "department_name",
    "year",
    "month",
    "scene_id",
    "harvest_year",
    "ndvi_poly_mean",
    "k_points_used",
)

SEASON_WINDOWS: dict[str, list[int]] = {
    "planting": [9, 10, 11, 12],
    "winter": [1, 2],
    "spring": [3, 4, 5, 6],
    "summer": [7, 8],
}

SEASONAL_FEATURE_PREFIX = "modis"

NDVI_MED_COLS = [
    "modis_planting_med",
    "modis_winter_med",
    "modis_spring_med",
    "modis_summer_med",
    "modis_annual_med",
]

NDVI_CONTRAST_COLS = [
    "modis_spring_minus_planting",
    "modis_summer_minus_spring",
]

NDVI_COUNT_COLS = [
    "modis_planting_n",
    "modis_winter_n",
    "modis_spring_n",
    "modis_summer_n",
    "modis_annual_n",
]
