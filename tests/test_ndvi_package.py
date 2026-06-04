"""Tests for ceres_package.ndvi."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ceres_package.ndvi.constants import DEPT_ID, MONTHLY_COLUMNS, MONTHLY_DROP_COLUMNS
from ceres_package.ndvi.export import export_ndvi_products
from ceres_package.ndvi.io import load_ndvi_monthly, load_ndvi_seasonal
from ceres_package.ndvi.monthly import slim_monthly_columns
from ceres_package.ndvi.paths import NDVI_DIR
from ceres_package.ndvi.pipelines import build_features, list_pipelines
from ceres_package.ndvi.seasonal import build_seasonal_features


def test_slim_monthly_drops_columns() -> None:
    raw = pd.DataFrame(
        {
            "department_code": ["01"],
            "department_name": ["Ain"],
            "year": [2020],
            "month": [3],
            "scene_id": ["x"],
            "harvest_year": [2020],
            "ndvi_poly_mean": [0.5],
            "k_points_used": [15],
            "ndvi_anchor_median": [0.3],
            "n_scenes_anchor": [1],
            "ndvi_poly_std": [0.1],
        }
    )
    out = slim_monthly_columns(raw)
    assert list(out.columns) == list(MONTHLY_COLUMNS)
    assert DEPT_ID in out.columns
    for c in MONTHLY_DROP_COLUMNS:
        assert c not in out.columns


def test_export_and_load_roundtrip() -> None:
    if not Path(
        "/mnt/c/Users/stani/Documents/ml-farm-recolt-forecast/data/processed/"
        "ndvi_monthly_by_department_polygon.csv"
    ).exists():
        return
    paths = export_ndvi_products(out_dir=NDVI_DIR)
    m = load_ndvi_monthly(paths["monthly"])
    s = load_ndvi_seasonal(paths["seasonal"])
    assert len(m) > 1000
    assert "dept_code" in s.columns
    assert "phenology_anomaly" in list_pipelines()
    X = build_features("spring_only", s.head(50), s.head(25).index)
    assert "modis_spring_med" in X.columns


def test_build_seasonal_smoke() -> None:
    monthly = pd.DataFrame(
        {
            DEPT_ID: ["01"] * 4,
            "department_name": ["Ain"] * 4,
            "year": [2019, 2019, 2020, 2020],
            "month": [3, 4, 3, 4],
            "scene_id": ["s"] * 4,
            "harvest_year": [2020] * 4,
            "ndvi_poly_mean": [0.5, 0.6, 0.55, 0.65],
            "k_points_used": [15] * 4,
        }
    )
    out = build_seasonal_features(monthly, start_year=2020, end_year=2020)
    assert len(out) == 1
    assert abs(out.iloc[0]["modis_spring_med"] - 0.6) < 1e-9
