"""
NDVI polygon workflow (K=15 dept mean → seasonal features → ML panels).

Public API for the team — prefer these imports over script internals.
"""

from ceres_package.ndvi.constants import (
    DEPT_ID,
    MONTHLY_COLUMNS,
    SEASONAL_FEATURE_PREFIX,
    SEASON_WINDOWS,
)
from ceres_package.ndvi.export import export_monthly_csv, export_ndvi_products, export_seasonal_csv
from ceres_package.ndvi.monthly import read_monthly_source, slim_monthly_columns
from ceres_package.ndvi.io import (
    build_ndvi_yield_panel,
    load_ndvi_monthly,
    load_ndvi_seasonal,
)
from ceres_package.ndvi.paths import (
    NDVI_DIR,
    PROJECT_ROOT,
    default_monthly_path,
    default_seasonal_path,
)
from ceres_package.ndvi.pipelines import build_features, list_pipelines
from ceres_package.ndvi.seasonal import build_seasonal_features

__all__ = [
    "MONTHLY_COLUMNS",
    "NDVI_DIR",
    "PROJECT_ROOT",
    "SEASON_WINDOWS",
    "SEASONAL_FEATURE_PREFIX",
    "build_features",
    "build_ndvi_yield_panel",
    "build_seasonal_features",
    "DEPT_ID",
    "export_monthly_csv",
    "export_seasonal_csv",
    "read_monthly_source",
    "slim_monthly_columns",
    "default_monthly_path",
    "default_seasonal_path",
    "export_ndvi_products",
    "list_pipelines",
    "load_ndvi_monthly",
    "load_ndvi_seasonal",
]
