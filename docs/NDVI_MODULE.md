# NDVI module — team guide

Function-based API in **`ceres_package.ndvi`**.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Data layout

```
data/NDVI/          # local export (CSVs gitignored; see .gitignore)
```

**GCS:** `gs://ceres-ai-bucket/NDVI/` — see **`GCP_NDVI_SYNC.md`**.

Monthly key: **`DEPT_ID`** (not `department_code`).

## Regenerate CSVs

```bash
python scripts/ndvi/export_products.py
```

Or in Python:

```python
from ceres_package.ndvi import export_ndvi_products

paths = export_ndvi_products()  # reads ml-farm source by default
```

Override source / output:

```python
export_ndvi_products(
    monthly_src="/path/to/upstream_monthly.csv",
    out_dir="data/NDVI",
    year_min=2010,
    year_max=2024,
)
```

Env vars: `CERES_NDVI_MONTHLY_SOURCE`, `CERES_NDVI_MONTHLY`, `CERES_NDVI_SEASONAL`.

## Load & join yield

```python
from ceres_package.ndvi import (
    load_ndvi_seasonal,
    load_ndvi_monthly,
    build_ndvi_yield_panel,
    build_features,
    list_pipelines,
)

monthly = load_ndvi_monthly()
seasonal = load_ndvi_seasonal()

# yield_df: columns dept_code, year, yield_t_ha, ...
# panel = build_ndvi_yield_panel(seasonal, yield_df)
```

## Public functions

| Function | Role |
|----------|------|
| `export_ndvi_products` | Write both CSVs |
| `read_monthly_source` | Slim columns from upstream file |
| `build_seasonal_features` | Monthly → seasonal aggregation |
| `load_ndvi_monthly` / `load_ndvi_seasonal` | Load Ceres products |
| `build_ndvi_yield_panel` | Join NDVI + yield |
| `list_pipelines` / `build_features` | Seasonal feature sets for ML |
