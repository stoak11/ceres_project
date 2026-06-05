# Ceres project layout (after registry merge)

Quick map so a `git pull` does not feel like a restructure.

## Code (tracked)

```
ceres_package/
  ml_logic/          # data load, preprocess, registry (train/predict/registry)
  ndvi/              # NDVI seasonal products + pipelines
  params.py          # GCS paths + DATA_CONFIG local cache targets
scripts/
  ml/                # train_and_register, promote_run
  ndvi/              # export_products
docs/                # ML_REGISTRY*, GCP_VMS, FEATURE_ENGINEERING, NDVI*
infra/gcp/           # ceres-dev-01 VM scripts
tests/
```

## Data (mostly gitignored)

```
raw_data/            # LOCAL CACHE for load_from_gcp — skeleton + README in git
data/NDVI/           # NDVI CSV exports (gitignored except docs)
data/marts/          # FE outputs for registry (gitignored, create locally)
models/registry/     # ML runs (gitignored)
```

**Important:** Removing `.gitkeep` under `raw_data/` broke fresh clones — restored in hotfix. Large files were always ignored; only the **folder scaffold** disappeared from git.

## What changed vs 2024 “PipeML” layout

| Change | Impact |
|--------|--------|
| + Registry (`train.py`, `registry.py`, …) | New path to train/save models — see `ML_REGISTRY_ONE_PAGER.md` |
| + `ceres_package/ndvi/` | National polygon NDVI; does not remove `raw_data/` |
| `ml_model.py` | No longer runs on import — use `python -m ceres_package.ml_logic.ml_model` or registry |
| `raw_data/**/.gitkeep` | Restored — required for `download_blob` local paths |

## Team workflow (unchanged for data)

1. `load_from_gcp("production" | "meteo_dept" | "soil" | …)` → fills `raw_data/…`
2. Feature engineering → `data/marts/*.parquet` (see `FEATURE_ENGINEERING.md`)
3. `scripts/ml/train_and_register.py --mart …` → `models/registry/`
