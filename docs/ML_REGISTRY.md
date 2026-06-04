# ML registry — maintainer reference

Team workflow: **`ML_REGISTRY_ONE_PAGER.md`**.

## Modules (registry path)

| Module | Responsibility |
|--------|----------------|
| `model_specs.py` | Catalog: build sklearn pipelines, rule scoring (`lag1`) |
| `train.py` | `TemporalSplit`, `train_and_register` (fit + holdout metrics + `save_run`) |
| `registry.py` | Disk I/O, `production.json`, optional `upload_run_to_gcs` |
| `predict.py` | `predict_frame`, `predict`, `predict_production` |

## Layout

```
models/registry/          # LOCAL_REGISTRY_PATH (gitignored)
  <run_id>/
    manifest.json
    pipeline.joblib       # sklearn only
  production.json         # alias -> run_id
```

## CLI

```bash
PYTHONPATH=. python scripts/ml/train_and_register.py \
  --mart path/to/mart.parquet \
  --model-id random_forest_default \
  --features col_a col_b \
  --train-max-year 2019 --test-min-year 2020

PYTHONPATH=. python scripts/ml/promote_run.py <run_id>
```

## GCS

`save_run` never uploads. After training:

```python
from ceres_package.ml_logic.registry import upload_run_to_gcs
upload_run_to_gcs(run_id)
```

Requires `BUCKET_NAME` and `GCP_PROJECT` in the environment.

---

## How this fits the rest of the repo (post-merge)

| Piece | Registry? | Notes |
|-------|-----------|--------|
| **`train_and_register` + CLIs** | **Yes** | Mart in → run folder out. Mart must already exist (`DEPT_ID`, `ANNEE`, `RENDEMENT` + features). |
| **`data.py` + `load_from_gcp`** | **Upstream** | Pulls sources from GCS into `data/` / cached paths (`params.DATA_CONFIG`). Builds panels for FE — **not** the registry. |
| **`ml_preprocess.py`** | **Upstream** | Cleaning / joins for exploratory pipelines. |
| **`ml_pipeline.py`** | **No** | Random train/test split + grid helpers for **notebooks** only. |
| **`ml_model.py`** | **No** | Legacy one-shot XGB script (`python -m ceres_package.ml_logic.ml_model`). **Do not import** — not registered. |

**Intended flow:** GCS sources → (your FE / meteo work) → **mart parquet** → `train_and_register` → optional `promote_run` → `predict_production` → API.

**Catalog today:** `lag1`, `histgb_default`, `ridge_default`, `random_forest_default`.

**Gaps to mind:**

- No mart builder in-repo yet on `master` — team creates marts locally or in notebooks, then points `--mart` at the file.
- `ml_model.py` on `master` still runs an old XGB path if executed; it does not write to `models/registry/`.
- Parquet marts need `pyarrow` (in `requirements.txt`).
