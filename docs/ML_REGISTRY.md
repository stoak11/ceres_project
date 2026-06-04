# ML registry — maintainer reference

See **`ML_REGISTRY_ONE_PAGER.md`** for the team workflow.

## Modules

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

## GCS

`save_run` never uploads. After training:

```python
from ceres_package.ml_logic.registry import upload_run_to_gcs
upload_run_to_gcs(run_id)
```

Requires `BUCKET_NAME` and `GCP_PROJECT`.

## Legacy

`ml_pipeline.py` — grid search for notebooks only; not the registry path.
