# Ceres Predict API — team onboarding

One-shot PR: after merge, this is the full working flow.

## Setup (once)

```bash
pip install -r requirements.txt
gcloud auth application-default login   # only if model not cached locally
```

## Verify (offline, no server)

```bash
make -f ceres_package/interface/Makefile smoke
```

Expected: `PASS` with `actual_qha: 85.0` for dept `51` / year `2023`.

## Run API

```bash
make -f ceres_package/interface/Makefile run
# or: ./scripts/run_predict_api.sh
```

Open http://127.0.0.1:8000/docs

```bash
make -f ceres_package/interface/Makefile ping
make -f ceres_package/interface/Makefile predict
```

## What ships in this PR

| Path | Role |
|------|------|
| `ceres_package/interface/` | FastAPI app, predictor, feature mart, smoke test |
| `scripts/run_predict_api.sh` | Start uvicorn |
| `apps/predict/` | Streamlit placeholder (phase 2) |

Full API reference: `ceres_package/interface/README.md`

## Env vars

| Variable | Default |
|----------|---------|
| `PULL_FROM_GCP` | `false` (use `raw_data/` cache) |
| `CERES_GCS_BUCKET` | `ceres-ai-bucket` |
| `CERES_GCS_MODEL_BLOB` | `models/ml_model.pkl` |

Set `PULL_FROM_GCP=true` if raw CSVs are missing locally.
