# Ceres Predict API

Team package: **dept + year → `model.predict()` + observed yield**.

Model: `gs://ceres-ai-bucket/models/ml_model.pkl`

---

## Quick start (team)

```bash
# 1. Setup (once)
pip install -r requirements.txt          # repo root
gcloud auth application-default login

# 2. Verify offline (no server)
make -f ceres_package/interface/Makefile smoke

# 3. Start API
make -f ceres_package/interface/Makefile run
```

Open **http://127.0.0.1:8000/docs** or:

```bash
make -f ceres_package/interface/Makefile ping
make -f ceres_package/interface/Makefile predict
```

---

## Endpoints

| Route | Description |
|-------|-------------|
| `GET /ping` | `{"response":"pong"}` |
| `GET /health` | Model warmed up |
| `GET /predict?dept=51&year=2023` | Prediction + actual yield |

Example response:

```json
{
  "dept_id": "51",
  "year": 2023,
  "yield_unit": "q/ha",
  "predict_qha": 86.801,
  "actual_qha": 85.0,
  "model_source": "gcs",
  "model_blob": "models/ml_model.pkl",
  "split": {"test_size": 0.2, "random_state": 42}
}
```

Regression anchor: **dept `51`, year `2023` → `actual_qha` = 85.0**

---

## Package layout

```
ceres_package/interface/
  api/fast.py       HTTP routes
  predictor.py      Predictor class (model + mart + predict)
  mart.py           Feature matrix + MinMaxScaler (random_state=42)
  gcs.py            Download model from GCS
  config.py         Env defaults
  smoke_test.py     Offline PASS/FAIL check
  run.sh            Start uvicorn
  Makefile          run | smoke | ping | predict
  .env.example      Optional env vars
```

---

## Env vars

| Variable | Default |
|----------|---------|
| `PULL_FROM_GCP` | `false` (use `raw_data/` cache) |
| `CERES_GCS_BUCKET` | `ceres-ai-bucket` |
| `CERES_GCS_MODEL_BLOB` | `models/ml_model.pkl` |
| `CERES_API_HOST` | `127.0.0.1` |
| `CERES_API_PORT` | `8000` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GCS 403 | `gcloud auth application-default login` |
| Missing CSVs | `PULL_FROM_GCP=true make -f ceres_package/interface/Makefile run` |
| Slow first start | Normal — model + mart load at startup (`/health` = ready) |
| `No data for dept=…` | Try `51` / `2023` |
