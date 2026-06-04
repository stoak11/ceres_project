# GCP sync — NDVI folder

**Project:** `ceres-project-498208`  
**Bucket:** `gs://ceres-ai-bucket`  
**Prefix:** `NDVI/`

**Local mirror (optional):** `data/NDVI/`

| File | Role |
|------|------|
| `README.md` | Overview |
| `COLUMN_REFERENCE.md` | Column glossary (`DEPT_ID`) |
| `ndvi_monthly_by_department_polygon.csv` | Monthly / dept |
| `ndvi_season_features.csv` | Season features |
| `ML_BENCHMARK_SYNTHESIS.md` | ML benchmark reference |

---

## Push `NDVI/` to GCS

From repo root:

```bash
export GCP_PROJECT=ceres-project-498208
export LOCAL=data/NDVI
export GCS_PREFIX=gs://ceres-ai-bucket/NDVI

for f in COLUMN_REFERENCE.md README.md ML_BENCHMARK_SYNTHESIS.md \
  ndvi_monthly_by_department_polygon.csv ndvi_season_features.csv; do
  gcloud storage cp "${LOCAL}/${f}" "${GCS_PREFIX}/${f}" --project="${GCP_PROJECT}"
done
```

---

## Regenerate CSVs locally

```bash
pip install -r requirements.txt
python scripts/ndvi/export_products.py
```

See **`NDVI_MODULE.md`**.
