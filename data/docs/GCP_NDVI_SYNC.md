# GCP sync — NDVI folder

**Project:** `ceres-project-498208`  
**Bucket:** `gs://ceres-ai-bucket`  
**Prefix:** `NDVI/` (replaces `raw_data/modis_ndvi/`)

**Local mirror (optional):** `data/NDVI/`

| File | Role |
|------|------|
| `README.md` | Overview |
| `COLUMN_REFERENCE.md` | Column glossary (`DEPT_ID`) |
| `ndvi_monthly_by_department_polygon.csv` | Monthly / dept (8 columns; no anchor/std) |
| `ndvi_season_features.csv` | Season features |
| `ML_BENCHMARK_SYNTHESIS.md` | ML benchmark reference (LOGO, top pipelines) |

---

## 1. Remove old layout on GCS

Delete the previous `raw_data` tree (polygon + duplicates + optional anchor):

```bash
export GCP_PROJECT=ceres-project-498208

gcloud storage rm -r gs://ceres-ai-bucket/raw_data/ \
  --project="${GCP_PROJECT}"
```

Confirm nothing else important lived only under `raw_data/` before running.

---

## 2. Push new `NDVI/` folder

From repo root:

```bash
export GCP_PROJECT=ceres-project-498208
export LOCAL=data/NDVI
export GCS_PREFIX=gs://ceres-ai-bucket/NDVI

# Upload each object (avoids accidental NDVI/NDVI/ nesting)
for f in COLUMN_REFERENCE.md README.md ML_BENCHMARK_SYNTHESIS.md \
  ndvi_monthly_by_department_polygon.csv ndvi_season_features.csv; do
  gcloud storage cp "${LOCAL}/${f}" "${GCS_PREFIX}/${f}" --project="${GCP_PROJECT}"
done
```

Verify:

```bash
gcloud storage ls "${GCS_PREFIX}/" --project="${GCP_PROJECT}"
gcloud storage cat "${GCS_PREFIX}/ndvi_monthly_by_department_polygon.csv" \
  --project="${GCP_PROJECT}" | head -1
# DEPT_ID,department_name,year,month,...
```

---

## 3. Regenerate CSVs locally

```bash
python scripts/ndvi/export_polygon_dept_id.py
```
