#!/usr/bin/env bash
# Grant Ceres VMs read/write on the data bucket (idempotent).
set -euo pipefail

PROJECT="${GCP_PROJECT:-ceres-project-498208}"
BUCKET="${BUCKET_NAME:-ceres-ai-bucket}"
SA_ID="${CERES_SA_ID:-ceres-vm}"
SA_EMAIL="${SA_ID}@${PROJECT}.iam.gserviceaccount.com"

gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA_ID}" \
    --project="${PROJECT}" \
    --display-name="Ceres VM worker (GCS + default compute scopes)"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project="${PROJECT}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None 2>/dev/null \
  || gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --project="${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

echo "Service account ready: ${SA_EMAIL}"
echo "Use --service-account=${SA_EMAIL} when creating VMs."
