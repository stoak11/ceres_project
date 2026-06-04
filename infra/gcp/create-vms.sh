#!/usr/bin/env bash
# Create single Ceres VM (ceres-dev-01) — ML, FE, and batch jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${GCP_PROJECT:-ceres-project-498208}"
ZONE="${GCP_ZONE:-europe-west1-b}"
SA_ID="${CERES_SA_ID:-ceres-vm}"
SA_EMAIL="${SA_ID}@${PROJECT}.iam.gserviceaccount.com"
NETWORK_TAG="${NETWORK_TAG:-ceres-worker}"
BOOTSTRAP="${SCRIPT_DIR}/startup/bootstrap.sh"
NAME="${CERES_VM_NAME:-ceres-dev-01}"
MT="${CERES_MACHINE_TYPE:-e2-highmem-8}"
DISK_GB="${CERES_DISK_GB:-500}"

"${SCRIPT_DIR}/setup-iam.sh"

if ! gcloud compute firewall-rules describe allow-iap-ssh-ceres --project="${PROJECT}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create allow-iap-ssh-ceres \
    --project="${PROJECT}" \
    --network=default \
    --direction=INGRESS \
    --action=allow \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --target-tags="${NETWORK_TAG}" \
    --description="SSH via IAP to Ceres VM"
fi

if gcloud compute instances describe "${NAME}" --project="${PROJECT}" --zone="${ZONE}" >/dev/null 2>&1; then
  echo "VM ${NAME} already exists. Run ./upgrade-ceres-vm.sh to resize."
  exit 0
fi

gcloud compute instances create "${NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MT}" \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --metadata-from-file="startup-script=${BOOTSTRAP}" \
  --tags="${NETWORK_TAG}" \
  --labels="role=ceres,project=ceres,env=prod"

echo "Created ${NAME} (${MT}, ${DISK_GB}GB)"
