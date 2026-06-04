#!/usr/bin/env bash
# Change machine type on ceres-dev-01 (instance is stopped automatically).
# Default target: e2-highmem-8 (8 vCPU, 64 GB). For disk+type use upgrade-ceres-vm.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-ceres-project-498208}"
ZONE="${GCP_ZONE:-europe-west1-b}"
NAME="${1:-ceres-dev-01}"
MT="${2:-e2-highmem-8}"

gcloud compute instances stop "${NAME}" --project="${PROJECT}" --zone="${ZONE}"
gcloud compute instances set-machine-type "${NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MT}"
gcloud compute instances start "${NAME}" --project="${PROJECT}" --zone="${ZONE}"
gcloud compute instances describe "${NAME}" --project="${PROJECT}" --zone="${ZONE}" \
  --format='table(name,machineType.basename(),status)'
