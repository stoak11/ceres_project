#!/usr/bin/env bash
# Upgrade ceres-dev-01 to project single-VM spec: e2-highmem-8 + 500GB disk.
set -euo pipefail

PROJECT="${GCP_PROJECT:-ceres-project-498208}"
ZONE="${GCP_ZONE:-europe-west1-b}"
NAME="${CERES_VM_NAME:-ceres-dev-01}"
MT="${CERES_MACHINE_TYPE:-e2-highmem-8}"
DISK_GB="${CERES_DISK_GB:-500}"

echo "Target: ${NAME} -> ${MT}, disk ${DISK_GB}GB (${ZONE})"
gcloud compute instances stop "${NAME}" --project="${PROJECT}" --zone="${ZONE}"
gcloud compute disks resize "${NAME}" --project="${PROJECT}" --zone="${ZONE}" --size="${DISK_GB}GB" --quiet
gcloud compute instances set-machine-type "${NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --machine-type="${MT}"
gcloud compute instances start "${NAME}" --project="${PROJECT}" --zone="${ZONE}"

gcloud compute instances describe "${NAME}" --project="${PROJECT}" --zone="${ZONE}" \
  --format='table(name,machineType.basename(),status)'

echo ""
echo "After SSH, grow root FS if needed:"
echo "  sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1"
echo "  df -h /"
