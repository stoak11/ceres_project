#!/bin/bash
# Ceres VM first-boot setup (Ubuntu 22.04). Idempotent enough for re-runs.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
CERES_HOME="${CERES_HOME:-/opt/ceres}"
CERES_USER="${CERES_USER:-$(logname 2>/dev/null || echo ubuntu)}"

apt-get update -qq
apt-get install -y -qq \
  git curl ca-certificates \
  python3.10 python3.10-venv python3.10-dev python3-pip \
  gdal-bin libgdal-dev build-essential \
  jq unzip

install -d -m 0755 "${CERES_HOME}" /etc/ceres /var/log/ceres
chown -R "${CERES_USER}:${CERES_USER}" "${CERES_HOME}" || true

if [[ ! -f /etc/ceres/env ]]; then
  cat >/etc/ceres/env <<'EOF'
GCP_PROJECT=ceres-project-498208
GCP_REGION=europe-west1
GCP_ZONE=europe-west1-b
BUCKET_NAME=ceres-ai-bucket
CERES_HOME=/opt/ceres
LOCAL_REGISTRY_PATH=/opt/ceres/models/registry
EOF
fi

# Shell convenience for interactive SSH
if ! grep -q 'CERES env' /etc/profile.d/ceres.sh 2>/dev/null; then
  cat >/etc/profile.d/ceres.sh <<'EOF'
# Ceres GCP VM — avoid snap gcloud (slow); use apt CLI from bootstrap
[[ -x /usr/bin/gcloud ]] && PATH="/usr/bin:${PATH//:\/snap\/bin/}"
[ -f /etc/ceres/env ] && set -a && . /etc/ceres/env && set +a
export GOOGLE_CLOUD_PROJECT="${GCP_PROJECT:-ceres-project-498208}"
export CLOUDSDK_CORE_PROJECT="${GOOGLE_CLOUD_PROJECT}"
export PYTHONPATH="${CERES_HOME}/ceres_project:${PYTHONPATH:-}"
EOF
fi

VENV="${CERES_HOME}/.venv"
if [[ ! -d "${VENV}" ]]; then
  python3.10 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel

# Always install apt CLI; Ubuntu snap gcloud is slow and not at /usr/bin/gcloud.
if [[ ! -x /usr/bin/gcloud ]]; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  apt-get install -y -qq google-cloud-cli
fi

touch /var/log/ceres/bootstrap.done
echo "Ceres bootstrap finished at $(date -Is)" | tee -a /var/log/ceres/bootstrap.log
