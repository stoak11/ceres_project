#!/usr/bin/env bash
# Source from an SSH session:  source /opt/ceres/ceres_project/infra/gcp/activate-ceres.sh
# Or after clone:               source infra/gcp/activate-ceres.sh

# Prefer apt gcloud over slow Ubuntu snap (/snap/bin/gcloud)
if [[ -x /usr/bin/gcloud ]]; then
  export PATH="/usr/bin:${PATH//:\/snap\/bin/}"
fi
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ceres-project-498208}"
export CLOUDSDK_CORE_PROJECT="${CLOUDSDK_CORE_PROJECT:-$GOOGLE_CLOUD_PROJECT}"

if [[ -f /etc/ceres/env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/ceres/env
  set +a
  export GOOGLE_CLOUD_PROJECT="${GCP_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
  export CLOUDSDK_CORE_PROJECT="${GCP_PROJECT:-$CLOUDSDK_CORE_PROJECT}"
else
  echo "Missing /etc/ceres/env — run bootstrap or copy env.ceres.example" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${CERES_HOME}/.venv" ]]; then
  echo "Missing ${CERES_HOME}/.venv — bootstrap may not have finished" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${CERES_HOME}/.venv/bin/activate"

REPO="${CERES_HOME}/ceres_project"
if [[ -d "${REPO}" ]]; then
  export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
  cd "${REPO}" || return 1 2>/dev/null || exit 1
else
  echo "Repo not at ${REPO} — sync from laptop (see infra/gcp/README.md)" >&2
fi

echo "Ceres env active: project=${GCP_PROJECT} zone=${GCP_ZONE} python=$(which python)"
