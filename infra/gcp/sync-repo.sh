#!/usr/bin/env bash
# Run on a VM after SSH: clone or update Ceres code and install Python deps.
set -euo pipefail

CERES_HOME="${CERES_HOME:-/opt/ceres}"
REPO_URL="${REPO_URL:-}"  # e.g. git@github.com:stoak11/ceres_project.git
BRANCH="${BRANCH:-main}"
TARGET="${CERES_HOME}/ceres_project"

if [[ -z "${REPO_URL}" ]]; then
  echo "Set REPO_URL to your git remote, then re-run." >&2
  echo "  REPO_URL=git@github.com:ORG/ceres_project.git $0" >&2
  exit 1
fi

if [[ -d "${TARGET}/.git" ]]; then
  git -C "${TARGET}" fetch --all
  git -C "${TARGET}" checkout "${BRANCH}"
  git -C "${TARGET}" pull --ff-only
else
  git clone --branch "${BRANCH}" "${REPO_URL}" "${TARGET}"
fi

# shellcheck disable=SC1091
source "${CERES_HOME}/.venv/bin/activate"
cd "${TARGET}"
pip install -r requirements.txt

echo "Installed. Example:"
echo "  cd ${TARGET} && PYTHONPATH=. python scripts/ml/train_and_register.py --help"
