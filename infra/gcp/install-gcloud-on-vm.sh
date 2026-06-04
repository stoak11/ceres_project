#!/usr/bin/env bash
# Run on ceres-dev-01 (once) if /usr/bin/gcloud is missing.
set -euo pipefail

if [[ -x /usr/bin/gcloud ]]; then
  /usr/bin/gcloud version
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
sudo apt-get update -qq
sudo apt-get install -y -qq apt-transport-https ca-certificates gnupg curl

if [[ ! -f /usr/share/keyrings/cloud.google.gpg ]]; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
fi

sudo apt-get update -qq
sudo NEEDRESTART_MODE=a apt-get install -y -qq google-cloud-cli
/usr/bin/gcloud version
