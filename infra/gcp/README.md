# Ceres GCP — single VM

Project: **ceres-project-498208** · Zone: **europe-west1-b** · Bucket: **gs://ceres-ai-bucket**

Architecture: **[docs/GCP_VMS.md](../../docs/GCP_VMS.md)**

## ceres-dev-01 (team VM)

| Spec | Value |
|------|--------|
| Machine | **e2-highmem-8** — 8 vCPU, **64 GB** RAM (max E2 8-core high-memory) |
| Disk | **500 GB** `pd-balanced` boot |
| OS | Ubuntu 22.04 LTS |
| Role | ML registry, meteo FE, ingest, DL/ML training (CPU), Streamlit |

No GPU. One repo, one SSH target.

```bash
gcloud config set project ceres-project-498208
gcloud compute ssh ceres-dev-01 --zone=europe-west1-b --tunnel-through-iap
```

## Provision / upgrade

```bash
cd infra/gcp
chmod +x *.sh startup/*.sh
./create-vms.sh          # greenfield only
./upgrade-ceres-vm.sh    # resize existing (stop + disk + machine type)
```

After disk upgrade, on the VM once:

```bash
sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1
df -h /
```

## Deploy repo

```bash
gcloud compute scp --recurse ./ceres_project ceres-dev-01:/opt/ceres/ \
  --zone=europe-west1-b --tunnel-through-iap
```

On VM:

```bash
source /opt/ceres/ceres_project/infra/gcp/activate-ceres.sh
pip install -r requirements.txt
```

