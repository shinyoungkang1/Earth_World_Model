#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
PREP_VM_NAME="${PREP_VM_NAME:-ewm-ssl4eo-prep-cpu}"
PREP_MACHINE_TYPE="${PREP_MACHINE_TYPE:-e2-standard-8}"
PREP_BOOT_DISK_SIZE_GB="${PREP_BOOT_DISK_SIZE_GB:-50}"
PREP_BOOT_DISK_TYPE="${PREP_BOOT_DISK_TYPE:-pd-balanced}"
PREP_IMAGE_FAMILY="${PREP_IMAGE_FAMILY:-ubuntu-2204-lts}"
PREP_IMAGE_PROJECT="${PREP_IMAGE_PROJECT:-ubuntu-os-cloud}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-ssl4eo-50k-data}"
DATA_DISK_SIZE_GB="${DATA_DISK_SIZE_GB:-500}"
DATA_DISK_TYPE="${DATA_DISK_TYPE:-pd-ssd}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-ewm-training-data}"
SCOPES="${SCOPES:-https://www.googleapis.com/auth/cloud-platform}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

instance_exists() {
  gcloud compute instances describe "$PREP_VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1
}

disk_exists() {
  gcloud compute disks describe "$DATA_DISK_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1
}

disk_attached() {
  gcloud compute instances describe "$PREP_VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(disks[].source.basename())" | tr ';' '\n' | grep -Fx "$DATA_DISK_NAME" >/dev/null 2>&1
}

require_cmd gcloud

if ! disk_exists; then
  gcloud compute disks create "$DATA_DISK_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --size="${DATA_DISK_SIZE_GB}GB" \
    --type="$DATA_DISK_TYPE"
fi

if ! instance_exists; then
  gcloud compute instances create "$PREP_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$PREP_MACHINE_TYPE" \
    --boot-disk-size="${PREP_BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type="$PREP_BOOT_DISK_TYPE" \
    --image-family="$PREP_IMAGE_FAMILY" \
    --image-project="$PREP_IMAGE_PROJECT" \
    --scopes="$SCOPES"
fi

if ! disk_attached; then
  gcloud compute instances attach-disk "$PREP_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --disk="$DATA_DISK_NAME" \
    --device-name="$DATA_DISK_DEVICE_NAME"
fi

echo "Prep VM ready: $PREP_VM_NAME"
echo "Data disk attached: $DATA_DISK_NAME"
