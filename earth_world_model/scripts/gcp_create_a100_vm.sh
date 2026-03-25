#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
VM_NAME="${VM_NAME:-ewm-phase2-a100-80}"
PREFERRED_ZONES="${PREFERRED_ZONES:-us-central1-c us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-1g}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
SCOPES="${SCOPES:-https://www.googleapis.com/auth/cloud-platform}"
METADATA="${METADATA:-install-nvidia-driver=True}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

instance_exists() {
  local zone="$1"
  gcloud compute instances describe "$VM_NAME" --project="$PROJECT_ID" --zone="$zone" >/dev/null 2>&1
}

require_cmd gcloud

for zone in $PREFERRED_ZONES; do
  if instance_exists "$zone"; then
    echo "Instance already exists: $VM_NAME ($zone)"
    echo "ZONE=$zone"
    exit 0
  fi

  log_file="$(mktemp)"
  if gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --machine-type="$MACHINE_TYPE" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --scopes="$SCOPES" \
    --metadata="$METADATA" >"$log_file" 2>&1; then
    cat "$log_file"
    rm -f "$log_file"
    echo "ZONE=$zone"
    exit 0
  fi

  cat "$log_file" >&2
  rm -f "$log_file"
  echo "Create failed in $zone, trying next preferred zone..." >&2
done

echo "Failed to create $VM_NAME in preferred zones: $PREFERRED_ZONES" >&2
exit 1
