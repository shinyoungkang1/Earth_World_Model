#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-ssl4eo-50k-data}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-ewm-training-data}"
START_VM="${START_VM:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

instance_exists() {
  gcloud compute instances describe "$A100_VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1
}

disk_attached() {
  gcloud compute instances describe "$A100_VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(disks[].source.basename())" | tr ';' '\n' | grep -Fx "$DATA_DISK_NAME" >/dev/null 2>&1
}

require_cmd gcloud

if ! instance_exists; then
  echo "A100 VM not found: $A100_VM_NAME" >&2
  exit 1
fi

if ! disk_attached; then
  gcloud compute instances attach-disk "$A100_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --disk="$DATA_DISK_NAME" \
    --device-name="$DATA_DISK_DEVICE_NAME"
fi

if [[ "$START_VM" == "1" ]]; then
  gcloud compute instances start "$A100_VM_NAME" --project="$PROJECT_ID" --zone="$ZONE"
fi

echo "A100 VM ready with attached data disk: $DATA_DISK_NAME"
