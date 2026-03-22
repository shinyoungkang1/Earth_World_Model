#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
VM_NAME="${VM_NAME:-ewm-ssl4eo-prep-cpu}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-ssl4eo-50k-data}"
STOP_VM_FIRST="${STOP_VM_FIRST:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

instance_exists() {
  gcloud compute instances describe "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1
}

disk_attached() {
  gcloud compute instances describe "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
    --format="value(disks[].source.basename())" | tr ';' '\n' | grep -Fx "$DATA_DISK_NAME" >/dev/null 2>&1
}

require_cmd gcloud

if ! instance_exists; then
  echo "VM not found: $VM_NAME" >&2
  exit 1
fi

if [[ "$STOP_VM_FIRST" == "1" ]]; then
  gcloud compute instances stop "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" || true
fi

if disk_attached; then
  gcloud compute instances detach-disk "$VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --disk="$DATA_DISK_NAME"
fi

echo "Detached $DATA_DISK_NAME from $VM_NAME"
