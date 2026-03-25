#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-c}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100-80}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-~/workspace/Mineral_Gas_Locator}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd gcloud

gcloud compute ssh "$A100_VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc 'cd $REMOTE_REPO_ROOT && \
    DATA_DISK_DEVICE=$DATA_DISK_DEVICE \
    DATA_DISK_MOUNT_POINT=$DATA_DISK_MOUNT_POINT \
    FORMAT_IF_NEEDED=0 \
    bash earth_world_model/scripts/setup_training_data_disk.sh'"

echo "Mounted data disk on $A100_VM_NAME"
