#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
PREP_VM_NAME="${PREP_VM_NAME:-ewm-ssl4eo-prep-cpu}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-~/workspace/Mineral_Gas_Locator}"
GCS_DATA_URI="${GCS_DATA_URI:?Set GCS_DATA_URI, for example gs://.../earth_world_model/data/raw/ssl4eo_zarr_50k}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATASET_SUBDIR="${DATASET_SUBDIR:-ssl4eo_zarr_50k}"
FORMAT_IF_NEEDED="${FORMAT_IF_NEEDED:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd gcloud

gcloud compute ssh "$PREP_VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc 'cd $REMOTE_REPO_ROOT && \
    GCS_DATA_URI=$GCS_DATA_URI \
    DATA_DISK_DEVICE=$DATA_DISK_DEVICE \
    DATA_DISK_MOUNT_POINT=$DATA_DISK_MOUNT_POINT \
    DATASET_SUBDIR=$DATASET_SUBDIR \
    FORMAT_IF_NEEDED=$FORMAT_IF_NEEDED \
    bash earth_world_model/scripts/stage_ssl4eo_to_data_disk.sh'"

echo "Staging command finished on $PREP_VM_NAME"
