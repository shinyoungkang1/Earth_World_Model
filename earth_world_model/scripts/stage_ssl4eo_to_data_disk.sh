#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCS_DATA_URI="${GCS_DATA_URI:?Set GCS_DATA_URI, for example gs://.../earth_world_model/data/raw/ssl4eo_zarr_50k}"
DATASET_SUBDIR="${DATASET_SUBDIR:-ssl4eo_zarr_50k}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
FORMAT_IF_NEEDED="${FORMAT_IF_NEEDED:-0}"
TARGET_DIR="${TARGET_DIR:-$DATA_DISK_MOUNT_POINT/$DATASET_SUBDIR}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

storage_rsync() {
  if gcloud storage rsync --help >/dev/null 2>&1; then
    gcloud storage rsync --recursive "$1" "$2"
    return 0
  fi
  if gcloud storage cp --help >/dev/null 2>&1; then
    gcloud storage cp --recursive "$1" "$2"
    return 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$1" "$2"
    return 0
  fi
  echo "Missing directory sync support: need 'gcloud storage rsync', 'gcloud storage cp --recursive', or 'gsutil'" >&2
  exit 1
}

require_cmd gcloud

env \
  DATA_DISK_DEVICE="$DATA_DISK_DEVICE" \
  DATA_DISK_MOUNT_POINT="$DATA_DISK_MOUNT_POINT" \
  FORMAT_IF_NEEDED="$FORMAT_IF_NEEDED" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_training_data_disk.sh"

mkdir -p "$TARGET_DIR"
storage_rsync "$GCS_DATA_URI" "$TARGET_DIR"

echo "Staged $GCS_DATA_URI to $TARGET_DIR"
