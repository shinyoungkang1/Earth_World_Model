#!/usr/bin/env bash
set -euo pipefail

DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
FORMAT_IF_NEEDED="${FORMAT_IF_NEEDED:-0}"
FILESYSTEM_TYPE="${FILESYSTEM_TYPE:-ext4}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd sudo

if [[ ! -b "$DATA_DISK_DEVICE" ]]; then
  echo "Data disk device not found: $DATA_DISK_DEVICE" >&2
  exit 1
fi

CANONICAL_DATA_DISK_DEVICE="$(readlink -f "$DATA_DISK_DEVICE")"

sudo mkdir -p "$DATA_DISK_MOUNT_POINT"

if mountpoint -q "$DATA_DISK_MOUNT_POINT"; then
  MOUNTED_SOURCE="$(findmnt -n -o SOURCE --target "$DATA_DISK_MOUNT_POINT" || true)"
  CANONICAL_MOUNTED_SOURCE="$(readlink -f "$MOUNTED_SOURCE" 2>/dev/null || printf '%s' "$MOUNTED_SOURCE")"
  if [[ "$CANONICAL_MOUNTED_SOURCE" != "$CANONICAL_DATA_DISK_DEVICE" ]]; then
    echo "Mount point $DATA_DISK_MOUNT_POINT is already using $MOUNTED_SOURCE, expected $DATA_DISK_DEVICE" >&2
    exit 1
  fi
  sudo chown "$(id -u)":"$(id -g)" "$DATA_DISK_MOUNT_POINT"
  echo "Data disk already mounted at $DATA_DISK_MOUNT_POINT"
  exit 0
fi

if ! sudo blkid "$DATA_DISK_DEVICE" >/dev/null 2>&1; then
  if [[ "$FORMAT_IF_NEEDED" != "1" ]]; then
    echo "Disk $DATA_DISK_DEVICE is unformatted. Re-run with FORMAT_IF_NEEDED=1 to create $FILESYSTEM_TYPE." >&2
    exit 1
  fi
  sudo mkfs."$FILESYSTEM_TYPE" -F "$DATA_DISK_DEVICE"
fi

sudo mount "$DATA_DISK_DEVICE" "$DATA_DISK_MOUNT_POINT"
sudo chown "$(id -u)":"$(id -g)" "$DATA_DISK_MOUNT_POINT"

echo "Mounted $DATA_DISK_DEVICE at $DATA_DISK_MOUNT_POINT"
