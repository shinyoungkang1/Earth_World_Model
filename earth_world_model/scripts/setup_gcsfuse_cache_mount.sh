#!/usr/bin/env bash
set -euo pipefail

BUCKET_NAME="${BUCKET_NAME:?Set BUCKET_NAME}"
ONLY_DIR="${ONLY_DIR:?Set ONLY_DIR}"
MOUNT_POINT="${MOUNT_POINT:?Set MOUNT_POINT}"
CACHE_DISK_MOUNT="${CACHE_DISK_MOUNT:?Set CACHE_DISK_MOUNT}"
CACHE_DIR="${CACHE_DIR:-$CACHE_DISK_MOUNT/gcsfuse-cache}"
CACHE_DISK_DEVICE="${CACHE_DISK_DEVICE:-}"
FILE_CACHE_MAX_SIZE_MB="${FILE_CACHE_MAX_SIZE_MB:-0}"
GCSFUSE_PROFILE="${GCSFUSE_PROFILE:-aiml-training}"
FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS="${FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS:-true}"
FILE_CACHE_MAX_PARALLEL_DOWNLOADS="${FILE_CACHE_MAX_PARALLEL_DOWNLOADS:-0}"
FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE="${FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE:-0}"
SEQUENTIAL_READ_SIZE_MB="${SEQUENTIAL_READ_SIZE_MB:-0}"
STAT_CACHE_MAX_SIZE_MB="${STAT_CACHE_MAX_SIZE_MB:-128}"
KERNEL_LIST_CACHE_TTL_SECS="${KERNEL_LIST_CACHE_TTL_SECS:-300}"
LOG_FILE="${LOG_FILE:-$CACHE_DISK_MOUNT/gcsfuse.log}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

install_gcsfuse_if_needed() {
  if command -v gcsfuse >/dev/null 2>&1; then
    return 0
  fi

  require_cmd curl
  require_cmd lsb_release
  sudo apt-get update
  sudo apt-get install -y curl lsb-release
  local repo="gcsfuse-$(lsb_release -c -s)"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt ${repo} main" \
    | sudo tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null
  sudo apt-get update
  sudo apt-get install -y gcsfuse
}

ensure_cache_disk_mounted() {
  sudo mkdir -p "$CACHE_DISK_MOUNT"
  if mountpoint -q "$CACHE_DISK_MOUNT"; then
    sudo chown "$(id -u)":"$(id -g)" "$CACHE_DISK_MOUNT"
    return 0
  fi
  if [[ -z "$CACHE_DISK_DEVICE" ]]; then
    sudo chown "$(id -u)":"$(id -g)" "$CACHE_DISK_MOUNT"
    return 0
  fi
  if ! sudo blkid "$CACHE_DISK_DEVICE" >/dev/null 2>&1; then
    sudo mkfs.ext4 -F "$CACHE_DISK_DEVICE"
  fi
  sudo mount "$CACHE_DISK_DEVICE" "$CACHE_DISK_MOUNT"
  sudo chown "$(id -u)":"$(id -g)" "$CACHE_DISK_MOUNT"
}

mount_bucket() {
  mkdir -p "$CACHE_DIR"
  if grep -qs " ${MOUNT_POINT} " /proc/mounts; then
    fusermount -u "$MOUNT_POINT" || sudo umount "$MOUNT_POINT"
  fi
  sudo mkdir -p "$MOUNT_POINT"
  sudo chown "$(id -u)":"$(id -g)" "$MOUNT_POINT"

  local args=(
    "--profile=${GCSFUSE_PROFILE}"
    "--implicit-dirs"
    "--only-dir=${ONLY_DIR}"
    "--cache-dir=${CACHE_DIR}"
    "--file-cache-cache-file-for-range-read=true"
    "--file-cache-enable-parallel-downloads=${FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS}"
    "--metadata-cache-negative-ttl-secs=0"
    "--metadata-cache-ttl-secs=-1"
    "--kernel-list-cache-ttl-secs=${KERNEL_LIST_CACHE_TTL_SECS}"
    "--stat-cache-max-size-mb=${STAT_CACHE_MAX_SIZE_MB}"
    "--log-file=${LOG_FILE}"
    "-o"
    "ro"
  )

  if [[ "$FILE_CACHE_MAX_SIZE_MB" != "0" ]]; then
    args+=("--file-cache-max-size-mb=${FILE_CACHE_MAX_SIZE_MB}")
  fi
  if [[ "$FILE_CACHE_MAX_PARALLEL_DOWNLOADS" != "0" ]]; then
    args+=("--file-cache-max-parallel-downloads=${FILE_CACHE_MAX_PARALLEL_DOWNLOADS}")
  fi
  if [[ "$FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE" != "0" ]]; then
    args+=("--file-cache-parallel-downloads-per-file=${FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE}")
  fi
  if [[ "$SEQUENTIAL_READ_SIZE_MB" != "0" ]]; then
    args+=("--sequential-read-size-mb=${SEQUENTIAL_READ_SIZE_MB}")
  fi

  gcsfuse "${args[@]}" "$BUCKET_NAME" "$MOUNT_POINT"
}

require_cmd sudo
install_gcsfuse_if_needed
ensure_cache_disk_mounted
mount_bucket

echo "Mounted gs://${BUCKET_NAME}/${ONLY_DIR} at ${MOUNT_POINT}"
echo "Cache dir: ${CACHE_DIR}"
echo "Log file: ${LOG_FILE}"
