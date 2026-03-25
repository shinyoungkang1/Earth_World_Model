#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
SOURCE_DISK_NAME="${SOURCE_DISK_NAME:?Set SOURCE_DISK_NAME}"
SOURCE_ZONE="${SOURCE_ZONE:?Set SOURCE_ZONE}"
TARGET_DISK_NAME="${TARGET_DISK_NAME:?Set TARGET_DISK_NAME}"
TARGET_ZONE="${TARGET_ZONE:?Set TARGET_ZONE}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-${SOURCE_DISK_NAME}-snap-$(date +%Y%m%d%H%M%S)}"
DISK_TYPE="${DISK_TYPE:-pd-ssd}"
DISK_SIZE_GB="${DISK_SIZE_GB:-}"
CREATE_SNAPSHOT="${CREATE_SNAPSHOT:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd gcloud

if [[ "$CREATE_SNAPSHOT" == "1" ]]; then
  gcloud compute disks snapshot "$SOURCE_DISK_NAME" \
    --project="$PROJECT_ID" \
    --zone="$SOURCE_ZONE" \
    --snapshot-names="$SNAPSHOT_NAME"
fi

create_args=(
  gcloud compute disks create "$TARGET_DISK_NAME"
  --project="$PROJECT_ID"
  --zone="$TARGET_ZONE"
  --type="$DISK_TYPE"
  --source-snapshot="$SNAPSHOT_NAME"
)

if [[ -n "$DISK_SIZE_GB" ]]; then
  create_args+=(--size="${DISK_SIZE_GB}")
fi

"${create_args[@]}"

echo "Cloned $SOURCE_DISK_NAME ($SOURCE_ZONE) -> $TARGET_DISK_NAME ($TARGET_ZONE) via snapshot $SNAPSHOT_NAME"
