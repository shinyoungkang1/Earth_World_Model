#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
VM_NAME="${VM_NAME:?Set VM_NAME, for example ewm-ssl4eo-prep-cpu}"
REMOTE_WORKSPACE_ROOT="${REMOTE_WORKSPACE_ROOT:-~/workspace}"
REMOTE_BUNDLE_PATH="${REMOTE_BUNDLE_PATH:-~/earth_world_model_bundle.tar.gz}"
LOCAL_BUNDLE_PATH="${LOCAL_BUNDLE_PATH:-/tmp/earth_world_model_bundle.tar.gz}"
REPO_ROOT="${REPO_ROOT:-/home/shin/Mineral_Gas_Locator}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tar
require_cmd gcloud

tar -C "$(dirname "$REPO_ROOT")" -czf "$LOCAL_BUNDLE_PATH" \
  --exclude='Mineral_Gas_Locator/.git' \
  --exclude='Mineral_Gas_Locator/data' \
  --exclude='Mineral_Gas_Locator/checkpoints' \
  --exclude='Mineral_Gas_Locator/earth_world_model/checkpoints' \
  --exclude='Mineral_Gas_Locator/results' \
  --exclude='Mineral_Gas_Locator/models' \
  --exclude='Mineral_Gas_Locator/.venv' \
  "$(basename "$REPO_ROOT")"

gcloud compute scp "$LOCAL_BUNDLE_PATH" "$VM_NAME:$REMOTE_BUNDLE_PATH" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"

gcloud compute ssh "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="mkdir -p $REMOTE_WORKSPACE_ROOT && tar -xzf $REMOTE_BUNDLE_PATH -C $REMOTE_WORKSPACE_ROOT"

echo "Repo synced to $VM_NAME:$REMOTE_WORKSPACE_ROOT"
