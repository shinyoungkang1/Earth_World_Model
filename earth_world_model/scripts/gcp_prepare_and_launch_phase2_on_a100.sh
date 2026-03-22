#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
PREP_VM_NAME="${PREP_VM_NAME:-ewm-ssl4eo-prep-cpu}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-ssl4eo-50k-data}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${GCS_DATA_URI##*/}}"
GCS_DATA_URI="${GCS_DATA_URI:?Set GCS_DATA_URI, for example gs://.../earth_world_model/data/raw/ssl4eo_zarr_50k}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://.../earth_world_model/runs/phase2_50k_a100_control_localdisk}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/ewm-data-disk/$DATASET_SUBDIR}"
CONFIG_PATH="${CONFIG_PATH:-~/workspace/Mineral_Gas_Locator/earth_world_model/configs/ssl4eo_zarr_trainval_50k_cuda_gpu_vm.yaml}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
INSTALL_DATAFLUX="${INSTALL_DATAFLUX:-0}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd bash

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  PREP_VM_NAME="$PREP_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_create_ssl4eo_prep_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$PREP_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_sync_repo_bundle_to_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  PREP_VM_NAME="$PREP_VM_NAME" \
  GCS_DATA_URI="$GCS_DATA_URI" \
  DATASET_SUBDIR="$DATASET_SUBDIR" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_stage_ssl4eo_to_prep_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$PREP_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_detach_ssl4eo_data_disk.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_attach_ssl4eo_data_disk_to_a100.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$A100_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_sync_repo_bundle_to_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_mount_training_data_disk_on_a100.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  GCS_RUN_URI="$GCS_RUN_URI" \
  LOCAL_DATA_ROOT="$LOCAL_DATA_ROOT" \
  CONFIG_PATH="$CONFIG_PATH" \
  SKIP_SETUP_GPU="$SKIP_SETUP_GPU" \
  INSTALL_DATAFLUX="$INSTALL_DATAFLUX" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_launch_phase2_localdisk_on_a100.sh"

echo "Phase 2 prep-and-launch flow completed for $A100_VM_NAME"
