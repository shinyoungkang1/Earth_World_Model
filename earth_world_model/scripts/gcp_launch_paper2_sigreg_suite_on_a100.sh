#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-c}"
PREFERRED_ZONES="${PREFERRED_ZONES:-$ZONE}"
PREP_VM_NAME="${PREP_VM_NAME:-ewm-prelim1714-prep-cpu}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100-80}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/shin/workspace/Mineral_Gas_Locator}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-prelim1714-data}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-ewm-training-data}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-$DATA_DISK_DEVICE_NAME}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATA_ROOT_BASE="${DATA_ROOT_BASE:-/mnt/ewm-data-disk/ewm_prelim_1650_10000_v1}"
YEARLY_NAME="${YEARLY_NAME:-yearly_1714}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_10000}"
YEARLY_TRAIN_MAX_SAMPLES="${YEARLY_TRAIN_MAX_SAMPLES:-1650}"
YEARLY_VAL_MAX_SAMPLES="${YEARLY_VAL_MAX_SAMPLES:-64}"
SSL4EO_TRAIN_MAX_SAMPLES="${SSL4EO_TRAIN_MAX_SAMPLES:-10000}"
SSL4EO_VAL_MAX_SAMPLES="${SSL4EO_VAL_MAX_SAMPLES:-1000}"
GCS_RUNS_ROOT="${GCS_RUNS_ROOT:-gs://omois-earth-world-model-phase2-20260320-11728/earth_world_model/runs/paper2_sigreg_suite_1650_10000}"
RUN_CORE_ABLATIONS="${RUN_CORE_ABLATIONS:-1}"
RUN_TIME_ABLATIONS="${RUN_TIME_ABLATIONS:-1}"
RUN_ARCH_ABLATIONS="${RUN_ARCH_ABLATIONS:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_CREATE_A100="${SKIP_CREATE_A100:-0}"
SKIP_DETACH_PREP_VM="${SKIP_DETACH_PREP_VM:-0}"
SKIP_SYNC_REPO="${SKIP_SYNC_REPO:-0}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
RUN_LAUNCHER_LOG_PATH="${RUN_LAUNCHER_LOG_PATH:-/tmp/paper2_sigreg_suite_launcher.log}"

quote_for_bash() {
  printf '%q' "$1"
}

if [[ "$SKIP_CREATE_A100" != "1" ]]; then
  env \
    PROJECT_ID="$PROJECT_ID" \
    VM_NAME="$A100_VM_NAME" \
    PREFERRED_ZONES="$PREFERRED_ZONES" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_create_a100_vm.sh"
fi

if [[ "$SKIP_DETACH_PREP_VM" != "1" ]]; then
  env \
    PROJECT_ID="$PROJECT_ID" \
    ZONE="$ZONE" \
    VM_NAME="$PREP_VM_NAME" \
    DATA_DISK_NAME="$DATA_DISK_NAME" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_detach_ssl4eo_data_disk.sh"
fi

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  DATA_DISK_DEVICE_NAME="$DATA_DISK_DEVICE_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_attach_ssl4eo_data_disk_to_a100.sh"

if [[ "$SKIP_SYNC_REPO" != "1" ]]; then
  env \
    PROJECT_ID="$PROJECT_ID" \
    ZONE="$ZONE" \
    VM_NAME="$A100_VM_NAME" \
    REPO_ROOT="$PROJECT_ROOT" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_sync_repo_bundle_to_vm.sh"
fi

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  REMOTE_REPO_ROOT="$REMOTE_REPO_ROOT" \
  DATA_DISK_DEVICE="$DATA_DISK_DEVICE" \
  DATA_DISK_MOUNT_POINT="$DATA_DISK_MOUNT_POINT" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_mount_training_data_disk_on_a100.sh"

gcloud compute ssh "$A100_VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc 'cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
    nohup env \
      DATA_ROOT_BASE=$(quote_for_bash "$DATA_ROOT_BASE") \
      GCS_RUNS_ROOT=$(quote_for_bash "$GCS_RUNS_ROOT") \
      YEARLY_NAME=$(quote_for_bash "$YEARLY_NAME") \
      SSL4EO_NAME=$(quote_for_bash "$SSL4EO_NAME") \
      YEARLY_TRAIN_MAX_SAMPLES=$(quote_for_bash "$YEARLY_TRAIN_MAX_SAMPLES") \
      YEARLY_VAL_MAX_SAMPLES=$(quote_for_bash "$YEARLY_VAL_MAX_SAMPLES") \
      SSL4EO_TRAIN_MAX_SAMPLES=$(quote_for_bash "$SSL4EO_TRAIN_MAX_SAMPLES") \
      SSL4EO_VAL_MAX_SAMPLES=$(quote_for_bash "$SSL4EO_VAL_MAX_SAMPLES") \
      RUN_CORE_ABLATIONS=$(quote_for_bash "$RUN_CORE_ABLATIONS") \
      RUN_TIME_ABLATIONS=$(quote_for_bash "$RUN_TIME_ABLATIONS") \
      RUN_ARCH_ABLATIONS=$(quote_for_bash "$RUN_ARCH_ABLATIONS") \
      FORCE_RERUN=$(quote_for_bash "$FORCE_RERUN") \
      SKIP_SETUP_GPU=$(quote_for_bash "$SKIP_SETUP_GPU") \
      bash earth_world_model/scripts/run_paper2_unified_suite_gpu_vm.sh \
      > $(quote_for_bash "$RUN_LAUNCHER_LOG_PATH") 2>&1 < /dev/null &'"

echo "Launched Paper 2 SIGReg suite on $A100_VM_NAME"
echo "A100_VM_NAME=$A100_VM_NAME"
echo "ZONE=$ZONE"
echo "DATA_ROOT_BASE=$DATA_ROOT_BASE"
echo "GCS_RUNS_ROOT=$GCS_RUNS_ROOT"
echo "RUN_LAUNCHER_LOG_PATH=$RUN_LAUNCHER_LOG_PATH"
