#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-c}"
PREFERRED_ZONES="${PREFERRED_ZONES:-$ZONE}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100-80}"
REMOTE_WORKSPACE_ROOT="${REMOTE_WORKSPACE_ROOT:-/home/shin/workspace}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-$REMOTE_WORKSPACE_ROOT/EO_WM}"
REPO_URL="${REPO_URL:-https://github.com/shinyoungkang1/Earth_World_Model.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-yearly10k-ssl4eo50k-data}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-ewm-training-data}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-$DATA_DISK_DEVICE_NAME}"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATA_ROOT_BASE="${DATA_ROOT_BASE:-/mnt/ewm-data-disk/ewm_prelim_yearly10k_ssl4eo50k_v1}"
YEARLY_NAME="${YEARLY_NAME:-yearly_10000}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_50000}"
YEARLY_TRAIN_MAX_SAMPLES="${YEARLY_TRAIN_MAX_SAMPLES:-10000}"
YEARLY_VAL_MAX_SAMPLES="${YEARLY_VAL_MAX_SAMPLES:-256}"
SSL4EO_TRAIN_MAX_SAMPLES="${SSL4EO_TRAIN_MAX_SAMPLES:-50000}"
SSL4EO_VAL_MAX_SAMPLES="${SSL4EO_VAL_MAX_SAMPLES:-1024}"
EWM_BATCH_SIZE="${EWM_BATCH_SIZE:-16}"
EWM_AUXILIARY_BATCH_SIZE="${EWM_AUXILIARY_BATCH_SIZE:-16}"
EWM_EVAL_BATCH_SIZE="${EWM_EVAL_BATCH_SIZE:-16}"
EWM_EPOCHS="${EWM_EPOCHS:-8}"
EWM_MIXED_STAGE_EPOCHS="${EWM_MIXED_STAGE_EPOCHS:-4}"
EWM_MIXED_STAGE_AUXILIARY_FRACTION="${EWM_MIXED_STAGE_AUXILIARY_FRACTION:-0.5}"
EWM_SUBCLIP_16_START="${EWM_SUBCLIP_16_START:-0}"
EWM_SUBCLIP_32_START="${EWM_SUBCLIP_32_START:-2}"
EWM_SUBCLIP_52_START="${EWM_SUBCLIP_52_START:-4}"
EWM_TUBELET_SIZE="${EWM_TUBELET_SIZE:-1}"
EWM_EVAL_CHECKPOINT_METRIC="${EWM_EVAL_CHECKPOINT_METRIC:-mean_masked_loss}"
EWM_EVAL_CHECKPOINT_MIN_EPOCH="${EWM_EVAL_CHECKPOINT_MIN_EPOCH:-5}"
GCS_RUNS_ROOT="${GCS_RUNS_ROOT:-gs://omois-earth-world-model-phase2-20260320-11728/earth_world_model/runs/paper2_factorized_world_model_selected_10000_50000_e8_b16_$(date +%Y%m%d_%H%M%S)}"
RUN_ID_FILTER="${RUN_ID_FILTER:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_CREATE_A100="${SKIP_CREATE_A100:-0}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-0}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
RUN_LAUNCHER_LOG_PATH="${RUN_LAUNCHER_LOG_PATH:-/tmp/paper2_factorized_world_model_launcher.log}"

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

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  DATA_DISK_DEVICE_NAME="$DATA_DISK_DEVICE_NAME" \
  START_VM=1 \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_attach_ssl4eo_data_disk_to_a100.sh"

if [[ "$SKIP_GIT_SYNC" != "1" ]]; then
  gcloud compute ssh "$A100_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="bash -lc '
      mkdir -p $(quote_for_bash "$REMOTE_WORKSPACE_ROOT") && \
      if [[ -d $(quote_for_bash "$REMOTE_REPO_ROOT")/.git ]]; then
        cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
        git remote set-url origin $(quote_for_bash "$REPO_URL") && \
        git fetch origin $(quote_for_bash "$REPO_BRANCH") --prune && \
        git checkout $(quote_for_bash "$REPO_BRANCH") && \
        git pull --ff-only origin $(quote_for_bash "$REPO_BRANCH"); \
      elif [[ ! -e $(quote_for_bash "$REMOTE_REPO_ROOT") ]]; then
        git clone --branch $(quote_for_bash "$REPO_BRANCH") $(quote_for_bash "$REPO_URL") $(quote_for_bash "$REMOTE_REPO_ROOT"); \
      else
        echo Remote path exists and is not a git repo: $(quote_for_bash "$REMOTE_REPO_ROOT") >&2; \
        exit 1; \
      fi'"
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
      EWM_BATCH_SIZE=$(quote_for_bash "$EWM_BATCH_SIZE") \
      EWM_AUXILIARY_BATCH_SIZE=$(quote_for_bash "$EWM_AUXILIARY_BATCH_SIZE") \
      EWM_EVAL_BATCH_SIZE=$(quote_for_bash "$EWM_EVAL_BATCH_SIZE") \
      EWM_EPOCHS=$(quote_for_bash "$EWM_EPOCHS") \
      EWM_MIXED_STAGE_EPOCHS=$(quote_for_bash "$EWM_MIXED_STAGE_EPOCHS") \
      EWM_MIXED_STAGE_AUXILIARY_FRACTION=$(quote_for_bash "$EWM_MIXED_STAGE_AUXILIARY_FRACTION") \
      EWM_SUBCLIP_16_START=$(quote_for_bash "$EWM_SUBCLIP_16_START") \
      EWM_SUBCLIP_32_START=$(quote_for_bash "$EWM_SUBCLIP_32_START") \
      EWM_SUBCLIP_52_START=$(quote_for_bash "$EWM_SUBCLIP_52_START") \
      EWM_TUBELET_SIZE=$(quote_for_bash "$EWM_TUBELET_SIZE") \
      EWM_EVAL_CHECKPOINT_METRIC=$(quote_for_bash "$EWM_EVAL_CHECKPOINT_METRIC") \
      EWM_EVAL_CHECKPOINT_MIN_EPOCH=$(quote_for_bash "$EWM_EVAL_CHECKPOINT_MIN_EPOCH") \
      RUN_ID_FILTER=$(quote_for_bash "$RUN_ID_FILTER") \
      FORCE_RERUN=$(quote_for_bash "$FORCE_RERUN") \
      SKIP_SETUP_GPU=$(quote_for_bash "$SKIP_SETUP_GPU") \
      bash earth_world_model/scripts/run_paper2_factorized_world_model_selected_yearly10k_ssl4eo50k_gpu_vm.sh \
      > $(quote_for_bash "$RUN_LAUNCHER_LOG_PATH") 2>&1 < /dev/null &'"

echo "Launched Paper 2 factorized world-model suite on $A100_VM_NAME"
echo "A100_VM_NAME=$A100_VM_NAME"
echo "ZONE=$ZONE"
echo "REMOTE_REPO_ROOT=$REMOTE_REPO_ROOT"
echo "DATA_ROOT_BASE=$DATA_ROOT_BASE"
echo "GCS_RUNS_ROOT=$GCS_RUNS_ROOT"
echo "RUN_LAUNCHER_LOG_PATH=$RUN_LAUNCHER_LOG_PATH"
