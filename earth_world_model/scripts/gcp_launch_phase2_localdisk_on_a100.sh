#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-~/workspace/Mineral_Gas_Locator}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://.../earth_world_model/runs/phase2_50k_a100_control_localdisk}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-/tmp/ewm_phase2_50k_gpu_control_localdisk}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/ewm-data-disk/ssl4eo_zarr_50k}"
CONFIG_PATH="${CONFIG_PATH:-$REMOTE_REPO_ROOT/earth_world_model/configs/ssl4eo_zarr_trainval_50k_cuda_gpu_vm.yaml}"
RUN_LOG_PATH="${RUN_LOG_PATH:-$LOCAL_RUN_ROOT/train.log}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$LOCAL_RUN_ROOT/checkpoints/ssl4eo_zarr_trainval_50k_cuda}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
INSTALL_DATAFLUX="${INSTALL_DATAFLUX:-0}"
EWM_NUM_WORKERS="${EWM_NUM_WORKERS:-8}"
EWM_EVAL_NUM_WORKERS="${EWM_EVAL_NUM_WORKERS:-8}"
EWM_PREFETCH_FACTOR="${EWM_PREFETCH_FACTOR:-2}"
EWM_EVAL_PREFETCH_FACTOR="${EWM_EVAL_PREFETCH_FACTOR:-2}"
EWM_EXPERIMENT_ID="${EWM_EXPERIMENT_ID:-}"
EWM_RUN_LABEL="${EWM_RUN_LABEL:-}"

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
    mkdir -p $LOCAL_RUN_ROOT $LOCAL_CHECKPOINT_DIR && \
    nohup env \
      GCS_RUN_URI=$GCS_RUN_URI \
      GCS_DATA_URI=placeholder-unused-for-localdisk \
      DATA_ACCESS_MODE=localdisk \
      LOCAL_RUN_ROOT=$LOCAL_RUN_ROOT \
      LOCAL_DATA_ROOT=$LOCAL_DATA_ROOT \
      LOCAL_CHECKPOINT_DIR=$LOCAL_CHECKPOINT_DIR \
      RUN_LOG_PATH=$RUN_LOG_PATH \
      CONFIG_PATH=$CONFIG_PATH \
      SKIP_SETUP_GPU=$SKIP_SETUP_GPU \
      INSTALL_DATAFLUX=$INSTALL_DATAFLUX \
      EWM_NUM_WORKERS=$EWM_NUM_WORKERS \
      EWM_EVAL_NUM_WORKERS=$EWM_EVAL_NUM_WORKERS \
      EWM_PREFETCH_FACTOR=$EWM_PREFETCH_FACTOR \
      EWM_EVAL_PREFETCH_FACTOR=$EWM_EVAL_PREFETCH_FACTOR \
      EWM_EXPERIMENT_ID=$EWM_EXPERIMENT_ID \
      EWM_RUN_LABEL=$EWM_RUN_LABEL \
      SKIP_DATA_SYNC=1 \
      USE_GCSFUSE_MOUNT=0 \
      USE_GCS_DATA_DIRECT=0 \
      bash earth_world_model/scripts/run_phase2_gpu_vm_localdisk.sh \
      > $LOCAL_RUN_ROOT/launcher.log 2>&1 < /dev/null &'"

echo "Launched local-disk phase2 on $A100_VM_NAME"
