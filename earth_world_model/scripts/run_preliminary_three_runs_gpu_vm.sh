#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT_BASE="${DATA_ROOT_BASE:-/mnt/ewm-data-disk/ewm_prelim_400_v1}"
GCS_RUNS_ROOT="${GCS_RUNS_ROOT:-gs://omois-earth-world-model-phase2-20260320-11728/earth_world_model/runs/prelim_400}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
YEARLY_TRAIN_MAX_SAMPLES="${YEARLY_TRAIN_MAX_SAMPLES:-400}"
YEARLY_VAL_MAX_SAMPLES="${YEARLY_VAL_MAX_SAMPLES:-32}"
SSL4EO_TRAIN_MAX_SAMPLES="${SSL4EO_TRAIN_MAX_SAMPLES:-400}"
SSL4EO_VAL_MAX_SAMPLES="${SSL4EO_VAL_MAX_SAMPLES:-128}"

run_one() {
  local run_id="$1"
  local config_path="$2"
  local local_data_root="$3"
  local local_run_root="/tmp/ewm_prelim_400_gpu/${run_id}"
  local local_checkpoint_dir="${local_run_root}/checkpoints/${run_id}"
  local gcs_run_uri="${GCS_RUNS_ROOT%/}/${run_id}"

  env \
    GCS_RUN_URI="$gcs_run_uri" \
    GCS_DATA_URI="gs://placeholder/localdisk_unused" \
    DATA_ACCESS_MODE="localdisk" \
    LOCAL_RUN_ROOT="$local_run_root" \
    LOCAL_DATA_ROOT="$local_data_root" \
    LOCAL_CHECKPOINT_DIR="$local_checkpoint_dir" \
    RUN_LOG_PATH="$local_run_root/train.log" \
    CONFIG_PATH="$config_path" \
    SKIP_SETUP_GPU="$SKIP_SETUP_GPU" \
    EWM_RUN_LABEL="$run_id" \
    EWM_EXPERIMENT_ID="$run_id" \
    EWM_YEARLY_TRAIN_INDEX_PATH="$DATA_ROOT_BASE/$YEARLY_NAME/train/dense_temporal_index.parquet" \
    EWM_YEARLY_VAL_INDEX_PATH="$DATA_ROOT_BASE/$YEARLY_NAME/val/dense_temporal_index.parquet" \
    EWM_YEARLY_TRAIN_MAX_SAMPLES="$YEARLY_TRAIN_MAX_SAMPLES" \
    EWM_YEARLY_VAL_MAX_SAMPLES="$YEARLY_VAL_MAX_SAMPLES" \
    EWM_SSL4EO_ROOT_DIR="$DATA_ROOT_BASE/$SSL4EO_NAME" \
    EWM_SSL4EO_TRAIN_MAX_SAMPLES="$SSL4EO_TRAIN_MAX_SAMPLES" \
    EWM_SSL4EO_VAL_MAX_SAMPLES="$SSL4EO_VAL_MAX_SAMPLES" \
    SKIP_DATA_SYNC="1" \
    USE_GCSFUSE_MOUNT="0" \
    USE_GCS_DATA_DIRECT="0" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase2_gpu_vm_localdisk.sh"
}

run_one \
  "yearly_${YEARLY_TRAIN_MAX_SAMPLES}_stagec_conv2d_1024" \
  "$PROJECT_ROOT/earth_world_model/configs/dense_temporal_index_pilot400_vjepa21_stagec_conv2d_1024_cuda_gpu_vm.yaml" \
  "$DATA_ROOT_BASE/$YEARLY_NAME"

run_one \
  "yearly_${YEARLY_TRAIN_MAX_SAMPLES}_staged_rope_conv3d_1024" \
  "$PROJECT_ROOT/earth_world_model/configs/dense_temporal_index_pilot400_vjepa21_staged_rope_conv3d_1024_cuda_gpu_vm.yaml" \
  "$DATA_ROOT_BASE/$YEARLY_NAME"

run_one \
  "ssl4eo_${SSL4EO_TRAIN_MAX_SAMPLES}_stagec_conv2d_1024" \
  "$PROJECT_ROOT/earth_world_model/configs/ssl4eo_zarr_trainval_400_vjepa21_stagec_dense_conv2d_1024_cuda_gpu_vm.yaml" \
  "$DATA_ROOT_BASE/$SSL4EO_NAME"
