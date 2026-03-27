#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
FORMAT_IF_NEEDED="${FORMAT_IF_NEEDED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DISK_MOUNT_POINT/ewm_prelim_400_v1}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT}_work}"
YEARLY_RAW_ROOTS="${YEARLY_RAW_ROOTS:-gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_exact_chip52w_batch100|gs://omois-earth-world-model-phase2-20260320-11728/ee_exact_chip52w_batch_1000_3000_v2|gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_interactive_exact_chip_100_10000_v1/raw}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
YEARLY_TRAIN_COUNT="${YEARLY_TRAIN_COUNT:-400}"
YEARLY_VAL_COUNT="${YEARLY_VAL_COUNT:-32}"
YEARLY_WORKERS="${YEARLY_WORKERS:-4}"
YEARLY_GCS_DOWNLOAD_WORKERS="${YEARLY_GCS_DOWNLOAD_WORKERS:-4}"
YEARLY_GCS_DOWNLOAD_TIMEOUT_SEC="${YEARLY_GCS_DOWNLOAD_TIMEOUT_SEC:-180}"
YEARLY_GCS_DOWNLOAD_RETRIES="${YEARLY_GCS_DOWNLOAD_RETRIES:-3}"
YEARLY_GCS_TRANSFER_BACKEND="${YEARLY_GCS_TRANSFER_BACKEND:-auto}"
SSL4EO_WORKERS="${SSL4EO_WORKERS:-4}"
TOTAL_WORKER_BUDGET="${TOTAL_WORKER_BUDGET:-0}"
SSL4EO_TRAIN_COUNT="${SSL4EO_TRAIN_COUNT:-400}"
SSL4EO_VAL_COUNT="${SSL4EO_VAL_COUNT:-128}"
SSL4EO_CACHE_DIR="${SSL4EO_CACHE_DIR:-${WORK_DIR}/ssl4eo_zarr_downloads}"
CONCURRENT_BRANCHES="${CONCURRENT_BRANCHES:-1}"
OVERWRITE="${OVERWRITE:-0}"

env \
  DATA_DISK_DEVICE="$DATA_DISK_DEVICE" \
  DATA_DISK_MOUNT_POINT="$DATA_DISK_MOUNT_POINT" \
  FORMAT_IF_NEEDED="$FORMAT_IF_NEEDED" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_training_data_disk.sh"

ARGS=(
  "$PROJECT_ROOT/earth_world_model/scripts/prepare_preliminary_datasets.py"
  --output-root "$OUTPUT_ROOT"
  --work-dir "$WORK_DIR"
  --yearly-name "$YEARLY_NAME"
  --ssl4eo-name "$SSL4EO_NAME"
  --yearly-train-count "$YEARLY_TRAIN_COUNT"
  --yearly-val-count "$YEARLY_VAL_COUNT"
  --yearly-workers "$YEARLY_WORKERS"
  --yearly-gcs-download-workers "$YEARLY_GCS_DOWNLOAD_WORKERS"
  --yearly-gcs-download-timeout-sec "$YEARLY_GCS_DOWNLOAD_TIMEOUT_SEC"
  --yearly-gcs-download-retries "$YEARLY_GCS_DOWNLOAD_RETRIES"
  --yearly-gcs-transfer-backend "$YEARLY_GCS_TRANSFER_BACKEND"
  --ssl4eo-workers "$SSL4EO_WORKERS"
  --total-worker-budget "$TOTAL_WORKER_BUDGET"
  --ssl4eo-train-count "$SSL4EO_TRAIN_COUNT"
  --ssl4eo-val-count "$SSL4EO_VAL_COUNT"
  --ssl4eo-cache-dir "$SSL4EO_CACHE_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "$CONCURRENT_BRANCHES" == "1" ]]; then
  ARGS+=(--concurrent-branches)
fi

IFS='|' read -r -a YEARLY_ROOT_ARRAY <<< "$YEARLY_RAW_ROOTS"
for raw_root in "${YEARLY_ROOT_ARRAY[@]}"; do
  if [[ -n "$raw_root" ]]; then
    ARGS+=(--yearly-raw-root "$raw_root")
  fi
done

python3 "${ARGS[@]}"
