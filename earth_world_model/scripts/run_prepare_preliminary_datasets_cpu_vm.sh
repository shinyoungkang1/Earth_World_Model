#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DISK_MOUNT_POINT="${DATA_DISK_MOUNT_POINT:-/mnt/ewm-data-disk}"
DATA_DISK_DEVICE="${DATA_DISK_DEVICE:-/dev/disk/by-id/google-ewm-training-data}"
FORMAT_IF_NEEDED="${FORMAT_IF_NEEDED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DISK_MOUNT_POINT/ewm_prelim_400_v1}"
WORK_DIR="${WORK_DIR:-/tmp/ewm_prelim_400_prep}"
YEARLY_RAW_ROOTS="${YEARLY_RAW_ROOTS:-gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_exact_chip52w_batch100:gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_interactive_exact_chip_100_10000_v1/raw}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
YEARLY_TRAIN_COUNT="${YEARLY_TRAIN_COUNT:-400}"
YEARLY_VAL_COUNT="${YEARLY_VAL_COUNT:-32}"
SSL4EO_TRAIN_COUNT="${SSL4EO_TRAIN_COUNT:-400}"
SSL4EO_VAL_COUNT="${SSL4EO_VAL_COUNT:-128}"
SSL4EO_CACHE_DIR="${SSL4EO_CACHE_DIR:-/tmp/ssl4eo_zarr_downloads}"
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
  --ssl4eo-train-count "$SSL4EO_TRAIN_COUNT"
  --ssl4eo-val-count "$SSL4EO_VAL_COUNT"
  --ssl4eo-cache-dir "$SSL4EO_CACHE_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

IFS=':' read -r -a YEARLY_ROOT_ARRAY <<< "$YEARLY_RAW_ROOTS"
for raw_root in "${YEARLY_ROOT_ARRAY[@]}"; do
  if [[ -n "$raw_root" ]]; then
    ARGS+=(--yearly-raw-root "$raw_root")
  fi
done

python3 "${ARGS[@]}"
