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
HLS_GCS_URI="${HLS_GCS_URI:-}"
HLS_NAME="${HLS_NAME:-hls}"
HLS_LOCAL_ROOT="${HLS_LOCAL_ROOT:-${OUTPUT_ROOT%/}/${HLS_NAME}}"
HLS_INDEX_PATH="${HLS_INDEX_PATH:-}"
HLS_INDEX_RELATIVE_PATH="${HLS_INDEX_RELATIVE_PATH:-hls_chip_index_v1.parquet}"
HLS_TRANSFER_BACKEND="${HLS_TRANSFER_BACKEND:-auto}"
HLS_BUILD_OBS_EVENTS="${HLS_BUILD_OBS_EVENTS:-1}"
HLS_OBS_EVENTS_PARQUET="${HLS_OBS_EVENTS_PARQUET:-${OUTPUT_ROOT%/}/obs_events/hls_obs_events_v1.parquet}"
HLS_OBS_EVENTS_METADATA_JSON="${HLS_OBS_EVENTS_METADATA_JSON:-${OUTPUT_ROOT%/}/obs_events/hls_obs_events_v1_metadata.json}"
HLS_MAX_SAMPLES="${HLS_MAX_SAMPLES:-0}"
HLS_MIN_QUALITY_SCORE="${HLS_MIN_QUALITY_SCORE:-0.0}"
RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP="${RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP:-0}"
REQUIRE_EXISTING_HLS_CHIP_PATHS="${REQUIRE_EXISTING_HLS_CHIP_PATHS:-0}"
YEARLY_TRAIN_COUNT="${YEARLY_TRAIN_COUNT:-400}"
YEARLY_VAL_COUNT="${YEARLY_VAL_COUNT:-32}"
YEARLY_WORKERS="${YEARLY_WORKERS:-4}"
YEARLY_METADATA_FLUSH_SIZE="${YEARLY_METADATA_FLUSH_SIZE:-64}"
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

storage_rsync() {
  local source_uri="$1"
  local destination_dir="$2"
  local backend="${HLS_TRANSFER_BACKEND}"

  mkdir -p "$destination_dir"

  if [[ "$backend" == "auto" || "$backend" == "gcloud" ]]; then
    if gcloud storage rsync --help >/dev/null 2>&1; then
      if gcloud storage rsync --recursive "$source_uri" "$destination_dir"; then
        return 0
      fi
      if [[ "$backend" == "gcloud" ]]; then
        return 1
      fi
    elif [[ "$backend" == "gcloud" ]]; then
      echo "gcloud storage rsync is unavailable" >&2
      return 1
    fi
  fi

  if [[ "$backend" == "auto" || "$backend" == "gsutil" ]]; then
    if command -v gsutil >/dev/null 2>&1; then
      gsutil -m rsync -r "$source_uri" "$destination_dir"
      return 0
    elif [[ "$backend" == "gsutil" ]]; then
      echo "gsutil is unavailable" >&2
      return 1
    fi
  fi

  echo "Unsupported HLS transfer backend: $backend" >&2
  return 1
}

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
  --yearly-metadata-flush-size "$YEARLY_METADATA_FLUSH_SIZE"
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

if [[ -n "$HLS_GCS_URI" ]]; then
  echo "[prep] Staging HLS from $HLS_GCS_URI -> $HLS_LOCAL_ROOT"
  storage_rsync "$HLS_GCS_URI" "$HLS_LOCAL_ROOT"
fi

RESOLVED_HLS_INDEX_PATH="$HLS_INDEX_PATH"
if [[ -z "$RESOLVED_HLS_INDEX_PATH" && -n "$HLS_LOCAL_ROOT" ]]; then
  RESOLVED_HLS_INDEX_PATH="${HLS_LOCAL_ROOT%/}/${HLS_INDEX_RELATIVE_PATH}"
fi

if [[ "$HLS_BUILD_OBS_EVENTS" == "1" && -n "$RESOLVED_HLS_INDEX_PATH" && -f "$RESOLVED_HLS_INDEX_PATH" ]]; then
  mkdir -p "$(dirname "$HLS_OBS_EVENTS_PARQUET")"
  HLS_ARGS=(
    "$PROJECT_ROOT/earth_world_model/scripts/build_hls_obs_events.py"
    --hls-index-path "$RESOLVED_HLS_INDEX_PATH"
    --output-parquet "$HLS_OBS_EVENTS_PARQUET"
    --output-metadata-json "$HLS_OBS_EVENTS_METADATA_JSON"
    --max-samples "$HLS_MAX_SAMPLES"
    --min-quality-score "$HLS_MIN_QUALITY_SCORE"
  )
  if [[ "$RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP" == "1" ]]; then
    HLS_ARGS+=(--resolve-missing-datetimes-from-chip)
  fi
  if [[ "$REQUIRE_EXISTING_HLS_CHIP_PATHS" == "1" ]]; then
    HLS_ARGS+=(--require-existing-chip-paths)
  fi
  echo "[prep] Building HLS obs_events from $RESOLVED_HLS_INDEX_PATH"
  python3 "${HLS_ARGS[@]}"
fi
