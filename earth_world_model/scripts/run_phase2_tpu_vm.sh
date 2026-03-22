#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/earth_world_model/configs/ssl4eo_zarr_trainval_phase2_scale_tpu_vm.yaml}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-$HOME/ewm_phase2_scale}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$LOCAL_RUN_ROOT/checkpoints/ssl4eo_zarr_trainval_phase2_scale_tpu}"
RUN_LOG_PATH="${RUN_LOG_PATH:-$LOCAL_RUN_ROOT/train.log}"
GCS_DATA_URI="${GCS_DATA_URI:?Set GCS_DATA_URI, for example gs://my-bucket/earth_world_model/data/raw/ssl4eo_zarr_minimal}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://my-bucket/earth_world_model/runs/phase2_scale}"
DATASET_BASENAME="${DATASET_BASENAME:-${GCS_DATA_URI##*/}}"
PREFERRED_LOCAL_DATA_ROOT="${PREFERRED_LOCAL_DATA_ROOT:-/mnt/ewm-data-disk/$DATASET_BASENAME}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-}"
FALLBACK_LOCAL_DATA_ROOT="${FALLBACK_LOCAL_DATA_ROOT:-$LOCAL_RUN_ROOT/data/$DATASET_BASENAME}"
RESUME_FROM_BASENAME="${RESUME_FROM_BASENAME:-}"
SKIP_SETUP_TPU="${SKIP_SETUP_TPU:-0}"
SKIP_DATA_SYNC="${SKIP_DATA_SYNC:-0}"
DATA_ACCESS_MODE="${DATA_ACCESS_MODE:-auto}"
USE_GCS_DATA_DIRECT="${USE_GCS_DATA_DIRECT:-}"
USE_GCSFUSE_MOUNT="${USE_GCSFUSE_MOUNT:-}"
CONTINUOUS_GCS_SYNC="${CONTINUOUS_GCS_SYNC:-1}"
GCSFUSE_MOUNT_POINT="${GCSFUSE_MOUNT_POINT:-$LOCAL_RUN_ROOT/gcsfuse_mount}"
GCSFUSE_CACHE_DISK_MOUNT="${GCSFUSE_CACHE_DISK_MOUNT:-$LOCAL_RUN_ROOT/gcsfuse_cache_disk}"
GCSFUSE_CACHE_DIR="${GCSFUSE_CACHE_DIR:-$GCSFUSE_CACHE_DISK_MOUNT/gcsfuse-cache}"
GCSFUSE_CACHE_DISK_DEVICE="${GCSFUSE_CACHE_DISK_DEVICE:-}"
GCSFUSE_FILE_CACHE_MAX_SIZE_MB="${GCSFUSE_FILE_CACHE_MAX_SIZE_MB:-0}"
GCSFUSE_PROFILE="${GCSFUSE_PROFILE:-aiml-training}"
FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS="${FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS:-true}"
FILE_CACHE_MAX_PARALLEL_DOWNLOADS="${FILE_CACHE_MAX_PARALLEL_DOWNLOADS:-0}"
FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE="${FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE:-0}"
SEQUENTIAL_READ_SIZE_MB="${SEQUENTIAL_READ_SIZE_MB:-0}"
GCSFUSE_STAT_CACHE_MAX_SIZE_MB="${GCSFUSE_STAT_CACHE_MAX_SIZE_MB:-256}"
GCSFUSE_KERNEL_LIST_CACHE_TTL_SECS="${GCSFUSE_KERNEL_LIST_CACHE_TTL_SECS:-300}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

dir_has_entries() {
  local dir_path="$1"
  [[ -d "$dir_path" ]] || return 1
  find "$dir_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

ensure_parent_dir() {
  local dir_path="$1"
  local parent_dir
  parent_dir="$(dirname "$dir_path")"
  mkdir -p "$parent_dir"
}

storage_rsync() {
  if gcloud storage rsync --help >/dev/null 2>&1; then
    gcloud storage rsync --recursive "$1" "$2"
    return 0
  fi
  if gcloud storage cp --help >/dev/null 2>&1; then
    gcloud storage cp --recursive "$1" "$2"
    return 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$1" "$2"
    return 0
  fi
  echo "Missing directory sync support: need 'gcloud storage rsync', 'gcloud storage cp --recursive', or 'gsutil'" >&2
  exit 1
}

storage_cp() {
  if gcloud storage cp --help >/dev/null 2>&1; then
    gcloud storage cp "$1" "$2"
    return 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil cp "$1" "$2"
    return 0
  fi
  echo "Missing file copy support: need 'gcloud storage cp' or 'gsutil'" >&2
  exit 1
}

require_cmd gcloud
require_cmd python

mkdir -p "$LOCAL_RUN_ROOT" "$LOCAL_CHECKPOINT_DIR"

if [[ "$USE_GCSFUSE_MOUNT" == "1" && "$USE_GCS_DATA_DIRECT" == "1" ]]; then
  echo "USE_GCSFUSE_MOUNT=1 and USE_GCS_DATA_DIRECT=1 are mutually exclusive" >&2
  exit 1
fi
if [[ "$USE_GCSFUSE_MOUNT" == "1" ]]; then
  DATA_ACCESS_MODE="gcsfuse"
elif [[ "$USE_GCS_DATA_DIRECT" == "1" ]]; then
  DATA_ACCESS_MODE="gcs_direct"
fi

case "$DATA_ACCESS_MODE" in
  auto)
    if [[ -n "$LOCAL_DATA_ROOT" ]] && dir_has_entries "$LOCAL_DATA_ROOT"; then
      DATA_ACCESS_MODE="localdisk"
      DATA_ROOT="$LOCAL_DATA_ROOT"
      SKIP_DATA_SYNC=1
    elif dir_has_entries "$PREFERRED_LOCAL_DATA_ROOT"; then
      DATA_ACCESS_MODE="localdisk"
      DATA_ROOT="$PREFERRED_LOCAL_DATA_ROOT"
      SKIP_DATA_SYNC=1
    else
      DATA_ACCESS_MODE="localsync"
      DATA_ROOT="${LOCAL_DATA_ROOT:-$FALLBACK_LOCAL_DATA_ROOT}"
    fi
    ;;
  localdisk)
    DATA_ROOT="${LOCAL_DATA_ROOT:-$PREFERRED_LOCAL_DATA_ROOT}"
    SKIP_DATA_SYNC=1
    ;;
  localsync)
    DATA_ROOT="${LOCAL_DATA_ROOT:-$FALLBACK_LOCAL_DATA_ROOT}"
    ;;
  gcsfuse)
    DATA_ROOT="$GCSFUSE_MOUNT_POINT"
    SKIP_DATA_SYNC=1
    ;;
  gcs_direct)
    DATA_ROOT="$GCS_DATA_URI"
    SKIP_DATA_SYNC=1
    ;;
  *)
    echo "Unsupported DATA_ACCESS_MODE: $DATA_ACCESS_MODE" >&2
    echo "Expected one of: auto, localdisk, localsync, gcsfuse, gcs_direct" >&2
    exit 1
    ;;
esac

USE_GCSFUSE_MOUNT=0
USE_GCS_DATA_DIRECT=0
if [[ "$DATA_ACCESS_MODE" == "gcsfuse" ]]; then
  USE_GCSFUSE_MOUNT=1
elif [[ "$DATA_ACCESS_MODE" == "gcs_direct" ]]; then
  USE_GCS_DATA_DIRECT=1
fi

if [[ "$DATA_ROOT" != gs://* && "$USE_GCSFUSE_MOUNT" != "1" ]]; then
  ensure_parent_dir "$DATA_ROOT"
  mkdir -p "$DATA_ROOT"
fi

echo "Selected data access mode: $DATA_ACCESS_MODE"
echo "Data root: $DATA_ROOT"

sync_outputs() {
  if [[ -d "$LOCAL_CHECKPOINT_DIR" ]]; then
    storage_rsync "$LOCAL_CHECKPOINT_DIR" "$GCS_RUN_URI/checkpoints" || true
  fi
  if [[ -f "$RUN_LOG_PATH" ]]; then
    storage_cp "$RUN_LOG_PATH" "$GCS_RUN_URI/train.log" || true
  fi
}

trap sync_outputs EXIT

if [[ "$SKIP_DATA_SYNC" == "1" ]]; then
  echo "Skipping SSL4EO data sync; using data root $DATA_ROOT"
else
  echo "Staging SSL4EO data from $GCS_DATA_URI to $DATA_ROOT"
  storage_rsync "$GCS_DATA_URI" "$DATA_ROOT"
fi

RESUME_FROM_LOCAL=""
if [[ -n "$RESUME_FROM_BASENAME" ]]; then
  RESUME_FROM_LOCAL="$LOCAL_CHECKPOINT_DIR/$RESUME_FROM_BASENAME"
  echo "Fetching resume checkpoint $RESUME_FROM_BASENAME"
  storage_cp "$GCS_RUN_URI/checkpoints/$RESUME_FROM_BASENAME" "$RESUME_FROM_LOCAL"
fi

if [[ "$SKIP_SETUP_TPU" != "1" ]]; then
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_tpu.sh"
fi

if [[ "$USE_GCSFUSE_MOUNT" == "1" ]]; then
  GCS_URI_NO_SCHEME="${GCS_DATA_URI#gs://}"
  GCSFUSE_BUCKET_NAME="${GCS_URI_NO_SCHEME%%/*}"
  GCSFUSE_ONLY_DIR="${GCS_URI_NO_SCHEME#${GCSFUSE_BUCKET_NAME}/}"
  env \
    BUCKET_NAME="$GCSFUSE_BUCKET_NAME" \
    ONLY_DIR="$GCSFUSE_ONLY_DIR" \
    MOUNT_POINT="$GCSFUSE_MOUNT_POINT" \
    CACHE_DISK_MOUNT="$GCSFUSE_CACHE_DISK_MOUNT" \
    CACHE_DIR="$GCSFUSE_CACHE_DIR" \
    CACHE_DISK_DEVICE="$GCSFUSE_CACHE_DISK_DEVICE" \
    GCSFUSE_PROFILE="$GCSFUSE_PROFILE" \
    FILE_CACHE_MAX_SIZE_MB="$GCSFUSE_FILE_CACHE_MAX_SIZE_MB" \
    FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS="$FILE_CACHE_ENABLE_PARALLEL_DOWNLOADS" \
    FILE_CACHE_MAX_PARALLEL_DOWNLOADS="$FILE_CACHE_MAX_PARALLEL_DOWNLOADS" \
    FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE="$FILE_CACHE_PARALLEL_DOWNLOADS_PER_FILE" \
    SEQUENTIAL_READ_SIZE_MB="$SEQUENTIAL_READ_SIZE_MB" \
    STAT_CACHE_MAX_SIZE_MB="$GCSFUSE_STAT_CACHE_MAX_SIZE_MB" \
    KERNEL_LIST_CACHE_TTL_SECS="$GCSFUSE_KERNEL_LIST_CACHE_TTL_SECS" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gcsfuse_cache_mount.sh"
fi

export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export EWM_BACKEND=tpu
export EWM_SSL4EO_ROOT="$DATA_ROOT"
export EWM_CHECKPOINT_DIR="$LOCAL_CHECKPOINT_DIR"
if [[ "$USE_GCSFUSE_MOUNT" == "1" ]]; then
  export EWM_FORCE_SINGLE_PROCESS_IO=1
fi

if [[ "$CONTINUOUS_GCS_SYNC" == "1" ]]; then
  nohup env \
    GCS_RUN_URI="$GCS_RUN_URI" \
    TRAIN_LOG="$RUN_LOG_PATH" \
    METRICS_PATH="$LOCAL_CHECKPOINT_DIR/metrics.jsonl" \
    SUMMARY_PATH="$LOCAL_CHECKPOINT_DIR/training_summary.json" \
    bash "$PROJECT_ROOT/earth_world_model/scripts/sync_run_artifacts_loop.sh" \
    > "$LOCAL_RUN_ROOT/artifact_sync.log" 2>&1 < /dev/null &
fi

cmd=(python "$PROJECT_ROOT/earth_world_model/train_tpu.py" --config "$CONFIG_PATH")
if [[ -n "$RESUME_FROM_LOCAL" ]]; then
  cmd+=(--resume-from "$RESUME_FROM_LOCAL")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$RUN_LOG_PATH"
