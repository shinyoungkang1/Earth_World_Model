#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-$HOME/ewm_phase2_scale_gpu}"
LOCAL_PROBE_ROOT="${LOCAL_PROBE_ROOT:-$LOCAL_RUN_ROOT/phase3_probe}"
LOCAL_PROBE_CACHE_DIR="${LOCAL_PROBE_CACHE_DIR:-$LOCAL_PROBE_ROOT/cache}"
LOCAL_PROBE_OUTPUT_DIR="${LOCAL_PROBE_OUTPUT_DIR:-$LOCAL_PROBE_ROOT/output}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$LOCAL_RUN_ROOT/checkpoints/ssl4eo_zarr_trainval_phase2_scale_cuda}"
PROBE_LOG_PATH="${PROBE_LOG_PATH:-$LOCAL_PROBE_ROOT/probe.log}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://my-bucket/earth_world_model/runs/phase2_scale_a100}"
GCS_PROBE_URI="${GCS_PROBE_URI:-$GCS_RUN_URI/phase3_probe}"
CHECKPOINT_BASENAME="${CHECKPOINT_BASENAME:-}"
PROBE_TASK="${PROBE_TASK:-landcover_forest__regr}"
PROBE_NETWORK="${PROBE_NETWORK:-teacher}"
PROBE_REPO_ID="${PROBE_REPO_ID:-embed2scale/SSL4EO-S12-downstream}"
PROBE_REPO_SUBDIR="${PROBE_REPO_SUBDIR:-data_example}"
PROBE_PART="${PROBE_PART:-part-000000}"
PROBE_LIMIT="${PROBE_LIMIT:-0}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
    return 0
  fi
  echo "Missing required command: python or python3" >&2
  exit 1
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

resolve_checkpoint_object() {
  if [[ -n "$CHECKPOINT_BASENAME" ]]; then
    echo "$GCS_RUN_URI/checkpoints/$CHECKPOINT_BASENAME"
    return 0
  fi

  if gcloud storage ls "$GCS_RUN_URI/checkpoints/ewm_best_val.pt" >/dev/null 2>&1; then
    echo "$GCS_RUN_URI/checkpoints/ewm_best_val.pt"
    return 0
  fi

  local latest_epoch_checkpoint
  latest_epoch_checkpoint="$(
    gcloud storage ls "${GCS_RUN_URI}/checkpoints/ewm_epoch_*.pt" 2>/dev/null \
      | sort \
      | tail -n 1
  )"
  if [[ -n "$latest_epoch_checkpoint" ]]; then
    echo "$latest_epoch_checkpoint"
    return 0
  fi

  echo "No checkpoint found under $GCS_RUN_URI/checkpoints" >&2
  exit 1
}

require_cmd gcloud
resolve_python_bin

mkdir -p "$LOCAL_CHECKPOINT_DIR" "$LOCAL_PROBE_CACHE_DIR" "$LOCAL_PROBE_OUTPUT_DIR"

sync_outputs() {
  if [[ -d "$LOCAL_PROBE_OUTPUT_DIR" ]]; then
    storage_rsync "$LOCAL_PROBE_OUTPUT_DIR" "$GCS_PROBE_URI/$PROBE_TASK" || true
  fi
  if [[ -f "$PROBE_LOG_PATH" ]]; then
    storage_cp "$PROBE_LOG_PATH" "$GCS_PROBE_URI/$PROBE_TASK/probe.log" || true
  fi
}

trap sync_outputs EXIT

CHECKPOINT_OBJECT="$(resolve_checkpoint_object)"
LOCAL_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR/$(basename "$CHECKPOINT_OBJECT")"

echo "Fetching checkpoint from $CHECKPOINT_OBJECT"
storage_cp "$CHECKPOINT_OBJECT" "$LOCAL_CHECKPOINT_PATH"

if [[ "$SKIP_SETUP_GPU" != "1" ]]; then
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"
fi

export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

cmd=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/earth_world_model/scripts/probe_ssl4eo_downstream.py"
  --checkpoint "$LOCAL_CHECKPOINT_PATH"
  --task "$PROBE_TASK"
  --network "$PROBE_NETWORK"
  --repo-id "$PROBE_REPO_ID"
  --repo-subdir "$PROBE_REPO_SUBDIR"
  --part "$PROBE_PART"
  --cache-dir "$LOCAL_PROBE_CACHE_DIR"
  --output-dir "$LOCAL_PROBE_OUTPUT_DIR"
  --device "$PROBE_DEVICE"
)

if [[ "$PROBE_LIMIT" != "0" ]]; then
  cmd+=(--limit "$PROBE_LIMIT")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$PROBE_LOG_PATH"
