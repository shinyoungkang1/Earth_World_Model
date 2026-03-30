#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PROJECT_ROOT="$HOME/workspace/Mineral_Gas_Locator"
if [[ ! -d "$DEFAULT_PROJECT_ROOT/earth_world_model" && -d "$HOME/workspace/earth_world_model" ]]; then
  DEFAULT_PROJECT_ROOT="$HOME/workspace"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
STAGE_NAME="${STAGE_NAME:-earth_engine_grouped_stage}"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
REQUESTS_PATH="${REQUESTS_PATH:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
YEAR="${YEAR:-2020}"
DEFAULT_CHIP_SIZE="${DEFAULT_CHIP_SIZE:-256}"
DEFAULT_RESOLUTION_METERS="${DEFAULT_RESOLUTION_METERS:-10}"
DEFAULT_REGION_SIDE_METERS="${DEFAULT_REGION_SIDE_METERS:-2560}"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs}"
PRECHECK="${PRECHECK:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
AUTHENTICATE="${AUTHENTICATE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "$REQUESTS_PATH" ]]; then
  echo "REQUESTS_PATH is required" >&2
  exit 1
fi

DEFAULT_SCRATCH_BASE="$HOME/ee_interactive_scratch"
if [[ -d /mnt/ewm-data-disk && -w /mnt/ewm-data-disk ]]; then
  DEFAULT_SCRATCH_BASE="/mnt/ewm-data-disk/ee_interactive_scratch"
fi
SCRATCH_BASE="${SCRATCH_BASE:-$DEFAULT_SCRATCH_BASE}"
STAGE_ROOT="${STAGE_ROOT:-$SCRATCH_BASE/$STAGE_NAME}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${STAGE_NAME}.log}"

mkdir -p "$LOG_DIR" "$STAGE_ROOT"
cd "$PROJECT_ROOT"

write_manifest() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["STAGE_ROOT"]) / "stage_manifest.json"
payload = {
    "stage_name": os.environ.get("STAGE_NAME"),
    "project_id": os.environ.get("PROJECT_ID"),
    "requests_path": os.environ.get("REQUESTS_PATH"),
    "sample_offset": int(os.environ.get("SAMPLE_OFFSET", "0")),
    "sample_limit": int(os.environ.get("SAMPLE_LIMIT", "0")),
    "year": int(os.environ.get("YEAR", "2020")),
    "default_chip_size": int(os.environ.get("DEFAULT_CHIP_SIZE", "256")),
    "default_resolution_meters": float(os.environ.get("DEFAULT_RESOLUTION_METERS", "10")),
    "default_region_side_meters": float(os.environ.get("DEFAULT_REGION_SIDE_METERS", "2560")),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

write_manifest

if [[ "$PRECHECK" == "1" ]]; then
  PREFLIGHT_ARGS=(
    earth_world_model/scripts/check_earth_engine_stage_ready.py
    --project "$PROJECT_ID"
    --locations-path "$REQUESTS_PATH"
    --stage-root "$STAGE_ROOT"
    --min-free-gb "$MIN_FREE_GB"
  )
  if [[ "$AUTHENTICATE" == "1" ]]; then
    PREFLIGHT_ARGS+=(--authenticate)
  fi
  "$PYTHON_BIN" "${PREFLIGHT_ARGS[@]}" >"$STAGE_ROOT/preflight.json"
fi

WEEK_STEP="${WEEK_STEP:-4}"
TOTAL_WEEKS="${TOTAL_WEEKS:-52}"
PARALLELISM="${PARALLELISM:-8}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-5}"
EE_MAX_RETRIES="${EE_MAX_RETRIES:-6}"
EE_RETRY_BACKOFF_SEC="${EE_RETRY_BACKOFF_SEC:-15}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="${GCS_PREFIX:-earth_engine_grouped_exact_chip}"
UPLOAD_COMPLETED_SAMPLES_TO_GCS="${UPLOAD_COMPLETED_SAMPLES_TO_GCS:-1}"
DELETE_LOCAL_AFTER_GCS_UPLOAD="${DELETE_LOCAL_AFTER_GCS_UPLOAD:-1}"

RAW_DIR="${RAW_DIR:-$STAGE_ROOT/raw}"
mkdir -p "$RAW_DIR"

ARGS=(
  earth_world_model/scripts/run_earth_engine_grouped_exact_chip_interactive_benchmark.py
  --project "$PROJECT_ID"
  --requests-path "$REQUESTS_PATH"
  --sample-offset "$SAMPLE_OFFSET"
  --sample-limit "$SAMPLE_LIMIT"
  --year "$YEAR"
  --default-chip-size "$DEFAULT_CHIP_SIZE"
  --default-resolution-meters "$DEFAULT_RESOLUTION_METERS"
  --default-region-side-meters "$DEFAULT_REGION_SIDE_METERS"
  --week-step "$WEEK_STEP"
  --total-weeks "$TOTAL_WEEKS"
  --parallelism "$PARALLELISM"
  --request-timeout-sec "$REQUEST_TIMEOUT_SEC"
  --max-retries "$MAX_RETRIES"
  --retry-backoff-sec "$RETRY_BACKOFF_SEC"
  --ee-max-retries "$EE_MAX_RETRIES"
  --ee-retry-backoff-sec "$EE_RETRY_BACKOFF_SEC"
  --output-dir "$RAW_DIR"
)
if [[ "$AUTHENTICATE" == "1" ]]; then
  ARGS+=(--authenticate)
fi

if [[ -n "$GCS_BUCKET" && "$UPLOAD_COMPLETED_SAMPLES_TO_GCS" == "1" ]]; then
  ARGS+=(--gcs-bucket "$GCS_BUCKET" --gcs-prefix "$GCS_PREFIX" --upload-completed-samples-to-gcs)
  if [[ "$DELETE_LOCAL_AFTER_GCS_UPLOAD" == "1" ]]; then
    ARGS+=(--delete-local-after-gcs-upload)
  fi
fi

{
  echo "[gee-grouped-stage] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[gee-grouped-stage] stage_name=$STAGE_NAME"
  echo "[gee-grouped-stage] stage_root=$STAGE_ROOT"
  echo "[gee-grouped-stage] log_path=$LOG_PATH"
  echo "[gee-grouped-stage] python_bin=$PYTHON_BIN"
  printf '[gee-grouped-stage] command='
  printf '%q ' "$PYTHON_BIN" "${ARGS[@]}"
  echo
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$PYTHON_BIN" "${ARGS[@]}"
  else
    "$PYTHON_BIN" "${ARGS[@]}"
  fi
  echo "[gee-grouped-stage] finish $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$LOG_PATH" 2>&1

echo "log_path=$LOG_PATH"
