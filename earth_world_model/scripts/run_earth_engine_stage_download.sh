#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PROJECT_ROOT="$HOME/workspace/Mineral_Gas_Locator"
if [[ ! -d "$DEFAULT_PROJECT_ROOT/earth_world_model" && -d "$HOME/workspace/earth_world_model" ]]; then
  DEFAULT_PROJECT_ROOT="$HOME/workspace"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
MODE="${MODE:-interactive}"
STAGE_NAME="${STAGE_NAME:-earth_engine_stage}"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
LOCATIONS_PATH="${LOCATIONS_PATH:-data/raw/earth_engine_interactive_points_100_10000_v1.csv}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
YEAR="${YEAR:-2020}"
CHIP_SIZE="${CHIP_SIZE:-256}"
RESOLUTION_METERS="${RESOLUTION_METERS:-10}"
REGION_SIDE_METERS="${REGION_SIDE_METERS:-2560}"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs}"
PRECHECK="${PRECHECK:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
AUTHENTICATE="${AUTHENTICATE:-0}"

DEFAULT_SCRATCH_BASE="$HOME/ee_interactive_scratch"
if [[ -d /mnt/ewm-data-disk && -w /mnt/ewm-data-disk ]]; then
  DEFAULT_SCRATCH_BASE="/mnt/ewm-data-disk/ee_interactive_scratch"
fi
SCRATCH_BASE="${SCRATCH_BASE:-$DEFAULT_SCRATCH_BASE}"
STAGE_ROOT="${STAGE_ROOT:-$SCRATCH_BASE/$STAGE_NAME}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${STAGE_NAME}.log}"

mkdir -p "$LOG_DIR" "$STAGE_ROOT"
export MODE STAGE_NAME PROJECT_ID LOCATIONS_PATH SAMPLE_OFFSET SAMPLE_LIMIT YEAR CHIP_SIZE RESOLUTION_METERS REGION_SIDE_METERS STAGE_ROOT PRECHECK MIN_FREE_GB AUTHENTICATE
cd "$PROJECT_ROOT"

write_manifest() {
  python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["STAGE_ROOT"]) / "stage_manifest.json"
payload = {
    "mode": os.environ.get("MODE"),
    "stage_name": os.environ.get("STAGE_NAME"),
    "project_id": os.environ.get("PROJECT_ID"),
    "locations_path": os.environ.get("LOCATIONS_PATH"),
    "sample_offset": int(os.environ.get("SAMPLE_OFFSET", "0")),
    "sample_limit": int(os.environ.get("SAMPLE_LIMIT", "0")),
    "year": int(os.environ.get("YEAR", "2020")),
    "chip_size": int(os.environ.get("CHIP_SIZE", "256")),
    "resolution_meters": float(os.environ.get("RESOLUTION_METERS", "10")),
    "region_side_meters": float(os.environ.get("REGION_SIDE_METERS", "2560")),
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
    --locations-path "$LOCATIONS_PATH"
    --stage-root "$STAGE_ROOT"
    --min-free-gb "$MIN_FREE_GB"
  )
  if [[ "$AUTHENTICATE" == "1" ]]; then
    PREFLIGHT_ARGS+=(--authenticate)
  fi
  python3 "${PREFLIGHT_ARGS[@]}" >"$STAGE_ROOT/preflight.json"
fi

if [[ "$MODE" == "interactive" ]]; then
  WEEK_STEP="${WEEK_STEP:-4}"
  TOTAL_WEEKS="${TOTAL_WEEKS:-52}"
  PARALLELISM="${PARALLELISM:-15}"
  REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
  MAX_RETRIES="${MAX_RETRIES:-3}"
  RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-5}"
  EE_MAX_RETRIES="${EE_MAX_RETRIES:-6}"
  EE_RETRY_BACKOFF_SEC="${EE_RETRY_BACKOFF_SEC:-15}"
  GCS_BUCKET="${GCS_BUCKET:-}"
  GCS_PREFIX="${GCS_PREFIX:-earth_engine_interactive_exact_chip}"
  UPLOAD_COMPLETED_SAMPLES_TO_GCS="${UPLOAD_COMPLETED_SAMPLES_TO_GCS:-1}"
  DELETE_LOCAL_AFTER_GCS_UPLOAD="${DELETE_LOCAL_AFTER_GCS_UPLOAD:-1}"

  RAW_DIR="${RAW_DIR:-$STAGE_ROOT/raw}"
  mkdir -p "$RAW_DIR"

  ARGS=(
    earth_world_model/scripts/run_earth_engine_exact_chip_interactive_benchmark.py
    --project "$PROJECT_ID"
    --locations-path "$LOCATIONS_PATH"
    --sample-offset "$SAMPLE_OFFSET"
    --sample-limit "$SAMPLE_LIMIT"
    --year "$YEAR"
    --chip-size "$CHIP_SIZE"
    --resolution-meters "$RESOLUTION_METERS"
    --region-side-meters "$REGION_SIDE_METERS"
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
elif [[ "$MODE" == "batch" ]]; then
  BUCKET="${BUCKET:-${GCS_BUCKET:-}}"
  WEEK_START_INDEX="${WEEK_START_INDEX:-0}"
  WEEK_COUNT="${WEEK_COUNT:-52}"
  SHARD_SIZE="${SHARD_SIZE:-256}"
  FILE_DIMENSIONS="${FILE_DIMENSIONS:-256}"
  PRIORITY="${PRIORITY:-}"
  POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
  MAX_POLL_MINUTES="${MAX_POLL_MINUTES:-20}"
  DESCRIPTION_PREFIX="${DESCRIPTION_PREFIX:-$STAGE_NAME}"
  FILE_NAME_PREFIX="${FILE_NAME_PREFIX:-$STAGE_NAME}"
  CANCEL_ACTIVE_FIRST="${CANCEL_ACTIVE_FIRST:-0}"
  OUTPUT_DIR="${OUTPUT_DIR:-$STAGE_ROOT/batch}"

  if [[ -z "$BUCKET" ]]; then
    echo "BUCKET is required for MODE=batch" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  ARGS=(
    earth_world_model/scripts/run_earth_engine_exact_chip_batch_benchmark.py
    --project "$PROJECT_ID"
    --bucket "$BUCKET"
    --locations-path "$LOCATIONS_PATH"
    --sample-offset "$SAMPLE_OFFSET"
    --sample-limit "$SAMPLE_LIMIT"
    --year "$YEAR"
    --week-start-index "$WEEK_START_INDEX"
    --week-count "$WEEK_COUNT"
    --chip-size "$CHIP_SIZE"
    --resolution-meters "$RESOLUTION_METERS"
    --region-side-meters "$REGION_SIDE_METERS"
    --description-prefix "$DESCRIPTION_PREFIX"
    --file-name-prefix "$FILE_NAME_PREFIX"
    --shard-size "$SHARD_SIZE"
    --file-dimensions "$FILE_DIMENSIONS"
    --poll-interval-sec "$POLL_INTERVAL_SEC"
    --max-poll-minutes "$MAX_POLL_MINUTES"
    --output-dir "$OUTPUT_DIR"
  )
  if [[ "$AUTHENTICATE" == "1" ]]; then
    ARGS+=(--authenticate)
  fi

  if [[ -n "$PRIORITY" ]]; then
    ARGS+=(--priority "$PRIORITY")
  fi
  if [[ "$CANCEL_ACTIVE_FIRST" == "1" ]]; then
    ARGS+=(--cancel-active-first)
  fi
else
  echo "Unsupported MODE=$MODE (expected interactive or batch)" >&2
  exit 1
fi

{
  echo "[gee-stage] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[gee-stage] mode=$MODE stage_name=$STAGE_NAME"
  echo "[gee-stage] stage_root=$STAGE_ROOT"
  echo "[gee-stage] log_path=$LOG_PATH"
  printf '[gee-stage] command='
  printf '%q ' python3 "${ARGS[@]}"
  echo
  /usr/bin/time -v python3 "${ARGS[@]}"
  echo "[gee-stage] finish $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$LOG_PATH" 2>&1

echo "log_path=$LOG_PATH"
