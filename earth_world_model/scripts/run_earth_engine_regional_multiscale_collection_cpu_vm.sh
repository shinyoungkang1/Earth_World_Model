#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/EO_WM}"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
LOCAL_REQUESTS_PATH="${LOCAL_REQUESTS_PATH:-}"
REGIONAL_REQUESTS_PATH="${REGIONAL_REQUESTS_PATH:-}"
YEARS="${YEARS:-2019,2020,2021}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
LOCAL_PARALLELISM="${LOCAL_PARALLELISM:-6}"
REGIONAL_PARALLELISM="${REGIONAL_PARALLELISM:-6}"
WEEK_STEP="${WEEK_STEP:-4}"
TOTAL_WEEKS="${TOTAL_WEEKS:-52}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-5}"
EE_MAX_RETRIES="${EE_MAX_RETRIES:-6}"
EE_RETRY_BACKOFF_SEC="${EE_RETRY_BACKOFF_SEC:-15}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
AUTHENTICATE="${AUTHENTICATE:-0}"
STAGE_NAME_PREFIX="${STAGE_NAME_PREFIX:-earth_engine_regional_multiscale_v1}"
STAGE_ROOT_BASE="${STAGE_ROOT_BASE:-/mnt/ewm-data-disk/ee_interactive_scratch/$STAGE_NAME_PREFIX}"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX_ROOT="${GCS_PREFIX_ROOT:-earth_engine_regional_multiscale_v1}"
UPLOAD_COMPLETED_SAMPLES_TO_GCS="${UPLOAD_COMPLETED_SAMPLES_TO_GCS:-1}"
DELETE_LOCAL_AFTER_GCS_UPLOAD="${DELETE_LOCAL_AFTER_GCS_UPLOAD:-1}"
SKIP_EXISTING_YEARS="${SKIP_EXISTING_YEARS:-1}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$STAGE_ROOT_BASE/regional_multiscale_collection_summary.json}"

if [[ -z "$LOCAL_REQUESTS_PATH" || -z "$REGIONAL_REQUESTS_PATH" ]]; then
  echo "LOCAL_REQUESTS_PATH and REGIONAL_REQUESTS_PATH are required" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

ARGS=(
  earth_world_model/scripts/run_earth_engine_regional_multiscale_collection_stage.py
  --project-root "$PROJECT_ROOT"
  --project-id "$PROJECT_ID"
  --local-requests-path "$LOCAL_REQUESTS_PATH"
  --regional-requests-path "$REGIONAL_REQUESTS_PATH"
  --years "$YEARS"
  --sample-offset "$SAMPLE_OFFSET"
  --sample-limit "$SAMPLE_LIMIT"
  --local-parallelism "$LOCAL_PARALLELISM"
  --regional-parallelism "$REGIONAL_PARALLELISM"
  --week-step "$WEEK_STEP"
  --total-weeks "$TOTAL_WEEKS"
  --request-timeout-sec "$REQUEST_TIMEOUT_SEC"
  --max-retries "$MAX_RETRIES"
  --retry-backoff-sec "$RETRY_BACKOFF_SEC"
  --ee-max-retries "$EE_MAX_RETRIES"
  --ee-retry-backoff-sec "$EE_RETRY_BACKOFF_SEC"
  --min-free-gb "$MIN_FREE_GB"
  --stage-name-prefix "$STAGE_NAME_PREFIX"
  --stage-root-base "$STAGE_ROOT_BASE"
  --log-dir "$LOG_DIR"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)
if [[ "$AUTHENTICATE" == "1" ]]; then
  ARGS+=(--authenticate)
fi
if [[ -n "$GCS_BUCKET" && "$UPLOAD_COMPLETED_SAMPLES_TO_GCS" == "1" ]]; then
  ARGS+=(--gcs-bucket "$GCS_BUCKET" --gcs-prefix-root "$GCS_PREFIX_ROOT" --upload-completed-samples-to-gcs)
  if [[ "$DELETE_LOCAL_AFTER_GCS_UPLOAD" == "1" ]]; then
    ARGS+=(--delete-local-after-gcs-upload)
  fi
fi
if [[ "$SKIP_EXISTING_YEARS" == "1" ]]; then
  ARGS+=(--skip-existing-years)
fi

python3 "${ARGS[@]}"
