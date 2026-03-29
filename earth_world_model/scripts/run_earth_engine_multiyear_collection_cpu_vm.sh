#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Run year-sharded S1/S2 exact-chip collection on a CPU VM for a fixed sample
# set. This keeps the collection multiyear while preserving the current yearly
# exact-chip collector as the inner execution unit.
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ID="${PROJECT_ID:-omois-483220}"
LOCATIONS_PATH="${LOCATIONS_PATH:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
YEARS="${YEARS:-}"
START_YEAR="${START_YEAR:-0}"
END_YEAR="${END_YEAR:-0}"
STAGE_NAME_PREFIX="${STAGE_NAME_PREFIX:-earth_engine_exact_chip_multiyear_v1}"
STAGE_ROOT_BASE="${STAGE_ROOT_BASE:-}"
LOG_DIR="${LOG_DIR:-}"

CHIP_SIZE="${CHIP_SIZE:-256}"
RESOLUTION_METERS="${RESOLUTION_METERS:-10}"
REGION_SIDE_METERS="${REGION_SIDE_METERS:-2560}"
WEEK_STEP="${WEEK_STEP:-4}"
TOTAL_WEEKS="${TOTAL_WEEKS:-52}"
PARALLELISM="${PARALLELISM:-15}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-5}"
EE_MAX_RETRIES="${EE_MAX_RETRIES:-6}"
EE_RETRY_BACKOFF_SEC="${EE_RETRY_BACKOFF_SEC:-15}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

AUTHENTICATE="${AUTHENTICATE:-0}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX_ROOT="${GCS_PREFIX_ROOT:-earth_engine_interactive_exact_chip_multiyear_v1}"
UPLOAD_COMPLETED_SAMPLES_TO_GCS="${UPLOAD_COMPLETED_SAMPLES_TO_GCS:-1}"
DELETE_LOCAL_AFTER_GCS_UPLOAD="${DELETE_LOCAL_AFTER_GCS_UPLOAD:-1}"
SKIP_EXISTING_YEARS="${SKIP_EXISTING_YEARS:-1}"

OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$PROJECT_ROOT/data/programs/earth_engine_exact_chip_multiyear_v1/collection_summary.json}"

if [[ -z "$LOCATIONS_PATH" ]]; then
  echo "LOCATIONS_PATH is required." >&2
  exit 1
fi
if [[ -z "$YEARS" && ( "$START_YEAR" == "0" || "$END_YEAR" == "0" ) ]]; then
  echo "Provide YEARS or both START_YEAR and END_YEAR." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_SUMMARY_JSON")"

env PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"

ARGS=(
  "$PROJECT_ROOT/earth_world_model/scripts/run_earth_engine_multiyear_collection_stage.py"
  --project-root "$PROJECT_ROOT"
  --project-id "$PROJECT_ID"
  --locations-path "$LOCATIONS_PATH"
  --sample-offset "$SAMPLE_OFFSET"
  --sample-limit "$SAMPLE_LIMIT"
  --stage-name-prefix "$STAGE_NAME_PREFIX"
  --stage-root-base "$STAGE_ROOT_BASE"
  --log-dir "$LOG_DIR"
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
  --min-free-gb "$MIN_FREE_GB"
  --gcs-bucket "$GCS_BUCKET"
  --gcs-prefix-root "$GCS_PREFIX_ROOT"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)

if [[ -n "$YEARS" ]]; then
  ARGS+=(--years "$YEARS")
else
  ARGS+=(--start-year "$START_YEAR" --end-year "$END_YEAR")
fi
if [[ "$AUTHENTICATE" == "1" ]]; then
  ARGS+=(--authenticate)
fi
if [[ "$UPLOAD_COMPLETED_SAMPLES_TO_GCS" == "1" ]]; then
  ARGS+=(--upload-completed-samples-to-gcs)
fi
if [[ "$DELETE_LOCAL_AFTER_GCS_UPLOAD" == "1" ]]; then
  ARGS+=(--delete-local-after-gcs-upload)
fi
if [[ "$SKIP_EXISTING_YEARS" == "1" ]]; then
  ARGS+=(--skip-existing-years)
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo "============================================================"
echo "DONE"
echo "  Summary: $OUTPUT_SUMMARY_JSON"
echo "============================================================"
