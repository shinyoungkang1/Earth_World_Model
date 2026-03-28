#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build daily decoder targets and the aligned benchmark on a CPU VM.
#
# This script is intended for a general-purpose GCP VM with Earth Engine access.
# It installs repo Python deps, runs the daily extraction stage, and optionally
# syncs the resulting parquet/json artifacts to GCS.
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ID="${PROJECT_ID:-omois-483220}"
LOCATIONS_PATH="${LOCATIONS_PATH:-}"
REQUESTS_PATH="${REQUESTS_PATH:-}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/benchmarks/decoder_target_benchmark_stage_v1}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-}"

SAMPLE_ID_COLUMN="${SAMPLE_ID_COLUMN:-sample_id}"
LATITUDE_COLUMN="${LATITUDE_COLUMN:-latitude}"
LONGITUDE_COLUMN="${LONGITUDE_COLUMN:-longitude}"
REQUEST_SAMPLE_ID_COLUMN="${REQUEST_SAMPLE_ID_COLUMN:-sample_id}"
REQUEST_DATE_COLUMN="${REQUEST_DATE_COLUMN:-date}"

BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
AGGREGATION="${AGGREGATION:-point}"
REGION_SIDE_METERS="${REGION_SIDE_METERS:-2560}"
PRODUCTS="${PRODUCTS:-modis,chirps,smap,era5}"
REQUIRE_SOURCES="${REQUIRE_SOURCES:-modis,chirps,smap}"
FORECAST_HORIZONS_DAYS="${FORECAST_HORIZONS_DAYS:-7,14,30}"
MAX_PARALLEL_PRODUCTS="${MAX_PARALLEL_PRODUCTS:-2}"

AUTHENTICATE="${AUTHENTICATE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_BENCHMARK_IF_EXISTS="${SKIP_BENCHMARK_IF_EXISTS:-0}"
SKIP_GCS_SYNC="${SKIP_GCS_SYNC:-1}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:-}"

if [[ -z "$LOCATIONS_PATH" ]]; then
  echo "LOCATIONS_PATH is required." >&2
  exit 1
fi

if [[ -z "$REQUESTS_PATH" && ( -z "$START_DATE" || -z "$END_DATE" ) ]]; then
  echo "Either REQUESTS_PATH or both START_DATE and END_DATE are required." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

env PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"

ARGS=(
  "$PROJECT_ROOT/earth_world_model/scripts/run_decoder_target_benchmark_stage.py"
  --project "$PROJECT_ID"
  --locations-path "$LOCATIONS_PATH"
  --sample-id-column "$SAMPLE_ID_COLUMN"
  --latitude-column "$LATITUDE_COLUMN"
  --longitude-column "$LONGITUDE_COLUMN"
  --request-sample-id-column "$REQUEST_SAMPLE_ID_COLUMN"
  --request-date-column "$REQUEST_DATE_COLUMN"
  --batch-size "$BATCH_SIZE"
  --max-samples "$MAX_SAMPLES"
  --max-requests "$MAX_REQUESTS"
  --aggregation "$AGGREGATION"
  --region-side-meters "$REGION_SIDE_METERS"
  --products "$PRODUCTS"
  --require-sources "$REQUIRE_SOURCES"
  --forecast-horizons-days "$FORECAST_HORIZONS_DAYS"
  --max-parallel-products "$MAX_PARALLEL_PRODUCTS"
  --output-dir "$OUTPUT_ROOT"
)

if [[ -n "$BENCHMARK_OUTPUT_DIR" ]]; then
  ARGS+=(--benchmark-output-dir "$BENCHMARK_OUTPUT_DIR")
fi

if [[ -n "$REQUESTS_PATH" ]]; then
  ARGS+=(--requests-path "$REQUESTS_PATH")
else
  ARGS+=(--start-date "$START_DATE" --end-date "$END_DATE")
fi

if [[ "$AUTHENTICATE" == "1" ]]; then
  ARGS+=(--authenticate)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi
if [[ "$SKIP_BENCHMARK_IF_EXISTS" == "1" ]]; then
  ARGS+=(--skip-benchmark-if-exists)
fi

"$PYTHON_BIN" "${ARGS[@]}"

if [[ "$SKIP_GCS_SYNC" != "1" ]]; then
  if [[ -z "$GCS_OUTPUT_URI" ]]; then
    echo "GCS_OUTPUT_URI is required when SKIP_GCS_SYNC != 1" >&2
    exit 1
  fi
  gsutil -m rsync -r "$OUTPUT_ROOT" "$GCS_OUTPUT_URI"
fi

echo "============================================================"
echo "DONE"
echo "  Output root: $OUTPUT_ROOT"
if [[ "$SKIP_GCS_SYNC" != "1" ]]; then
  echo "  GCS sync:    $GCS_OUTPUT_URI"
fi
echo "============================================================"
