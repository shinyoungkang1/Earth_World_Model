#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build the full sensor-data program on a CPU VM:
# - HLS obs_events
# - daily decoder targets
# - aligned decoder benchmark
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ID="${PROJECT_ID:-omois-483220}"
STAGES="${STAGES:-core_check,hls,decoder}"
HLS_INDEX_PATH="${HLS_INDEX_PATH:-}"
YEARLY_ROOT="${YEARLY_ROOT:-}"
SSL4EO_ROOT="${SSL4EO_ROOT:-}"
YEARLY_TRAIN_INDEX_PATH="${YEARLY_TRAIN_INDEX_PATH:-}"
YEARLY_VAL_INDEX_PATH="${YEARLY_VAL_INDEX_PATH:-}"
LOCATIONS_PATH="${LOCATIONS_PATH:-}"
REQUESTS_PATH="${REQUESTS_PATH:-}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/programs/sensor_data_program_v1}"

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

HLS_MAX_SAMPLES="${HLS_MAX_SAMPLES:-0}"
HLS_MIN_QUALITY_SCORE="${HLS_MIN_QUALITY_SCORE:-0.0}"
RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP="${RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP:-0}"
REQUIRE_EXISTING_HLS_CHIP_PATHS="${REQUIRE_EXISTING_HLS_CHIP_PATHS:-0}"
REQUIRE_CORE_READY="${REQUIRE_CORE_READY:-0}"

AUTHENTICATE="${AUTHENTICATE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_BENCHMARK_IF_EXISTS="${SKIP_BENCHMARK_IF_EXISTS:-0}"
SKIP_GCS_SYNC="${SKIP_GCS_SYNC:-1}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:-}"

if [[ "$STAGES" == *"hls"* && -z "$HLS_INDEX_PATH" ]]; then
  echo "HLS_INDEX_PATH is required when STAGES includes hls." >&2
  exit 1
fi

if [[ "$STAGES" == *"decoder"* ]]; then
  if [[ -z "$LOCATIONS_PATH" ]]; then
    echo "LOCATIONS_PATH is required when STAGES includes decoder." >&2
    exit 1
  fi
  if [[ -z "$REQUESTS_PATH" && ( -z "$START_DATE" || -z "$END_DATE" ) ]]; then
    echo "Either REQUESTS_PATH or both START_DATE and END_DATE are required when STAGES includes decoder." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_ROOT"

env PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"

ARGS=(
  "$PROJECT_ROOT/earth_world_model/scripts/run_sensor_data_program_stage.py"
  --project "$PROJECT_ID"
  --stages "$STAGES"
  --hls-index-path "$HLS_INDEX_PATH"
  --yearly-root "$YEARLY_ROOT"
  --ssl4eo-root "$SSL4EO_ROOT"
  --yearly-train-index-path "$YEARLY_TRAIN_INDEX_PATH"
  --yearly-val-index-path "$YEARLY_VAL_INDEX_PATH"
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
  --hls-max-samples "$HLS_MAX_SAMPLES"
  --hls-min-quality-score "$HLS_MIN_QUALITY_SCORE"
  --output-root "$OUTPUT_ROOT"
)

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
if [[ "$RESOLVE_MISSING_HLS_DATETIMES_FROM_CHIP" == "1" ]]; then
  ARGS+=(--resolve-missing-hls-datetimes-from-chip)
fi
if [[ "$REQUIRE_EXISTING_HLS_CHIP_PATHS" == "1" ]]; then
  ARGS+=(--require-existing-hls-chip-paths)
fi
if [[ "$REQUIRE_CORE_READY" == "1" ]]; then
  ARGS+=(--require-core-ready)
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
