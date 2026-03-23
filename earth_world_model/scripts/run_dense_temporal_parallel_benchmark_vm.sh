#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/Mineral_Gas_Locator}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dense_temporal_bench_2020_256_parallel4}"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs/dense_temporal_bench_2020_256_parallel4}"
LOCATIONS_PATH="${LOCATIONS_PATH:-data/raw/dense_temporal_seed_locations_pilot10.csv}"
YEARS="${YEARS:-2020}"
CADENCE_DAYS="${CADENCE_DAYS:-7}"
SEQUENCE_LIMIT="${SEQUENCE_LIMIT:-10}"
CHIP_SIZE="${CHIP_SIZE:-256}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-zarr}"
SHARD_COUNT="${SHARD_COUNT:-4}"
S2_CLOUD_COVER_MAX="${S2_CLOUD_COVER_MAX:-80.0}"
S2_SHORTLIST_PER_BIN="${S2_SHORTLIST_PER_BIN:-5}"
MAX_ITEMS_PER_SEARCH="${MAX_ITEMS_PER_SEARCH:-2000}"
LOG_EVERY_FRAME_REQUESTS="${LOG_EVERY_FRAME_REQUESTS:-10}"
SOURCE_BACKEND="${SOURCE_BACKEND:-planetary_computer}"
DIRECT_ASSET_SOURCE="${DIRECT_ASSET_SOURCE:-remote}"
LOCAL_ASSET_CACHE_DIR="${LOCAL_ASSET_CACHE_DIR:-$OUTPUT_DIR/asset_cache}"
ALLOW_NONCANONICAL_PIXEL_ACCESS="${ALLOW_NONCANONICAL_PIXEL_ACCESS:-0}"

if [[ "$SOURCE_BACKEND" == "cdse" ]]; then
  STAC_API_URL="${STAC_API_URL:-https://stac.dataspace.copernicus.eu/v1}"
  SIGNING_MODE="${SIGNING_MODE:-none}"
  S2_COLLECTION="${S2_COLLECTION:-sentinel-2-l2a}"
  S1_COLLECTION="${S1_COLLECTION:-sentinel-1-grd}"
  S2_REQUIRED_ASSETS="${S2_REQUIRED_ASSETS:-B02_10m,B03_10m,B04_10m,B05_20m,B06_20m,B07_20m,B08_10m,B8A_20m,B11_20m,B12_20m,SCL_20m}"
  S1_REQUIRED_ASSETS="${S1_REQUIRED_ASSETS:-VV,VH}"
  S2_BAND_ASSETS="${S2_BAND_ASSETS:-B02_10m,B03_10m,B04_10m,B05_20m,B06_20m,B07_20m,B08_10m,B8A_20m,B11_20m,B12_20m}"
  S1_BAND_ASSETS="${S1_BAND_ASSETS:-VV,VH}"
  S2_SCL_ASSET="${S2_SCL_ASSET:-SCL_20m}"
else
  STAC_API_URL="${STAC_API_URL:-https://planetarycomputer.microsoft.com/api/stac/v1}"
  SIGNING_MODE="${SIGNING_MODE:-planetary_computer}"
  S2_COLLECTION="${S2_COLLECTION:-sentinel-2-l2a}"
  S1_COLLECTION="${S1_COLLECTION:-sentinel-1-grd}"
  S2_REQUIRED_ASSETS="${S2_REQUIRED_ASSETS:-B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL}"
  S1_REQUIRED_ASSETS="${S1_REQUIRED_ASSETS:-VV,VH}"
  S2_BAND_ASSETS="${S2_BAND_ASSETS:-B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12}"
  S1_BAND_ASSETS="${S1_BAND_ASSETS:-VV,VH}"
  S2_SCL_ASSET="${S2_SCL_ASSET:-SCL}"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

PLAN_DIR="$OUTPUT_DIR/plan"
MATERIALIZED_ROOT="$OUTPUT_DIR/materialized"
mkdir -p "$PLAN_DIR" "$MATERIALIZED_ROOT"

start_epoch="$(date +%s)"

echo "[dense-bench] starting planner at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG_DIR/benchmark.log"
/usr/bin/time -v -o "$LOG_DIR/planner.time" \
python3 earth_world_model/scripts/plan_dense_temporal_sequences.py \
  --locations-path "$LOCATIONS_PATH" \
  --output-dir "$PLAN_DIR" \
  --sample-id-column sample_id \
  --latitude-column latitude \
  --longitude-column longitude \
  --years "$YEARS" \
  --cadence-days "$CADENCE_DAYS" \
  --sequence-limit "$SEQUENCE_LIMIT" \
  --stac-api-url "$STAC_API_URL" \
  --s2-collection "$S2_COLLECTION" \
  --s1-collection "$S1_COLLECTION" \
  --s2-required-assets "$S2_REQUIRED_ASSETS" \
  --s1-required-assets "$S1_REQUIRED_ASSETS" \
  --s2-cloud-cover-max "$S2_CLOUD_COVER_MAX" \
  --s2-shortlist-per-bin "$S2_SHORTLIST_PER_BIN" \
  --max-items-per-search "$MAX_ITEMS_PER_SEARCH" \
  >"$LOG_DIR/planner.log" 2>&1

echo "[dense-bench] starting ${SHARD_COUNT} materializer shards at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/benchmark.log"

declare -a pids=()
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_out="$MATERIALIZED_ROOT/shard_${shard_idx}"
  mkdir -p "$shard_out"
  /usr/bin/time -v -o "$LOG_DIR/materialize_shard_${shard_idx}.time" \
  python3 earth_world_model/scripts/materialize_dense_temporal_sequences.py \
    --plan-path "$PLAN_DIR/sequence_plan.parquet" \
    --output-dir "$shard_out" \
    --stac-api-url "$STAC_API_URL" \
    --signing-mode "$SIGNING_MODE" \
    --direct-asset-source "$DIRECT_ASSET_SOURCE" \
    --local-asset-cache-dir "$LOCAL_ASSET_CACHE_DIR" \
    --pixel-access-mode direct \
    --chip-size "$CHIP_SIZE" \
    --sequence-limit "$SEQUENCE_LIMIT" \
    --shard-index "$shard_idx" \
    --shard-count "$SHARD_COUNT" \
    --min-paired-bins 1 \
    --min-materialized-paired-bins 1 \
    --max-bins-per-sequence 0 \
    --output-format "$OUTPUT_FORMAT" \
    --materialization-order scene \
    --log-every-frame-requests "$LOG_EVERY_FRAME_REQUESTS" \
    --s2-band-assets "$S2_BAND_ASSETS" \
    --s1-band-assets "$S1_BAND_ASSETS" \
    --s2-scl-asset "$S2_SCL_ASSET" \
    $( [[ "$ALLOW_NONCANONICAL_PIXEL_ACCESS" == "1" ]] && printf '%s' '--allow-noncanonical-pixel-access' ) \
    >"$LOG_DIR/materialize_shard_${shard_idx}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

end_epoch="$(date +%s)"
wall_clock_seconds="$((end_epoch - start_epoch))"

{
  printf '{\n'
  printf '  "years": "%s",\n' "$YEARS"
  printf '  "sequence_limit": %s,\n' "$SEQUENCE_LIMIT"
  printf '  "chip_size": %s,\n' "$CHIP_SIZE"
  printf '  "shard_count": %s,\n' "$SHARD_COUNT"
  printf '  "output_format": "%s",\n' "$OUTPUT_FORMAT"
  printf '  "wall_clock_seconds": %s,\n' "$wall_clock_seconds"
  printf '  "status": %s\n' "$status"
  printf '}\n'
} >"$OUTPUT_DIR/benchmark_summary.json"

echo "[dense-bench] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status} wall_clock_seconds=${wall_clock_seconds}" | tee -a "$LOG_DIR/benchmark.log"
du -sh "$OUTPUT_DIR" | tee -a "$LOG_DIR/benchmark.log"

exit "$status"
