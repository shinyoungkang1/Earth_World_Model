#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/Mineral_Gas_Locator}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dense_temporal_bench_2020_256_zarr_opt1}"
LOG_PATH="${LOG_PATH:-$HOME/bench_logs/dense_temporal_bench_2020_256_zarr_opt1.log}"
LOCATIONS_PATH="${LOCATIONS_PATH:-data/raw/dense_temporal_seed_locations_pilot10.csv}"
YEARS="${YEARS:-2020}"
CADENCE_DAYS="${CADENCE_DAYS:-7}"
SEQUENCE_LIMIT="${SEQUENCE_LIMIT:-10}"
CHIP_SIZE="${CHIP_SIZE:-256}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-zarr}"

mkdir -p "$(dirname "$LOG_PATH")"
cd "$PROJECT_ROOT"
pkill -f run_dense_temporal_local_pilot.py || true

/usr/bin/time -v python3 earth_world_model/scripts/run_dense_temporal_local_pilot.py \
  --locations-path "$LOCATIONS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --years "$YEARS" \
  --cadence-days "$CADENCE_DAYS" \
  --sequence-limit "$SEQUENCE_LIMIT" \
  --chip-size "$CHIP_SIZE" \
  --output-format "$OUTPUT_FORMAT" \
  >"$LOG_PATH" 2>&1
