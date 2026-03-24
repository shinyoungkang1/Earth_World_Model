#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/Mineral_Gas_Locator}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/ee_interactive_vm_run}"
RAW_DIR="${RAW_DIR:-$SCRATCH_ROOT/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$SCRATCH_ROOT/processed}"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/ee_interactive_vm_run.log}"

PROJECT_ID="${PROJECT_ID:-omois-483220}"
LOCATIONS_PATH="${LOCATIONS_PATH:-data/raw/earth_engine_interactive_points_100_150_v1.csv}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"
YEAR="${YEAR:-2020}"
CHIP_SIZE="${CHIP_SIZE:-256}"
RESOLUTION_METERS="${RESOLUTION_METERS:-10}"
REGION_SIDE_METERS="${REGION_SIDE_METERS:-2560}"
WEEK_STEP="${WEEK_STEP:-4}"
TOTAL_WEEKS="${TOTAL_WEEKS:-52}"
PARALLELISM="${PARALLELISM:-5}"

PROCESS_AFTER_DOWNLOAD="${PROCESS_AFTER_DOWNLOAD:-1}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-npz}"

GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="${GCS_PREFIX:-earth_engine_interactive_exact_chip}"
UPLOAD_COMPLETED_SAMPLES_TO_GCS="${UPLOAD_COMPLETED_SAMPLES_TO_GCS:-0}"
DELETE_LOCAL_SAMPLE_AFTER_GCS_UPLOAD="${DELETE_LOCAL_SAMPLE_AFTER_GCS_UPLOAD:-0}"
UPLOAD_RAW_TO_GCS="${UPLOAD_RAW_TO_GCS:-0}"
UPLOAD_PROCESSED_TO_GCS="${UPLOAD_PROCESSED_TO_GCS:-0}"
DELETE_LOCAL_RAW_AFTER_UPLOAD="${DELETE_LOCAL_RAW_AFTER_UPLOAD:-0}"
DELETE_LOCAL_PROCESSED_AFTER_UPLOAD="${DELETE_LOCAL_PROCESSED_AFTER_UPLOAD:-0}"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

{
  echo "[ee-interactive-vm] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[ee-interactive-vm] raw_dir=$RAW_DIR"
  echo "[ee-interactive-vm] processed_dir=$PROCESSED_DIR"
  echo "[ee-interactive-vm] locations_path=$LOCATIONS_PATH offset=$SAMPLE_OFFSET limit=$SAMPLE_LIMIT"
  echo "[ee-interactive-vm] project=$PROJECT_ID year=$YEAR week_step=$WEEK_STEP total_weeks=$TOTAL_WEEKS parallelism=$PARALLELISM"

  BENCHMARK_ARGS=(
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
    --output-dir "$RAW_DIR"
  )

  if [[ -n "$GCS_BUCKET" && "$UPLOAD_COMPLETED_SAMPLES_TO_GCS" == "1" ]]; then
    BENCHMARK_ARGS+=(--gcs-bucket "$GCS_BUCKET" --gcs-prefix "$GCS_PREFIX" --upload-completed-samples-to-gcs)
    if [[ "$DELETE_LOCAL_SAMPLE_AFTER_GCS_UPLOAD" == "1" ]]; then
      BENCHMARK_ARGS+=(--delete-local-after-gcs-upload)
    fi
  fi

  /usr/bin/time -v python3 earth_world_model/scripts/run_earth_engine_exact_chip_interactive_benchmark.py "${BENCHMARK_ARGS[@]}"

  if [[ "$PROCESS_AFTER_DOWNLOAD" == "1" ]]; then
    /usr/bin/time -v python3 earth_world_model/scripts/process_earth_engine_exact_chip_exports.py \
      --input-dir "$RAW_DIR" \
      --locations-path "$LOCATIONS_PATH" \
      --year "$YEAR" \
      --chip-size "$CHIP_SIZE" \
      --output-dir "$PROCESSED_DIR" \
      --output-format "$OUTPUT_FORMAT"
  fi

  if [[ -n "$GCS_BUCKET" && "$UPLOAD_RAW_TO_GCS" == "1" ]]; then
    gcloud storage rsync "$RAW_DIR" "gs://$GCS_BUCKET/$GCS_PREFIX/raw" --recursive
    if [[ "$DELETE_LOCAL_RAW_AFTER_UPLOAD" == "1" ]]; then
      rm -rf "$RAW_DIR"
    fi
  fi

  if [[ -n "$GCS_BUCKET" && "$UPLOAD_PROCESSED_TO_GCS" == "1" ]]; then
    gcloud storage rsync "$PROCESSED_DIR" "gs://$GCS_BUCKET/$GCS_PREFIX/processed" --recursive
    if [[ "$DELETE_LOCAL_PROCESSED_AFTER_UPLOAD" == "1" ]]; then
      rm -rf "$PROCESSED_DIR"
    fi
  fi

  echo "[ee-interactive-vm] finish $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$LOG_PATH" 2>&1

echo "log_path=$LOG_PATH"
