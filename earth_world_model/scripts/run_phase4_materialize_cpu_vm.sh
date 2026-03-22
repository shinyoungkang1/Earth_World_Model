#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_PHASE4_ROOT="${LOCAL_PHASE4_ROOT:-$HOME/ewm_phase4_materialize}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$LOCAL_PHASE4_ROOT/data}"
LOCAL_FEATURES_DIR="${LOCAL_FEATURES_DIR:-$LOCAL_DATA_ROOT/features}"
LOCAL_DERIVED_DIR="${LOCAL_DERIVED_DIR:-$LOCAL_DATA_ROOT/derived}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$LOCAL_PHASE4_ROOT/logs}"
MATERIALIZE_LOG_PATH="${MATERIALIZE_LOG_PATH:-$LOCAL_LOG_DIR/materialize.log}"
GCS_PROJECT_DATA_URI="${GCS_PROJECT_DATA_URI:-}"
GCS_DATA_URI="${GCS_DATA_URI:-}"
GCS_PHASE4_URI="${GCS_PHASE4_URI:-}"
COHORT_GCS_PATH="${COHORT_GCS_PATH:-}"
COHORT_METADATA_GCS_PATH="${COHORT_METADATA_GCS_PATH:-}"
COHORT_OUTPUT_DIR="${COHORT_OUTPUT_DIR:-}"
BASIN_ID="${BASIN_ID:-swpa_core_washington_greene}"
LABEL_COLUMN="${LABEL_COLUMN:-label_f12_ge_500000}"
HOLDOUT_QUANTILE="${HOLDOUT_QUANTILE:-0.8}"
MAX_TRAINING_WELLS="${MAX_TRAINING_WELLS:-1600}"
MAX_GRID_CELLS="${MAX_GRID_CELLS:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SAMPLE_YEAR="${SAMPLE_YEAR:-2024}"
IMAGERY_ANCHOR_MODE="${IMAGERY_ANCHOR_MODE:-year_from_date_column}"
ANCHOR_DATE_COLUMN="${ANCHOR_DATE_COLUMN:-first_prod_date}"
ANCHOR_YEAR_OFFSET="${ANCHOR_YEAR_OFFSET:--1}"
S2_CLOUD_COVER_MAX="${S2_CLOUD_COVER_MAX:-35.0}"
MAX_ITEMS_PER_WINDOW="${MAX_ITEMS_PER_WINDOW:-200}"
MIN_S2_CLEAR_FRACTION="${MIN_S2_CLEAR_FRACTION:-0.5}"
MIN_S1_VALID_FRACTION="${MIN_S1_VALID_FRACTION:-0.9}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SKIP_SETUP_CPU="${SKIP_SETUP_CPU:-0}"
SKIP_GRID="${SKIP_GRID:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ENABLE_PERIODIC_SYNC="${ENABLE_PERIODIC_SYNC:-1}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
PYTHON_BIN="${PYTHON_BIN:-}"
COHORT_LOCAL_DIR=""
COHORT_LOCAL_PATH=""
COHORT_METADATA_LOCAL_PATH=""
COHORT_OUTPUT_DIR_RESOLVED=""
PERIODIC_SYNC_PID=""

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
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
    return 0
  fi
  echo "Missing required command: python3 or python" >&2
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

resolve_project_data_uri() {
  if [[ -n "$GCS_PROJECT_DATA_URI" ]]; then
    echo "$GCS_PROJECT_DATA_URI"
    return 0
  fi
  if [[ -z "$GCS_DATA_URI" ]]; then
    echo "Set GCS_PROJECT_DATA_URI or GCS_DATA_URI." >&2
    exit 1
  fi
  echo "${GCS_DATA_URI%/raw/ssl4eo_zarr_minimal}"
}

in_cohort_mode() {
  [[ -n "$COHORT_GCS_PATH" ]]
}

resolve_cohort_metadata_gcs_path() {
  if [[ -n "$COHORT_METADATA_GCS_PATH" ]]; then
    echo "$COHORT_METADATA_GCS_PATH"
    return 0
  fi
  echo "${COHORT_GCS_PATH%.parquet}_metadata.json"
}

resolve_cohort_local_dir() {
  echo "$LOCAL_FEATURES_DIR/multiregion"
}

resolve_cohort_local_path() {
  local local_dir
  local_dir="$(resolve_cohort_local_dir)"
  echo "$local_dir/$(basename "$COHORT_GCS_PATH")"
}

resolve_cohort_metadata_local_path() {
  local local_dir
  local_dir="$(resolve_cohort_local_dir)"
  echo "$local_dir/$(basename "$(resolve_cohort_metadata_gcs_path)")"
}

resolve_cohort_output_dir() {
  if [[ -n "$COHORT_OUTPUT_DIR" ]]; then
    echo "$COHORT_OUTPUT_DIR"
    return 0
  fi
  local local_dir
  local cohort_filename
  local stem
  local_dir="$(resolve_cohort_local_dir)"
  cohort_filename="$(basename "$COHORT_GCS_PATH")"
  stem="${cohort_filename%.parquet}"
  echo "$local_dir/${stem}_ewm_s2s1"
}

resolve_cohort_output_gcs_uri() {
  local stem
  stem="$(basename "$COHORT_GCS_PATH")"
  stem="${stem%.parquet}"
  echo "$(dirname "$COHORT_GCS_PATH")/${stem}_ewm_s2s1"
}

log_suffix() {
  if [[ "${SHARD_COUNT}" -le 1 ]]; then
    echo ""
    return 0
  fi
  printf "__shard_%02d_of_%02d" "${SHARD_INDEX}" "${SHARD_COUNT}"
}

prepare_repo_data_link() {
  mkdir -p "$LOCAL_DATA_ROOT" "$LOCAL_FEATURES_DIR" "$LOCAL_DERIVED_DIR" "$LOCAL_LOG_DIR"
  if [[ -L "$PROJECT_ROOT/data" ]]; then
    rm -f "$PROJECT_ROOT/data"
  elif [[ -e "$PROJECT_ROOT/data" ]]; then
    local existing_realpath
    existing_realpath="$(realpath "$PROJECT_ROOT/data")"
    local target_realpath
    target_realpath="$(realpath "$LOCAL_DATA_ROOT")"
    if [[ "$existing_realpath" != "$target_realpath" ]]; then
      echo "Refusing to replace existing data directory at $PROJECT_ROOT/data" >&2
      echo "Expected it to be a symlink or already point at $LOCAL_DATA_ROOT" >&2
      exit 1
    fi
    return 0
  fi
  ln -s "$LOCAL_DATA_ROOT" "$PROJECT_ROOT/data"
}

sync_outputs() {
  if in_cohort_mode; then
    if [[ -n "$COHORT_OUTPUT_DIR_RESOLVED" && -d "$COHORT_OUTPUT_DIR_RESOLVED" ]]; then
      storage_rsync "$COHORT_OUTPUT_DIR_RESOLVED" "$(resolve_cohort_output_gcs_uri)" || true
    fi
    if [[ -n "$GCS_PHASE4_URI" && -f "$MATERIALIZE_LOG_PATH" ]]; then
      storage_cp "$MATERIALIZE_LOG_PATH" "$GCS_PHASE4_URI/logs/materialize_cpu$(log_suffix).log" || true
    fi
    return 0
  fi

  local project_data_uri
  project_data_uri="$(resolve_project_data_uri)"

  if [[ -d "$LOCAL_FEATURES_DIR/$BASIN_ID" ]]; then
    storage_rsync "$LOCAL_FEATURES_DIR/$BASIN_ID" "$project_data_uri/features/$BASIN_ID" || true
  fi
  if [[ -d "$LOCAL_DERIVED_DIR/$BASIN_ID" ]]; then
    storage_rsync "$LOCAL_DERIVED_DIR/$BASIN_ID" "$project_data_uri/derived/$BASIN_ID" || true
  fi
  if [[ -n "$GCS_PHASE4_URI" && -f "$MATERIALIZE_LOG_PATH" ]]; then
    storage_cp "$MATERIALIZE_LOG_PATH" "$GCS_PHASE4_URI/logs/materialize_cpu$(log_suffix).log" || true
  fi
}

start_periodic_sync() {
  if [[ "$ENABLE_PERIODIC_SYNC" != "1" ]]; then
    return 0
  fi
  if ! [[ "$SYNC_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$SYNC_INTERVAL_SECONDS" -le 0 ]]; then
    echo "Skipping periodic sync because SYNC_INTERVAL_SECONDS is invalid: $SYNC_INTERVAL_SECONDS" >&2
    return 0
  fi
  (
    while true; do
      sleep "$SYNC_INTERVAL_SECONDS"
      echo "Periodic sync at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
      sync_outputs
    done
  ) &
  PERIODIC_SYNC_PID=$!
}

stop_periodic_sync() {
  if [[ -n "$PERIODIC_SYNC_PID" ]] && kill -0 "$PERIODIC_SYNC_PID" >/dev/null 2>&1; then
    kill "$PERIODIC_SYNC_PID" >/dev/null 2>&1 || true
    wait "$PERIODIC_SYNC_PID" 2>/dev/null || true
  fi
}

cleanup() {
  stop_periodic_sync
  sync_outputs
}

require_cmd gcloud
resolve_python_bin
PROJECT_DATA_URI="$(resolve_project_data_uri)"

trap cleanup EXIT
prepare_repo_data_link

if in_cohort_mode; then
  COHORT_LOCAL_DIR="$(resolve_cohort_local_dir)"
  COHORT_LOCAL_PATH="$(resolve_cohort_local_path)"
  COHORT_METADATA_LOCAL_PATH="$(resolve_cohort_metadata_local_path)"
  COHORT_OUTPUT_DIR_RESOLVED="$(resolve_cohort_output_dir)"

  echo "Staging fixed cohort from $COHORT_GCS_PATH"
  mkdir -p "$COHORT_LOCAL_DIR" "$COHORT_OUTPUT_DIR_RESOLVED"
  storage_cp "$COHORT_GCS_PATH" "$COHORT_LOCAL_PATH"
  storage_cp "$(resolve_cohort_metadata_gcs_path)" "$COHORT_METADATA_LOCAL_PATH"

  if [[ "$SKIP_SETUP_CPU" != "1" ]]; then
    bash "$PROJECT_ROOT/earth_world_model/scripts/setup_phase4_cpu_vm.sh"
  fi

  export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
  export PYTHONUNBUFFERED=1

  cmd=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/scripts/materialize_ewm_s2s1_chips_v1.py"
    --repo-root "$PROJECT_ROOT"
    --cohort-path "$COHORT_LOCAL_PATH"
    --output-dir "$COHORT_OUTPUT_DIR_RESOLVED"
    --imagery-anchor-mode "$IMAGERY_ANCHOR_MODE"
    --anchor-date-column "$ANCHOR_DATE_COLUMN"
    --anchor-year-offset "$ANCHOR_YEAR_OFFSET"
    --sample-year "$SAMPLE_YEAR"
    --s2-cloud-cover-max "$S2_CLOUD_COVER_MAX"
    --max-items-per-window "$MAX_ITEMS_PER_WINDOW"
    --min-s2-clear-fraction "$MIN_S2_CLEAR_FRACTION"
    --min-s1-valid-fraction "$MIN_S1_VALID_FRACTION"
    --shard-index "$SHARD_INDEX"
    --shard-count "$SHARD_COUNT"
  )
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  start_periodic_sync
  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$MATERIALIZE_LOG_PATH"
  exit 0
fi

echo "Staging basin gas tables from $PROJECT_DATA_URI"
mkdir -p "$LOCAL_FEATURES_DIR/$BASIN_ID" "$LOCAL_DERIVED_DIR/$BASIN_ID"
storage_cp "$PROJECT_DATA_URI/features/$BASIN_ID/gas_training_table_v2.csv" "$LOCAL_FEATURES_DIR/$BASIN_ID/gas_training_table_v2.csv"
storage_cp "$PROJECT_DATA_URI/features/$BASIN_ID/gas_training_table_v2_metadata.json" "$LOCAL_FEATURES_DIR/$BASIN_ID/gas_training_table_v2_metadata.json"
if gcloud storage ls "$PROJECT_DATA_URI/derived/$BASIN_ID/gas_prospect_cells_v1.csv" >/dev/null 2>&1; then
  storage_cp "$PROJECT_DATA_URI/derived/$BASIN_ID/gas_prospect_cells_v1.csv" "$LOCAL_DERIVED_DIR/$BASIN_ID/gas_prospect_cells_v1.csv"
fi

if [[ "$SKIP_SETUP_CPU" != "1" ]]; then
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_phase4_cpu_vm.sh"
fi

export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

cmd=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/scripts/materialize_ewm_s2s1_chips_v1.py"
  --repo-root "$PROJECT_ROOT"
  --basin-id "$BASIN_ID"
  --label-column "$LABEL_COLUMN"
  --holdout-quantile "$HOLDOUT_QUANTILE"
  --max-training-wells "$MAX_TRAINING_WELLS"
  --max-grid-cells "$MAX_GRID_CELLS"
  --random-state "$RANDOM_STATE"
  --sample-year "$SAMPLE_YEAR"
  --s2-cloud-cover-max "$S2_CLOUD_COVER_MAX"
  --max-items-per-window "$MAX_ITEMS_PER_WINDOW"
  --min-s2-clear-fraction "$MIN_S2_CLEAR_FRACTION"
  --min-s1-valid-fraction "$MIN_S1_VALID_FRACTION"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
if [[ "$SKIP_GRID" == "1" ]]; then
  cmd+=(--skip-grid)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  cmd+=(--skip-existing)
fi

start_periodic_sync
echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$MATERIALIZE_LOG_PATH"
