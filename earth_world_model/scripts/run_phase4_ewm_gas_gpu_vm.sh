#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_PHASE4_ROOT="${LOCAL_PHASE4_ROOT:-$HOME/ewm_phase4_gas}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$LOCAL_PHASE4_ROOT/data}"
LOCAL_FEATURES_DIR="${LOCAL_FEATURES_DIR:-$LOCAL_DATA_ROOT/features}"
LOCAL_DERIVED_DIR="${LOCAL_DERIVED_DIR:-$LOCAL_DATA_ROOT/derived}"
PROJECT_MODELS_DIR="${PROJECT_MODELS_DIR:-$PROJECT_ROOT/models}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$LOCAL_PHASE4_ROOT/checkpoints}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$LOCAL_PHASE4_ROOT/logs}"
MATERIALIZE_LOG_PATH="${MATERIALIZE_LOG_PATH:-$LOCAL_LOG_DIR/materialize.log}"
PIPELINE_LOG_PATH="${PIPELINE_LOG_PATH:-$LOCAL_LOG_DIR/pipeline.log}"
GCS_PROJECT_DATA_URI="${GCS_PROJECT_DATA_URI:-}"
GCS_DATA_URI="${GCS_DATA_URI:-}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://my-bucket/earth_world_model/runs/phase2_scale_a100}"
GCS_PHASE4_URI="${GCS_PHASE4_URI:-$GCS_RUN_URI/phase4_ewm_gas}"
BASIN_ID="${BASIN_ID:-swpa_core_washington_greene}"
CHECKPOINT_BASENAME="${CHECKPOINT_BASENAME:-ewm_best_val.pt}"
LABEL_COLUMN="${LABEL_COLUMN:-label_f12_ge_500000}"
HOLDOUT_QUANTILE="${HOLDOUT_QUANTILE:-0.8}"
MAX_TRAINING_WELLS="${MAX_TRAINING_WELLS:-1600}"
MAX_GRID_CELLS="${MAX_GRID_CELLS:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SAMPLE_YEAR="${SAMPLE_YEAR:-2024}"
S2_CLOUD_COVER_MAX="${S2_CLOUD_COVER_MAX:-35.0}"
MAX_ITEMS_PER_WINDOW="${MAX_ITEMS_PER_WINDOW:-200}"
MIN_S2_CLEAR_FRACTION="${MIN_S2_CLEAR_FRACTION:-0.5}"
MIN_S1_VALID_FRACTION="${MIN_S1_VALID_FRACTION:-0.9}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
PHASE4_DEVICE="${PHASE4_DEVICE:-cuda}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
SKIP_GRID="${SKIP_GRID:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

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
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
    return 0
  fi
  echo "Missing required command: python or python3" >&2
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

sync_outputs() {
  local project_data_uri
  project_data_uri="$(resolve_project_data_uri)"

  if [[ -d "$LOCAL_FEATURES_DIR/$BASIN_ID" ]]; then
    storage_rsync "$LOCAL_FEATURES_DIR/$BASIN_ID" "$project_data_uri/features/$BASIN_ID" || true
  fi
  if [[ -d "$LOCAL_DERIVED_DIR/$BASIN_ID" ]]; then
    storage_rsync "$LOCAL_DERIVED_DIR/$BASIN_ID" "$project_data_uri/derived/$BASIN_ID" || true
  fi
  if [[ -d "$PROJECT_MODELS_DIR/$BASIN_ID" ]]; then
    storage_rsync "$PROJECT_MODELS_DIR/$BASIN_ID" "$GCS_PHASE4_URI/models/$BASIN_ID" || true
  fi
  if [[ -f "$MATERIALIZE_LOG_PATH" ]]; then
    storage_cp "$MATERIALIZE_LOG_PATH" "$GCS_PHASE4_URI/logs/materialize.log" || true
  fi
  if [[ -f "$PIPELINE_LOG_PATH" ]]; then
    storage_cp "$PIPELINE_LOG_PATH" "$GCS_PHASE4_URI/logs/pipeline.log" || true
  fi
}

prepare_repo_data_link() {
  mkdir -p "$LOCAL_DATA_ROOT" "$LOCAL_FEATURES_DIR" "$LOCAL_DERIVED_DIR" "$LOCAL_CHECKPOINT_DIR" "$LOCAL_LOG_DIR"
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
  mkdir -p "$PROJECT_MODELS_DIR"
}

require_cmd gcloud
resolve_python_bin
PROJECT_DATA_URI="$(resolve_project_data_uri)"

trap sync_outputs EXIT
prepare_repo_data_link

echo "Staging basin gas tables from $PROJECT_DATA_URI"
mkdir -p "$LOCAL_FEATURES_DIR/$BASIN_ID" "$LOCAL_DERIVED_DIR/$BASIN_ID"
storage_cp "$PROJECT_DATA_URI/features/$BASIN_ID/gas_training_table_v2.csv" "$LOCAL_FEATURES_DIR/$BASIN_ID/gas_training_table_v2.csv"
storage_cp "$PROJECT_DATA_URI/features/$BASIN_ID/gas_training_table_v2_metadata.json" "$LOCAL_FEATURES_DIR/$BASIN_ID/gas_training_table_v2_metadata.json"
if gcloud storage ls "$PROJECT_DATA_URI/derived/$BASIN_ID/gas_prospect_cells_v1.csv" >/dev/null 2>&1; then
  storage_cp "$PROJECT_DATA_URI/derived/$BASIN_ID/gas_prospect_cells_v1.csv" "$LOCAL_DERIVED_DIR/$BASIN_ID/gas_prospect_cells_v1.csv"
fi

LOCAL_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR/$CHECKPOINT_BASENAME"
echo "Fetching checkpoint $CHECKPOINT_BASENAME from $GCS_RUN_URI/checkpoints"
storage_cp "$GCS_RUN_URI/checkpoints/$CHECKPOINT_BASENAME" "$LOCAL_CHECKPOINT_PATH"

if [[ "$SKIP_SETUP_GPU" != "1" ]]; then
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"
fi

export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

materialize_cmd=(
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
)
if [[ "$SKIP_GRID" == "1" ]]; then
  materialize_cmd+=(--skip-grid)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  materialize_cmd+=(--skip-existing)
fi

echo "Running: ${materialize_cmd[*]}"
"${materialize_cmd[@]}" 2>&1 | tee "$MATERIALIZE_LOG_PATH"

pipeline_cmd=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/scripts/run_ewm_gas_pipeline_v1.py"
  --repo-root "$PROJECT_ROOT"
  --basin-id "$BASIN_ID"
  --checkpoint "$LOCAL_CHECKPOINT_PATH"
  --label-column "$LABEL_COLUMN"
  --holdout-quantile "$HOLDOUT_QUANTILE"
  --max-training-wells "$MAX_TRAINING_WELLS"
  --max-grid-cells "$MAX_GRID_CELLS"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --random-state "$RANDOM_STATE"
  --device "$PHASE4_DEVICE"
  --s2s1-format raw_ssl4eo
)
if [[ "$SKIP_GRID" == "1" ]]; then
  pipeline_cmd+=(--skip-scoring)
fi

echo "Running: ${pipeline_cmd[*]}"
"${pipeline_cmd[@]}" 2>&1 | tee "$PIPELINE_LOG_PATH"
