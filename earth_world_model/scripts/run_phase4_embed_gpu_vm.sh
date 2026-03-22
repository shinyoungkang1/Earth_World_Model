#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_PHASE4_ROOT="${LOCAL_PHASE4_ROOT:-$HOME/ewm_phase4_embed}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$LOCAL_PHASE4_ROOT/data}"
LOCAL_FEATURES_DIR="${LOCAL_FEATURES_DIR:-$LOCAL_DATA_ROOT/features}"
LOCAL_DERIVED_DIR="${LOCAL_DERIVED_DIR:-$LOCAL_DATA_ROOT/derived}"
PROJECT_MODELS_DIR="${PROJECT_MODELS_DIR:-$PROJECT_ROOT/models}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$LOCAL_PHASE4_ROOT/checkpoints}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$LOCAL_PHASE4_ROOT/logs}"
PIPELINE_LOG_PATH="${PIPELINE_LOG_PATH:-$LOCAL_LOG_DIR/pipeline.log}"
GCS_PROJECT_DATA_URI="${GCS_PROJECT_DATA_URI:-}"
GCS_DATA_URI="${GCS_DATA_URI:-}"
GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI, for example gs://my-bucket/earth_world_model/runs/phase2_scale_a100}"
GCS_PHASE4_URI="${GCS_PHASE4_URI:-$GCS_RUN_URI/phase4_ewm_gas}"
COHORT_GCS_PATH="${COHORT_GCS_PATH:-}"
COHORT_METADATA_GCS_PATH="${COHORT_METADATA_GCS_PATH:-}"
COHORT_OUTPUT_DIR="${COHORT_OUTPUT_DIR:-}"
BASIN_ID="${BASIN_ID:-swpa_core_washington_greene}"
CHECKPOINT_BASENAME="${CHECKPOINT_BASENAME:-ewm_best_val.pt}"
LABEL_COLUMN="${LABEL_COLUMN:-label_f12_ge_500000}"
TASK_TYPE="${TASK_TYPE:-regression}"
TARGET_COLUMN="${TARGET_COLUMN:-f12_gas}"
TARGET_TRANSFORM="${TARGET_TRANSFORM:-log1p}"
HOLDOUT_QUANTILE="${HOLDOUT_QUANTILE:-0.8}"
MAX_TRAINING_WELLS="${MAX_TRAINING_WELLS:-1600}"
MAX_GRID_CELLS="${MAX_GRID_CELLS:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
PHASE4_DEVICE="${PHASE4_DEVICE:-cuda}"
SKIP_SETUP_GPU="${SKIP_SETUP_GPU:-0}"
SKIP_SCORING="${SKIP_SCORING:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
COHORT_LOCAL_DIR=""
COHORT_LOCAL_PATH=""
COHORT_METADATA_LOCAL_PATH=""
COHORT_FEATURE_DIR=""
COHORT_RUN_OUTPUT_DIR_RESOLVED=""

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

resolve_cohort_feature_dir() {
  local local_dir
  local cohort_filename
  local stem
  local_dir="$(resolve_cohort_local_dir)"
  cohort_filename="$(basename "$COHORT_GCS_PATH")"
  stem="${cohort_filename%.parquet}"
  echo "$local_dir/${stem}_ewm_s2s1"
}

resolve_cohort_feature_gcs_uri() {
  local stem
  stem="$(basename "$COHORT_GCS_PATH")"
  stem="${stem%.parquet}"
  echo "$(dirname "$COHORT_GCS_PATH")/${stem}_ewm_s2s1"
}

resolve_cohort_run_output_dir() {
  if [[ -n "$COHORT_OUTPUT_DIR" ]]; then
    echo "$COHORT_OUTPUT_DIR"
    return 0
  fi
  local local_dir
  local stem
  local_dir="$(resolve_cohort_local_dir)"
  stem="$(basename "$COHORT_GCS_PATH")"
  stem="${stem%.parquet}"
  echo "$local_dir/${stem}_ewm_phase4"
}

resolve_cohort_run_output_gcs_uri() {
  local stem
  stem="$(basename "$COHORT_GCS_PATH")"
  stem="${stem%.parquet}"
  echo "$GCS_PHASE4_URI/${stem}_ewm_phase4"
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

sync_outputs() {
  if in_cohort_mode; then
    if [[ -n "$COHORT_RUN_OUTPUT_DIR_RESOLVED" && -d "$COHORT_RUN_OUTPUT_DIR_RESOLVED" ]]; then
      storage_rsync "$COHORT_RUN_OUTPUT_DIR_RESOLVED" "$(resolve_cohort_run_output_gcs_uri)" || true
    fi
    if [[ -f "$PIPELINE_LOG_PATH" ]]; then
      storage_cp "$PIPELINE_LOG_PATH" "$GCS_PHASE4_URI/logs/pipeline_gpu.log" || true
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
  if [[ -d "$PROJECT_MODELS_DIR/$BASIN_ID" ]]; then
    storage_rsync "$PROJECT_MODELS_DIR/$BASIN_ID" "$GCS_PHASE4_URI/models/$BASIN_ID" || true
  fi
  if [[ -f "$PIPELINE_LOG_PATH" ]]; then
    storage_cp "$PIPELINE_LOG_PATH" "$GCS_PHASE4_URI/logs/pipeline_gpu.log" || true
  fi
}

require_cmd gcloud
resolve_python_bin
PROJECT_DATA_URI="$(resolve_project_data_uri)"

trap sync_outputs EXIT
prepare_repo_data_link

if in_cohort_mode; then
  COHORT_LOCAL_DIR="$(resolve_cohort_local_dir)"
  COHORT_LOCAL_PATH="$(resolve_cohort_local_path)"
  COHORT_METADATA_LOCAL_PATH="$(resolve_cohort_metadata_local_path)"
  COHORT_FEATURE_DIR="$(resolve_cohort_feature_dir)"
  COHORT_RUN_OUTPUT_DIR_RESOLVED="$(resolve_cohort_run_output_dir)"

  echo "Staging fixed cohort and EWM chips from GCS"
  mkdir -p "$COHORT_LOCAL_DIR" "$COHORT_FEATURE_DIR" "$COHORT_RUN_OUTPUT_DIR_RESOLVED"
  storage_cp "$COHORT_GCS_PATH" "$COHORT_LOCAL_PATH"
  storage_cp "$(resolve_cohort_metadata_gcs_path)" "$COHORT_METADATA_LOCAL_PATH"
  storage_rsync "$(resolve_cohort_feature_gcs_uri)" "$COHORT_FEATURE_DIR"

  if [[ ! -f "$COHORT_FEATURE_DIR/ewm_s2s1_chip_index_v1.parquet" ]] && compgen -G "$COHORT_FEATURE_DIR/ewm_s2s1_chip_index_v1__shard_*.parquet" >/dev/null; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_ewm_s2s1_chip_indexes_v1.py" \
      --repo-root "$PROJECT_ROOT" \
      --features-dir "$COHORT_FEATURE_DIR"
  fi

  LOCAL_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR/$CHECKPOINT_BASENAME"
  echo "Fetching checkpoint $CHECKPOINT_BASENAME from $GCS_RUN_URI/checkpoints"
  storage_cp "$GCS_RUN_URI/checkpoints/$CHECKPOINT_BASENAME" "$LOCAL_CHECKPOINT_PATH"

  if [[ "$SKIP_SETUP_GPU" != "1" ]]; then
    bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"
  fi

  export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
  export PYTHONUNBUFFERED=1

  cmd=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/scripts/run_ewm_gas_pipeline_v2.py"
    --repo-root "$PROJECT_ROOT"
    --cohort-path "$COHORT_LOCAL_PATH"
    --cohort-metadata-path "$COHORT_METADATA_LOCAL_PATH"
    --checkpoint "$LOCAL_CHECKPOINT_PATH"
    --index-path "$COHORT_FEATURE_DIR/ewm_s2s1_chip_index_v1.parquet"
    --output-dir "$COHORT_RUN_OUTPUT_DIR_RESOLVED"
    --task-type "$TASK_TYPE"
    --target-column "$TARGET_COLUMN"
    --target-transform "$TARGET_TRANSFORM"
    --holdout-quantile "$HOLDOUT_QUANTILE"
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
    --random-state "$RANDOM_STATE"
    --device "$PHASE4_DEVICE"
    --s2s1-format raw_ssl4eo
  )
  if [[ "$SKIP_SCORING" == "1" ]]; then
    cmd+=(--skip-scoring)
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$PIPELINE_LOG_PATH"
  exit 0
fi

echo "Staging basin Phase 4 artifacts from $PROJECT_DATA_URI"
mkdir -p "$LOCAL_FEATURES_DIR/$BASIN_ID" "$LOCAL_DERIVED_DIR/$BASIN_ID"
storage_rsync "$PROJECT_DATA_URI/features/$BASIN_ID" "$LOCAL_FEATURES_DIR/$BASIN_ID"
if gcloud storage ls "$PROJECT_DATA_URI/derived/$BASIN_ID/" >/dev/null 2>&1; then
  storage_rsync "$PROJECT_DATA_URI/derived/$BASIN_ID" "$LOCAL_DERIVED_DIR/$BASIN_ID"
fi

if [[ ! -f "$LOCAL_FEATURES_DIR/$BASIN_ID/ewm_s2s1_chip_index_v1.parquet" ]] && compgen -G "$LOCAL_FEATURES_DIR/$BASIN_ID/ewm_s2s1_chip_index_v1__shard_*.parquet" >/dev/null; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_ewm_s2s1_chip_indexes_v1.py" \
    --repo-root "$PROJECT_ROOT" \
    --basin-id "$BASIN_ID"
fi

LOCAL_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR/$CHECKPOINT_BASENAME"
echo "Fetching checkpoint $CHECKPOINT_BASENAME from $GCS_RUN_URI/checkpoints"
storage_cp "$GCS_RUN_URI/checkpoints/$CHECKPOINT_BASENAME" "$LOCAL_CHECKPOINT_PATH"

if [[ "$SKIP_SETUP_GPU" != "1" ]]; then
  bash "$PROJECT_ROOT/earth_world_model/scripts/setup_gpu_vm.sh"
fi

export PYTHONPATH="$PROJECT_ROOT/earth_world_model/src:$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

cmd=(
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
if [[ "$SKIP_SCORING" == "1" ]]; then
  cmd+=(--skip-scoring)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$PIPELINE_LOG_PATH"
