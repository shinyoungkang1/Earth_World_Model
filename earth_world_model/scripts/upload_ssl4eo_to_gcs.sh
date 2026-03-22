#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <local_ssl4eo_root> <gcs_data_uri> [phase2_seed_checkpoint] [gcs_run_uri]" >&2
  echo "Example: $0 data/raw/ssl4eo_zarr_minimal gs://my-bucket/earth_world_model/data/raw/ssl4eo_zarr_minimal earth_world_model/checkpoints/ssl4eo_zarr_trainval_phase2_scale_cuda/ewm_epoch_001.pt gs://my-bucket/earth_world_model/runs/phase2_scale" >&2
  exit 1
fi

LOCAL_SSL4EO_ROOT="$1"
GCS_DATA_URI="$2"
PHASE2_SEED_CHECKPOINT="${3:-}"
GCS_RUN_URI="${4:-}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required but not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_SSL4EO_ROOT" ]]; then
  echo "Local SSL4EO root not found: $LOCAL_SSL4EO_ROOT" >&2
  exit 1
fi

echo "Uploading dataset to $GCS_DATA_URI"
gcloud storage rsync --recursive --exclude='(^|.*/)\.cache($|/.*)' "$LOCAL_SSL4EO_ROOT" "$GCS_DATA_URI"

if [[ -n "$PHASE2_SEED_CHECKPOINT" ]]; then
    if [[ ! -f "$PHASE2_SEED_CHECKPOINT" ]]; then
        echo "Seed checkpoint not found: $PHASE2_SEED_CHECKPOINT" >&2
        exit 1
    fi
  if [[ -z "$GCS_RUN_URI" ]]; then
    GCS_PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$GCS_DATA_URI")")")"
    GCS_RUN_URI="$GCS_PROJECT_ROOT/runs/phase2_scale"
  fi
  echo "Uploading seed checkpoint to $GCS_RUN_URI/checkpoints/"
  gcloud storage cp "$PHASE2_SEED_CHECKPOINT" "$GCS_RUN_URI/checkpoints/"
fi
