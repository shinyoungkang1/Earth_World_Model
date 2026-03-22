#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:?Set LOCAL_DATA_ROOT, for example /mnt/ewm-data-disk/ssl4eo_zarr_50k}"

export DATA_ACCESS_MODE="${DATA_ACCESS_MODE:-localdisk}"
export SKIP_DATA_SYNC="${SKIP_DATA_SYNC:-1}"
export USE_GCSFUSE_MOUNT="${USE_GCSFUSE_MOUNT:-0}"
export USE_GCS_DATA_DIRECT="${USE_GCS_DATA_DIRECT:-0}"

bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase2_gpu_vm.sh"
