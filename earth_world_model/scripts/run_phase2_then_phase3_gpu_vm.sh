#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE2_SKIP_SETUP="${SKIP_SETUP_GPU:-0}"

bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase2_gpu_vm.sh"

export SKIP_SETUP_GPU=1
bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase3_probe_gpu_vm.sh"

export SKIP_SETUP_GPU="$PHASE2_SKIP_SETUP"
