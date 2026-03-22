#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE2_SKIP_SETUP="${SKIP_SETUP_TPU:-0}"

bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase2_tpu_vm.sh"

export SKIP_SETUP_TPU=1
bash "$PROJECT_ROOT/earth_world_model/scripts/run_phase3_probe_tpu_vm.sh"

export SKIP_SETUP_TPU="$PHASE2_SKIP_SETUP"
