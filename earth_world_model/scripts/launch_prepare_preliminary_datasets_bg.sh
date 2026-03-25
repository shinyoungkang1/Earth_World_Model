#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$HOME/bench_logs}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
RUN_TAG="${RUN_TAG:-${YEARLY_NAME}_${SSL4EO_NAME}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/prep_${RUN_TAG}.log}"
PID_PATH="${PID_PATH:-$LOG_DIR/prep_${RUN_TAG}.pid}"

mkdir -p "$LOG_DIR"

nohup bash "$PROJECT_ROOT/earth_world_model/scripts/run_prepare_preliminary_datasets_cpu_vm.sh" \
  >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_PATH"
echo "PID:$PID"
echo "LOG:$LOG_PATH"
echo "PIDFILE:$PID_PATH"
