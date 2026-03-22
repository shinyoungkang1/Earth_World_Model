#!/usr/bin/env bash
set -euo pipefail

GCS_RUN_URI="${GCS_RUN_URI:?Set GCS_RUN_URI}"
TRAIN_LOG="${TRAIN_LOG:?Set TRAIN_LOG}"
METRICS_PATH="${METRICS_PATH:?Set METRICS_PATH}"
SUMMARY_PATH="${SUMMARY_PATH:?Set SUMMARY_PATH}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"

copy_if_exists() {
  local source_path="$1"
  local dest_path="$2"
  if [[ -f "$source_path" ]]; then
    gcloud storage cp "$source_path" "$dest_path" >/dev/null 2>&1 || true
  fi
}

while pgrep -f run_phase2_gpu_vm.sh >/dev/null 2>&1 || pgrep -f run_phase2_tpu_vm.sh >/dev/null 2>&1 || pgrep -f train_tpu.py >/dev/null 2>&1; do
  copy_if_exists "$TRAIN_LOG" "$GCS_RUN_URI/train.log"
  copy_if_exists "$METRICS_PATH" "$GCS_RUN_URI/metrics.jsonl"
  copy_if_exists "$SUMMARY_PATH" "$GCS_RUN_URI/training_summary.json"
  sleep "$CHECK_INTERVAL_SEC"
done

copy_if_exists "$TRAIN_LOG" "$GCS_RUN_URI/train.log"
copy_if_exists "$METRICS_PATH" "$GCS_RUN_URI/metrics.jsonl"
copy_if_exists "$SUMMARY_PATH" "$GCS_RUN_URI/training_summary.json"
