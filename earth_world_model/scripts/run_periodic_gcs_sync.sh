#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <local_dir> <gcs_uri> <pid_pattern> [interval_seconds] [log_path] [pidfile_path]" >&2
  exit 1
fi

LOCAL_DIR="$1"
GCS_URI="$2"
PID_PATTERN="$3"
INTERVAL_SECONDS="${4:-180}"
LOG_PATH="${5:-$HOME/phase4_periodic_gcs_sync.log}"
PIDFILE_PATH="${6:-$HOME/phase4_periodic_gcs_sync.pid}"

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SECONDS" -le 0 ]]; then
  echo "Invalid interval: $INTERVAL_SECONDS" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$PIDFILE_PATH")"

if [[ -f "$PIDFILE_PATH" ]]; then
  existing_pid="$(cat "$PIDFILE_PATH" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    echo "already_running pid=$existing_pid"
    exit 0
  fi
fi

(
  while pgrep -f -- "$PID_PATTERN" >/dev/null 2>&1; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) sync-start local_dir=$LOCAL_DIR gcs_uri=$GCS_URI"
    gcloud storage rsync --recursive "$LOCAL_DIR" "$GCS_URI" || true
    sleep "$INTERVAL_SECONDS"
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final-sync local_dir=$LOCAL_DIR gcs_uri=$GCS_URI"
  gcloud storage rsync --recursive "$LOCAL_DIR" "$GCS_URI" || true
) >>"$LOG_PATH" 2>&1 < /dev/null &

sync_pid=$!
echo "$sync_pid" > "$PIDFILE_PATH"
echo "started pid=$sync_pid log=$LOG_PATH pidfile=$PIDFILE_PATH"
