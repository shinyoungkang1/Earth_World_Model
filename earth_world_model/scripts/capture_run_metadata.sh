#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH}"
RUN_COMMAND="${RUN_COMMAND:?Set RUN_COMMAND}"
SAVE_CODE_SNAPSHOT="${SAVE_CODE_SNAPSHOT:-1}"

mkdir -p "$CHECKPOINT_DIR"

cp "$CONFIG_PATH" "$CHECKPOINT_DIR/config_source.yaml"

python3 - <<'PY' "$CONFIG_PATH" "$CHECKPOINT_DIR/config_resolved.yaml"
import os
import re
import sys
from pathlib import Path

import yaml


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*)\}",
            lambda match: os.environ.get(match.group(1), match.group(2)),
            value,
        )
        return os.path.expanduser(os.path.expandvars(value))
    return value


config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
resolved = expand(config)
output_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
PY

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf '%s\n' "$RUN_COMMAND"
} > "$CHECKPOINT_DIR/launch_command.sh"
chmod 0644 "$CHECKPOINT_DIR/launch_command.sh"

{
  for name in CONFIG_PATH GCS_RUN_URI GCS_DATA_URI LOCAL_RUN_ROOT LOCAL_DATA_ROOT LOCAL_CHECKPOINT_DIR RUN_LOG_PATH DATA_ACCESS_MODE CUDA_DDP_PROCS CONTINUOUS_GCS_SYNC FINAL_GCS_SYNC SAVE_CODE_SNAPSHOT EWM_EXPERIMENT_ID EWM_RUN_LABEL EWM_LAUNCHER_SCRIPT; do
    if [[ -n "${!name:-}" ]]; then
      printf 'export %s=%q\n' "$name" "${!name}"
    fi
  done
  while IFS='=' read -r key value; do
    printf 'export %s=%q\n' "$key" "$value"
  done < <(env | sort | grep '^EWM_' || true)
} > "$CHECKPOINT_DIR/launch_env.sh"
chmod 0644 "$CHECKPOINT_DIR/launch_env.sh"

if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$CHECKPOINT_DIR/git_rev.txt"
  git -C "$PROJECT_ROOT" status --short > "$CHECKPOINT_DIR/git_status.txt" || true
  git -C "$PROJECT_ROOT" diff --binary HEAD -- > "$CHECKPOINT_DIR/git_diff.patch" || true
  git -C "$PROJECT_ROOT" ls-files --others --exclude-standard > "$CHECKPOINT_DIR/git_untracked.txt" || true
else
  printf 'unknown\n' > "$CHECKPOINT_DIR/git_rev.txt"
  printf 'not_a_git_checkout\n' > "$CHECKPOINT_DIR/git_status.txt"
  : > "$CHECKPOINT_DIR/git_diff.patch"
  : > "$CHECKPOINT_DIR/git_untracked.txt"
fi

if [[ "$SAVE_CODE_SNAPSHOT" == "1" ]]; then
  snapshot_paths=(
    "earth_world_model/train_tpu.py"
    "earth_world_model/src"
    "earth_world_model/scripts"
    "earth_world_model/configs"
    "earth_world_model/experiments"
    "earth_world_model/requirements.txt"
  )
  existing_paths=()
  for snapshot_path in "${snapshot_paths[@]}"; do
    if [[ -e "$PROJECT_ROOT/$snapshot_path" ]]; then
      existing_paths+=("$snapshot_path")
    fi
  done
  if (( ${#existing_paths[@]} > 0 )); then
    tar -czf "$CHECKPOINT_DIR/code_snapshot.tar.gz" -C "$PROJECT_ROOT" "${existing_paths[@]}"
  fi
fi
