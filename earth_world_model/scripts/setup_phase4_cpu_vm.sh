#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "Missing required command: python3 or python" >&2
    exit 1
  fi
fi

ensure_pip() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  if "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y python3-pip python3-venv
    else
      apt-get update
      apt-get install -y python3-pip python3-venv
    fi
    return 0
  fi
  echo "Unable to bootstrap pip for $PYTHON_BIN" >&2
  exit 1
}

ensure_pip
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/earth_world_model/requirements.txt"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
PY
then
  "$PYTHON_BIN" -m pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu
fi

"$PYTHON_BIN" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
