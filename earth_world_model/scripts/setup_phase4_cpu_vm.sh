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
