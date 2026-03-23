#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_DATAFLUX="${INSTALL_DATAFLUX:-0}"
TORCH_PACKAGE_SPEC="${TORCH_PACKAGE_SPEC:-torch}"
TORCH_WHL_INDEX_URL="${TORCH_WHL_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Missing required command: python or python3" >&2
    exit 1
  fi
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  if "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-pip
  else
    echo "Missing pip and unable to bootstrap it automatically" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/earth_world_model/requirements.txt"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
then
  "$PYTHON_BIN" -m pip install "$TORCH_PACKAGE_SPEC" --index-url "$TORCH_WHL_INDEX_URL"
fi
if [[ "$INSTALL_DATAFLUX" == "1" ]]; then
  "$PYTHON_BIN" -m pip install gcs-torch-dataflux
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"$PYTHON_BIN" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY
