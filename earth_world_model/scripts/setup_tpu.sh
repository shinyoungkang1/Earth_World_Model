#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_PATH="$PROJECT_ROOT/earth_world_model/requirements.txt"

python -m pip install --upgrade pip
python -m pip install "torch~=2.5.0" "torch_xla[tpu]~=2.5.0" \
  -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install -r "$REQUIREMENTS_PATH"

python - <<'PY'
import torch_xla.core.xla_model as xm
print("TPU device:", xm.xla_device())
PY
