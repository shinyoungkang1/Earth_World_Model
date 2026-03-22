#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=earth_world_model/src:. PYTHONUNBUFFERED=1 python - "$@" <<'PY'
import sys
from earth_world_model.train_tpu import main

if __name__ == "__main__":
    sys.argv = ["train_tpu.py", *sys.argv[1:]]
    main()
PY
