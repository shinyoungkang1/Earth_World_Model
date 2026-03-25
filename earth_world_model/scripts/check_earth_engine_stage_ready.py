#!/usr/bin/env python3
"""Preflight checks for Earth Engine stage launches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ee_stage_utils import (
    check_stage_storage,
    decode_csv_text,
    detect_nul_bytes,
    read_csv_bytes,
    smoke_test_ee,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for Earth Engine download stages.")
    parser.add_argument("--project", required=True, help="Google Cloud project / EE project id.")
    parser.add_argument("--locations-path", required=True, help="CSV used by the stage.")
    parser.add_argument("--stage-root", required=True, help="Scratch or stage root for the run.")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free GiB required on the stage filesystem.")
    parser.add_argument("--authenticate", action="store_true", help="Run interactive ee.Authenticate() before EE init.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.locations_path).expanduser()
    stage_root = Path(args.stage_root).expanduser()
    stage_root.mkdir(parents=True, exist_ok=True)

    csv_bytes = read_csv_bytes(path)
    csv_text = decode_csv_text(csv_bytes, path=path)
    line_count = csv_text.count("\n")
    storage = check_stage_storage(stage_root, min_free_gb=float(args.min_free_gb))
    ee_smoke_value = smoke_test_ee(args.project, authenticate=bool(args.authenticate))

    payload = {
        "project": args.project,
        "locations_path": str(path),
        "stage_root": str(stage_root),
        "csv_nul_count": detect_nul_bytes(csv_bytes),
        "csv_line_count_estimate": line_count,
        "storage": storage,
        "ee_smoke_value": ee_smoke_value,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
