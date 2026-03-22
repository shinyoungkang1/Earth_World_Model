#!/usr/bin/env python3
"""Run a tiny live STAC smoke test for the dense temporal pipeline.

This script orchestrates:

1. sequence planning against a live STAC API
2. dense temporal materialization for a very small number of paired bins

It is intentionally conservative so we can validate the full path before doing
larger weekly corpus builds.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = REPO_ROOT / "earth_world_model" / "scripts" / "plan_dense_temporal_sequences.py"
MATERIALIZER_PATH = REPO_ROOT / "earth_world_model" / "scripts" / "materialize_dense_temporal_sequences.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny live dense temporal STAC smoke test.")
    parser.add_argument("--locations-path", required=True, help="CSV or parquet with sample_id, latitude, longitude.")
    parser.add_argument("--output-dir", required=True, help="Output directory for plan + materialization artifacts.")
    parser.add_argument("--years", default="2024")
    parser.add_argument("--cadence-days", type=int, default=7)
    parser.add_argument("--sequence-limit", type=int, default=2)
    parser.add_argument("--max-bins-per-sequence", type=int, default=4)
    parser.add_argument("--include-unpaired-bins", action="store_true")
    parser.add_argument("--output-format", choices=["npz", "zarr", "both"], default="both")
    parser.add_argument("--signing-mode", choices=["none", "planetary_computer"], default="none")
    parser.add_argument("--stac-api-url", default="https://stac.dataspace.copernicus.eu/v1")
    parser.add_argument("--pixel-access-mode", choices=["direct", "cdse_process"], default="direct")
    parser.add_argument("--cdse-process-url", default="https://sh.dataspace.copernicus.eu/api/v1/process")
    parser.add_argument(
        "--cdse-token-url",
        default="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )
    parser.add_argument("--cdse-resolution-m", type=float, default=10.0)
    parser.add_argument("--cdse-request-timeout", type=int, default=120)
    parser.add_argument("--fuse-s1-s2-per-bin", action="store_true")
    parser.add_argument("--s2-cloud-cover-max", type=float, default=80.0)
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--min-paired-bins", type=int, default=1)
    parser.add_argument("--min-materialized-paired-bins", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run-only", action="store_true")
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(arg) for arg in args), flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plan_dir = output_dir / "plan"
    materialized_dir = output_dir / "materialized"
    plan_dir.mkdir(parents=True, exist_ok=True)
    materialized_dir.mkdir(parents=True, exist_ok=True)

    planner_cmd = [
        sys.executable,
        str(PLANNER_PATH),
        "--locations-path",
        str(args.locations_path),
        "--output-dir",
        str(plan_dir),
        "--sample-id-column",
        "sample_id",
        "--latitude-column",
        "latitude",
        "--longitude-column",
        "longitude",
        "--years",
        str(args.years),
        "--cadence-days",
        str(args.cadence_days),
        "--sequence-limit",
        str(args.sequence_limit),
        "--stac-api-url",
        str(args.stac_api_url),
        "--s2-cloud-cover-max",
        str(args.s2_cloud_cover_max),
    ]
    run_command(planner_cmd)

    if args.dry_run_only:
        return 0

    materializer_cmd = [
        sys.executable,
        str(MATERIALIZER_PATH),
        "--plan-path",
        str(plan_dir / "sequence_plan.parquet"),
        "--output-dir",
        str(materialized_dir),
        "--stac-api-url",
        str(args.stac_api_url),
        "--pixel-access-mode",
        str(args.pixel_access_mode),
        "--cdse-process-url",
        str(args.cdse_process_url),
        "--cdse-token-url",
        str(args.cdse_token_url),
        "--cdse-resolution-m",
        str(args.cdse_resolution_m),
        "--cdse-request-timeout",
        str(args.cdse_request_timeout),
        "--signing-mode",
        str(args.signing_mode),
        "--chip-size",
        str(args.chip_size),
        "--sequence-limit",
        str(args.sequence_limit),
        "--min-paired-bins",
        str(args.min_paired_bins),
        "--min-materialized-paired-bins",
        str(args.min_materialized_paired_bins),
        "--max-bins-per-sequence",
        str(args.max_bins_per_sequence),
        "--output-format",
        str(args.output_format),
    ]
    if not args.include_unpaired_bins:
        materializer_cmd.append("--paired-bins-only")
    if args.fuse_s1_s2_per_bin:
        materializer_cmd.append("--fuse-s1-s2-per-bin")
    if args.skip_existing:
        materializer_cmd.append("--skip-existing")
    run_command(materializer_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
