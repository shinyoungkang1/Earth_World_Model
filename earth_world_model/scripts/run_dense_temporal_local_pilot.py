#!/usr/bin/env python3
"""Run a local weekly one-year dense temporal pilot.

This is the first non-smoke pilot entrypoint for Planetary Computer backed
location-year sequence generation. It intentionally keeps the workflow simple:

1. plan weekly location-year sequences
2. materialize full-year sequence tensors locally
3. keep both paired and unpaired weeks so cloud handling can be inspected
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

PLANETARY_COMPUTER_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PLANETARY_COMPUTER_S2_COLLECTION = "sentinel-2-l2a"
PLANETARY_COMPUTER_S1_COLLECTION = "sentinel-1-grd"
PLANETARY_COMPUTER_S2_REQUIRED_ASSETS = "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL"
PLANETARY_COMPUTER_S1_REQUIRED_ASSETS = "VV,VH"
PLANETARY_COMPUTER_S2_BAND_ASSETS = "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12"
PLANETARY_COMPUTER_S1_BAND_ASSETS = "VV,VH"
PLANETARY_COMPUTER_S2_SCL_ASSET = "SCL"
DEFAULT_LOCATIONS_PATH = (
    REPO_ROOT / "data" / "raw" / "dense_temporal_seed_locations_pilot10.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "dense_temporal_planetary_local10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 10-sequence local weekly one-year dense temporal pilot against Microsoft Planetary Computer."
    )
    parser.add_argument("--locations-path", default=str(DEFAULT_LOCATIONS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--years", default="2020")
    parser.add_argument("--cadence-days", type=int, default=7)
    parser.add_argument("--sequence-limit", type=int, default=10)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--grid-size-deg", type=float, default=1.0)
    parser.add_argument("--search-padding-deg", type=float, default=0.10)
    parser.add_argument("--s2-cloud-cover-max", type=float, default=80.0)
    parser.add_argument("--max-items-per-search", type=int, default=2000)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--output-format", choices=["npz", "zarr", "both"], default="both")
    parser.add_argument("--min-paired-bins", type=int, default=1)
    parser.add_argument("--min-materialized-paired-bins", type=int, default=1)
    parser.add_argument("--max-bins-per-sequence", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
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
        "--grid-size-deg",
        str(args.grid_size_deg),
        "--search-padding-deg",
        str(args.search_padding_deg),
        "--stac-api-url",
        PLANETARY_COMPUTER_STAC_API_URL,
        "--s2-collection",
        PLANETARY_COMPUTER_S2_COLLECTION,
        "--s1-collection",
        PLANETARY_COMPUTER_S1_COLLECTION,
        "--s2-required-assets",
        PLANETARY_COMPUTER_S2_REQUIRED_ASSETS,
        "--s1-required-assets",
        PLANETARY_COMPUTER_S1_REQUIRED_ASSETS,
        "--s2-cloud-cover-max",
        str(args.s2_cloud_cover_max),
        "--max-items-per-search",
        str(args.max_items_per_search),
    ]
    if args.group_column:
        planner_cmd.extend(["--group-column", str(args.group_column)])
    run_command(planner_cmd)

    if args.plan_only:
        return 0

    materializer_cmd = [
        sys.executable,
        str(MATERIALIZER_PATH),
        "--plan-path",
        str(plan_dir / "sequence_plan.parquet"),
        "--output-dir",
        str(materialized_dir),
        "--stac-api-url",
        PLANETARY_COMPUTER_STAC_API_URL,
        "--signing-mode",
        "planetary_computer",
        "--pixel-access-mode",
        "direct",
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
        "--s2-band-assets",
        PLANETARY_COMPUTER_S2_BAND_ASSETS,
        "--s1-band-assets",
        PLANETARY_COMPUTER_S1_BAND_ASSETS,
        "--s2-scl-asset",
        PLANETARY_COMPUTER_S2_SCL_ASSET,
    ]
    if args.skip_existing:
        materializer_cmd.append("--skip-existing")
    run_command(materializer_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
