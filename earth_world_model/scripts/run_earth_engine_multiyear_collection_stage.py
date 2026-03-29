#!/usr/bin/env python3
"""Run the exact-chip Earth Engine collector year by year for a shared sample set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate year-sharded exact-chip Earth Engine collection for a fixed sample set.",
    )
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--project-id", default="omois-483220")
    parser.add_argument("--locations-path", required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--years", default="", help="Comma-separated years, e.g. 2019,2020,2021.")
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    parser.add_argument("--stage-name-prefix", default="earth_engine_exact_chip_multiyear_v1")
    parser.add_argument("--stage-root-base", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--resolution-meters", type=float, default=10.0)
    parser.add_argument("--region-side-meters", type=float, default=2560.0)
    parser.add_argument("--week-step", type=int, default=4)
    parser.add_argument("--total-weeks", type=int, default=52)
    parser.add_argument("--parallelism", type=int, default=15)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=5.0)
    parser.add_argument("--ee-max-retries", type=int, default=6)
    parser.add_argument("--ee-retry-backoff-sec", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--gcs-bucket", default="")
    parser.add_argument("--gcs-prefix-root", default="earth_engine_interactive_exact_chip_multiyear_v1")
    parser.add_argument("--upload-completed-samples-to-gcs", action="store_true")
    parser.add_argument("--delete-local-after-gcs-upload", action="store_true")
    parser.add_argument("--skip-existing-years", action="store_true")
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def resolve_years(args: argparse.Namespace) -> list[int]:
    years: list[int] = []
    if args.years:
        years.extend(int(token.strip()) for token in str(args.years).split(",") if token.strip())
    if args.start_year and args.end_year:
        if args.end_year < args.start_year:
            raise SystemExit("end-year must be >= start-year")
        years.extend(range(int(args.start_year), int(args.end_year) + 1))
    years = sorted(set(years))
    if not years:
        raise SystemExit("Provide either --years or both --start-year and --end-year.")
    return years


def stage_root_for_year(args: argparse.Namespace, year: int) -> Path:
    if args.stage_root_base:
        return Path(args.stage_root_base) / f"year_{year}"
    mounted_base = Path("/mnt/ewm-data-disk/ee_interactive_scratch")
    if mounted_base.exists() and os.access(mounted_base.parent, os.W_OK):
        default_base = mounted_base / args.stage_name_prefix
    else:
        default_base = Path.home() / "ee_interactive_scratch" / args.stage_name_prefix
    return default_base / f"year_{year}"


def log_path_for_year(args: argparse.Namespace, year: int) -> Path:
    log_dir = Path(args.log_dir) if args.log_dir else (Path.home() / "bench_logs")
    return log_dir / f"{args.stage_name_prefix}_year_{year}.log"


def run_year(args: argparse.Namespace, year: int) -> dict[str, object]:
    stage_name = f"{args.stage_name_prefix}_year_{year}"
    stage_root = stage_root_for_year(args, year)
    log_path = log_path_for_year(args, year)
    stage_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completion_marker = stage_root / "stage_complete.json"
    if args.skip_existing_years and completion_marker.exists():
        payload = json.loads(completion_marker.read_text(encoding="utf-8"))
        payload["status"] = "skipped_existing"
        return payload

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(Path(args.project_root)),
            "MODE": "interactive",
            "STAGE_NAME": stage_name,
            "PROJECT_ID": str(args.project_id),
            "LOCATIONS_PATH": str(args.locations_path),
            "SAMPLE_OFFSET": str(int(args.sample_offset)),
            "SAMPLE_LIMIT": str(int(args.sample_limit)),
            "YEAR": str(int(year)),
            "CHIP_SIZE": str(int(args.chip_size)),
            "RESOLUTION_METERS": str(float(args.resolution_meters)),
            "REGION_SIDE_METERS": str(float(args.region_side_meters)),
            "WEEK_STEP": str(int(args.week_step)),
            "TOTAL_WEEKS": str(int(args.total_weeks)),
            "PARALLELISM": str(int(args.parallelism)),
            "REQUEST_TIMEOUT_SEC": str(float(args.request_timeout_sec)),
            "MAX_RETRIES": str(int(args.max_retries)),
            "RETRY_BACKOFF_SEC": str(float(args.retry_backoff_sec)),
            "EE_MAX_RETRIES": str(int(args.ee_max_retries)),
            "EE_RETRY_BACKOFF_SEC": str(float(args.ee_retry_backoff_sec)),
            "MIN_FREE_GB": str(float(args.min_free_gb)),
            "AUTHENTICATE": "1" if args.authenticate else "0",
            "STAGE_ROOT": str(stage_root),
            "LOG_PATH": str(log_path),
            "PRECHECK": "1",
        }
    )
    if args.gcs_bucket and args.upload_completed_samples_to_gcs:
        env["GCS_BUCKET"] = str(args.gcs_bucket)
        env["GCS_PREFIX"] = f"{args.gcs_prefix_root.strip('/')}/year_{year}"
        env["UPLOAD_COMPLETED_SAMPLES_TO_GCS"] = "1"
        env["DELETE_LOCAL_AFTER_GCS_UPLOAD"] = "1" if args.delete_local_after_gcs_upload else "0"

    started_at = time.perf_counter()
    subprocess.run(
        ["bash", str(SCRIPT_DIR / "run_earth_engine_stage_download.sh")],
        env=env,
        check=True,
    )
    elapsed_sec = float(time.perf_counter() - started_at)
    payload: dict[str, object] = {
        "status": "completed",
        "year": int(year),
        "stage_name": stage_name,
        "stage_root": str(stage_root),
        "log_path": str(log_path),
        "elapsed_sec": elapsed_sec,
        "gcs_prefix": f"gs://{args.gcs_bucket}/{args.gcs_prefix_root.strip('/')}/year_{year}" if args.gcs_bucket and args.upload_completed_samples_to_gcs else None,
    }
    completion_marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    years = resolve_years(args)
    summary: dict[str, object] = {
        "stage": "earth_engine_exact_chip_multiyear_v1",
        "project_id": str(args.project_id),
        "locations_path": str(args.locations_path),
        "sample_offset": int(args.sample_offset),
        "sample_limit": int(args.sample_limit),
        "years": years,
        "runs": [],
    }
    started_at = time.perf_counter()
    for year in years:
        summary["runs"].append(run_year(args, year))
    summary["total_elapsed_sec"] = float(time.perf_counter() - started_at)
    output_path = Path(args.output_summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
