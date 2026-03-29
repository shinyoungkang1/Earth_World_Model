#!/usr/bin/env python3
"""Run regional multiscale Earth Engine collection across years and request roles."""

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
        description="Orchestrate year-sharded regional multiscale Earth Engine collection.",
    )
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--project-id", default="omois-483220")
    parser.add_argument("--local-requests-path", required=True)
    parser.add_argument("--regional-requests-path", required=True)
    parser.add_argument("--years", default="2019,2020,2021")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--local-parallelism", type=int, default=6)
    parser.add_argument("--regional-parallelism", type=int, default=6)
    parser.add_argument("--week-step", type=int, default=4)
    parser.add_argument("--total-weeks", type=int, default=52)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=5.0)
    parser.add_argument("--ee-max-retries", type=int, default=6)
    parser.add_argument("--ee-retry-backoff-sec", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--stage-name-prefix", default="earth_engine_regional_multiscale_v1")
    parser.add_argument("--stage-root-base", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--gcs-bucket", default="")
    parser.add_argument("--gcs-prefix-root", default="earth_engine_regional_multiscale_v1")
    parser.add_argument("--upload-completed-samples-to-gcs", action="store_true")
    parser.add_argument("--delete-local-after-gcs-upload", action="store_true")
    parser.add_argument("--skip-existing-years", action="store_true")
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def resolve_years(arg: str) -> list[int]:
    years = sorted({int(token.strip()) for token in str(arg).split(",") if token.strip()})
    if not years:
        raise SystemExit("Provide at least one year via --years")
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


def log_root(args: argparse.Namespace) -> Path:
    return Path(args.log_dir) if args.log_dir else (Path.home() / "bench_logs")


def run_role(
    *,
    args: argparse.Namespace,
    year: int,
    role_name: str,
    requests_path: str,
    parallelism: int,
    stage_root: Path,
    log_dir: Path,
) -> dict[str, object]:
    role_stage_name = f"{args.stage_name_prefix}_{role_name}_year_{year}"
    role_root = stage_root / role_name
    role_root.mkdir(parents=True, exist_ok=True)
    role_log_path = log_dir / f"{role_stage_name}.log"
    completion_marker = role_root / "stage_complete.json"
    if args.skip_existing_years and completion_marker.exists():
        payload = json.loads(completion_marker.read_text(encoding="utf-8"))
        payload["status"] = "skipped_existing"
        return payload

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(Path(args.project_root)),
            "STAGE_NAME": role_stage_name,
            "PROJECT_ID": str(args.project_id),
            "REQUESTS_PATH": str(requests_path),
            "SAMPLE_OFFSET": str(int(args.sample_offset)),
            "SAMPLE_LIMIT": str(int(args.sample_limit)),
            "YEAR": str(int(year)),
            "DEFAULT_CHIP_SIZE": "256",
            "DEFAULT_RESOLUTION_METERS": "10",
            "DEFAULT_REGION_SIDE_METERS": "2560",
            "WEEK_STEP": str(int(args.week_step)),
            "TOTAL_WEEKS": str(int(args.total_weeks)),
            "PARALLELISM": str(int(parallelism)),
            "REQUEST_TIMEOUT_SEC": str(float(args.request_timeout_sec)),
            "MAX_RETRIES": str(int(args.max_retries)),
            "RETRY_BACKOFF_SEC": str(float(args.retry_backoff_sec)),
            "EE_MAX_RETRIES": str(int(args.ee_max_retries)),
            "EE_RETRY_BACKOFF_SEC": str(float(args.ee_retry_backoff_sec)),
            "MIN_FREE_GB": str(float(args.min_free_gb)),
            "AUTHENTICATE": "1" if args.authenticate else "0",
            "STAGE_ROOT": str(role_root),
            "LOG_PATH": str(role_log_path),
            "PRECHECK": "1",
        }
    )
    if args.gcs_bucket and args.upload_completed_samples_to_gcs:
        env["GCS_BUCKET"] = str(args.gcs_bucket)
        env["GCS_PREFIX"] = f"{args.gcs_prefix_root.strip('/')}/year_{year}/{role_name}"
        env["UPLOAD_COMPLETED_SAMPLES_TO_GCS"] = "1"
        env["DELETE_LOCAL_AFTER_GCS_UPLOAD"] = "1" if args.delete_local_after_gcs_upload else "0"

    started_at = time.perf_counter()
    subprocess.run(
        ["bash", str(SCRIPT_DIR / "run_earth_engine_grouped_stage_download.sh")],
        env=env,
        check=True,
    )
    payload: dict[str, object] = {
        "status": "completed",
        "year": int(year),
        "role_name": role_name,
        "stage_name": role_stage_name,
        "stage_root": str(role_root),
        "log_path": str(role_log_path),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "requests_path": str(requests_path),
        "gcs_prefix": f"gs://{args.gcs_bucket}/{args.gcs_prefix_root.strip('/')}/year_{year}/{role_name}"
        if args.gcs_bucket and args.upload_completed_samples_to_gcs
        else None,
    }
    completion_marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_year(args: argparse.Namespace, year: int) -> dict[str, object]:
    stage_root = stage_root_for_year(args, year)
    stage_root.mkdir(parents=True, exist_ok=True)
    logs = log_root(args)
    logs.mkdir(parents=True, exist_ok=True)
    year_marker = stage_root / "year_complete.json"
    if args.skip_existing_years and year_marker.exists():
        payload = json.loads(year_marker.read_text(encoding="utf-8"))
        payload["status"] = "skipped_existing"
        return payload

    year_payload: dict[str, object] = {
        "year": int(year),
        "stage_root": str(stage_root),
        "roles": [
            run_role(
                args=args,
                year=year,
                role_name="local_grid",
                requests_path=str(args.local_requests_path),
                parallelism=int(args.local_parallelism),
                stage_root=stage_root,
                log_dir=logs,
            ),
            run_role(
                args=args,
                year=year,
                role_name="regional_context",
                requests_path=str(args.regional_requests_path),
                parallelism=int(args.regional_parallelism),
                stage_root=stage_root,
                log_dir=logs,
            ),
        ],
    }
    year_marker.write_text(json.dumps(year_payload, indent=2), encoding="utf-8")
    return year_payload


def main() -> None:
    args = parse_args()
    years = resolve_years(args.years)
    summary: dict[str, object] = {
        "stage": "earth_engine_regional_multiscale_v1",
        "project_id": str(args.project_id),
        "years": years,
        "local_requests_path": str(args.local_requests_path),
        "regional_requests_path": str(args.regional_requests_path),
        "sample_offset": int(args.sample_offset),
        "sample_limit": int(args.sample_limit),
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
