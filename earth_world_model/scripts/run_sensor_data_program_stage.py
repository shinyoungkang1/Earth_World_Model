#!/usr/bin/env python3
"""Build the combined HLS obs_events + decoder-target benchmark data program."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sensor data program: HLS obs_events plus daily decoder targets and benchmark assembly.",
    )
    parser.add_argument("--project", default="")
    parser.add_argument("--locations-path", default="")
    parser.add_argument("--requests-path", default="")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument("--request-sample-id-column", default="sample_id")
    parser.add_argument("--request-date-column", default="date")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--aggregation", choices=("point", "square_mean"), default="point")
    parser.add_argument("--region-side-meters", type=float, default=2560.0)
    parser.add_argument("--products", default="modis,chirps,smap,era5")
    parser.add_argument("--require-sources", default="modis,chirps,smap")
    parser.add_argument("--forecast-horizons-days", default="7,14,30")
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--hls-index-path", default="")
    parser.add_argument("--yearly-root", default="")
    parser.add_argument("--ssl4eo-root", default="")
    parser.add_argument("--yearly-train-index-path", default="")
    parser.add_argument("--yearly-val-index-path", default="")
    parser.add_argument("--hls-max-samples", type=int, default=0)
    parser.add_argument("--hls-min-quality-score", type=float, default=0.0)
    parser.add_argument("--resolve-missing-hls-datetimes-from-chip", action="store_true")
    parser.add_argument("--require-existing-hls-chip-paths", action="store_true")
    parser.add_argument("--require-core-ready", action="store_true")
    parser.add_argument(
        "--stages",
        default="core_check,hls,decoder",
        help="Comma-separated stage set. Allowed: core_check,hls,decoder.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-benchmark-if-exists", action="store_true")
    parser.add_argument(
        "--max-parallel-products",
        type=int,
        default=1,
        help="Maximum number of decoder product extraction subprocesses to run in parallel.",
    )
    parser.add_argument(
        "--profile-json",
        default="",
        help="Optional path for program-level profile metadata JSON. Defaults to <output-root>/status/sensor_data_program_profile.json.",
    )
    return parser.parse_args()


def parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_stages(raw: str) -> list[str]:
    allowed = {"core_check", "hls", "decoder"}
    stages = parse_csv_tokens(raw)
    if not stages:
        raise SystemExit("At least one stage must be selected.")
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise SystemExit(f"Unknown stages {unknown}; allowed values are {sorted(allowed)}")
    ordered: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        if stage not in seen:
            ordered.append(stage)
            seen.add(stage)
    return ordered


def run_command(argv: list[str]) -> None:
    print("Running:", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    stages = parse_stages(args.stages)
    output_root = Path(args.output_root)
    obs_events_dir = output_root / "obs_events"
    targets_dir = output_root / "state_targets_daily"
    benchmarks_dir = output_root / "benchmarks"
    status_dir = output_root / "status"
    obs_events_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    profile_json = Path(args.profile_json) if args.profile_json else (status_dir / "sensor_data_program_profile.json")
    profile_json.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, object] = {
        "stage": "sensor_data_program_v1",
        "selected_stages": stages,
        "max_parallel_products": int(max(1, args.max_parallel_products)),
        "stages": {},
    }

    if "core_check" in stages:
        stage_started = time.perf_counter()
        core_status_json = status_dir / "core_observation_status.json"
        if not (args.skip_existing and core_status_json.exists()):
            core_argv = [
                sys.executable,
                str(SCRIPT_DIR / "inspect_core_observation_assets.py"),
                "--yearly-root", args.yearly_root,
                "--ssl4eo-root", args.ssl4eo_root,
                "--yearly-train-index-path", args.yearly_train_index_path,
                "--yearly-val-index-path", args.yearly_val_index_path,
                "--output-json", str(core_status_json),
            ]
            if args.require_core_ready:
                core_argv.append("--require-ready")
            run_command(core_argv)
            profile["stages"]["core_check"] = {
                "status": "completed",
                "elapsed_sec": float(time.perf_counter() - stage_started),
                "status_json": str(core_status_json),
            }
        else:
            print("Skipping core observation check: existing status found.", flush=True)
            profile["stages"]["core_check"] = {
                "status": "skipped_existing",
                "elapsed_sec": 0.0,
                "status_json": str(core_status_json),
            }

    if "hls" in stages:
        stage_started = time.perf_counter()
        if not args.hls_index_path:
            raise SystemExit("--hls-index-path is required when stages include hls")
        hls_parquet = obs_events_dir / "hls_obs_events_v1.parquet"
        hls_metadata = obs_events_dir / "hls_obs_events_v1_metadata.json"
        if not (args.skip_existing and hls_parquet.exists() and hls_metadata.exists()):
            hls_argv = [
                sys.executable,
                str(SCRIPT_DIR / "build_hls_obs_events.py"),
                "--hls-index-path", args.hls_index_path,
                "--output-parquet", str(hls_parquet),
                "--output-metadata-json", str(hls_metadata),
                "--max-samples", str(int(args.hls_max_samples)),
                "--min-quality-score", str(float(args.hls_min_quality_score)),
            ]
            if args.resolve_missing_hls_datetimes_from_chip:
                hls_argv.append("--resolve-missing-datetimes-from-chip")
            if args.require_existing_hls_chip_paths:
                hls_argv.append("--require-existing-chip-paths")
            run_command(hls_argv)
            profile["stages"]["hls"] = {
                "status": "completed",
                "elapsed_sec": float(time.perf_counter() - stage_started),
                "output_parquet": str(hls_parquet),
                "output_metadata_json": str(hls_metadata),
            }
        else:
            print("Skipping HLS obs_events: existing outputs found.", flush=True)
            profile["stages"]["hls"] = {
                "status": "skipped_existing",
                "elapsed_sec": 0.0,
                "output_parquet": str(hls_parquet),
                "output_metadata_json": str(hls_metadata),
            }

    if "decoder" in stages:
        stage_started = time.perf_counter()
        if not args.project:
            raise SystemExit("--project is required when stages include decoder")
        if not args.locations_path:
            raise SystemExit("--locations-path is required when stages include decoder")
        decoder_argv = [
            sys.executable,
            str(SCRIPT_DIR / "run_decoder_target_benchmark_stage.py"),
            "--project", args.project,
            "--locations-path", args.locations_path,
            "--sample-id-column", args.sample_id_column,
            "--latitude-column", args.latitude_column,
            "--longitude-column", args.longitude_column,
            "--request-sample-id-column", args.request_sample_id_column,
            "--request-date-column", args.request_date_column,
            "--batch-size", str(int(args.batch_size)),
            "--max-samples", str(int(args.max_samples)),
            "--max-requests", str(int(args.max_requests)),
            "--aggregation", args.aggregation,
            "--region-side-meters", str(float(args.region_side_meters)),
            "--products", args.products,
            "--require-sources", args.require_sources,
            "--forecast-horizons-days", args.forecast_horizons_days,
            "--output-dir", str(targets_dir),
            "--benchmark-output-dir", str(benchmarks_dir),
            "--max-parallel-products", str(int(args.max_parallel_products)),
            "--profile-json", str(status_dir / "decoder_target_stage_profile.json"),
        ]
        if args.requests_path:
            decoder_argv.extend(["--requests-path", args.requests_path])
        else:
            decoder_argv.extend(["--start-date", args.start_date, "--end-date", args.end_date])
        if args.authenticate:
            decoder_argv.append("--authenticate")
        if args.skip_existing:
            decoder_argv.append("--skip-existing")
        if args.skip_benchmark_if_exists:
            decoder_argv.append("--skip-benchmark-if-exists")
        run_command(decoder_argv)
        profile["stages"]["decoder"] = {
            "status": "completed",
            "elapsed_sec": float(time.perf_counter() - stage_started),
            "targets_dir": str(targets_dir),
            "benchmarks_dir": str(benchmarks_dir),
            "profile_json": str(status_dir / "decoder_target_stage_profile.json"),
        }

    profile["total_elapsed_sec"] = float(time.perf_counter() - started_at)
    profile_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
