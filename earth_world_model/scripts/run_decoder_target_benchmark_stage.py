#!/usr/bin/env python3
"""Run all daily decoder-target extractions and assemble the aligned benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract daily decoder targets from GEE and build decoder_target_benchmark_v1.",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--locations-path", required=True)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark-output-dir",
        default="",
        help="Optional separate output directory for the merged benchmark artifacts. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--max-parallel-products",
        type=int,
        default=1,
        help="Maximum number of decoder product extraction subprocesses to run in parallel.",
    )
    parser.add_argument(
        "--profile-json",
        default="",
        help="Optional path for stage timing/profile metadata JSON. Defaults to <output-dir>/decoder_target_stage_profile.json.",
    )
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument(
        "--products",
        default="modis,chirps,smap,era5",
        help="Comma-separated product set to run. Allowed: modis,chirps,smap,era5.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip product extraction when both parquet and metadata already exist.",
    )
    parser.add_argument(
        "--skip-benchmark-if-exists",
        action="store_true",
        help="Skip benchmark assembly if both benchmark outputs already exist.",
    )
    parser.add_argument(
        "--forecast-horizons-days",
        default="7,14,30",
        help="Comma-separated forecasting horizons for benchmark assembly.",
    )
    parser.add_argument(
        "--require-sources",
        default="modis,chirps,smap",
        help="Comma-separated primary sources required in the final benchmark rows.",
    )
    return parser.parse_args()


def parse_products(raw: str) -> list[str]:
    allowed = ("modis", "chirps", "smap", "era5")
    products = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not products:
        raise SystemExit("At least one product must be selected.")
    unknown = [token for token in products if token not in allowed]
    if unknown:
        raise SystemExit(f"Unknown products {unknown}; allowed values are {list(allowed)}")
    # Preserve input order while deduplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for token in products:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
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
    ]
    if args.requests_path:
        common.extend(["--requests-path", args.requests_path])
    else:
        common.extend(["--start-date", args.start_date, "--end-date", args.end_date])
    if args.authenticate:
        common.append("--authenticate")
    return common


def run_command(argv: list[str]) -> None:
    print("Running:", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def run_timed_command(argv: list[str]) -> dict[str, float | int | str]:
    started = time.perf_counter()
    run_command(argv)
    ended = time.perf_counter()
    return {
        "elapsed_sec": float(ended - started),
    }


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output_dir = Path(args.benchmark_output_dir) if args.benchmark_output_dir else output_dir
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    profile_json = Path(args.profile_json) if args.profile_json else (output_dir / "decoder_target_stage_profile.json")
    profile_json.parent.mkdir(parents=True, exist_ok=True)

    common = build_common_args(args)
    product_defs = {
        "modis": SCRIPT_DIR / "extract_modis_lst_daily_targets.py",
        "chirps": SCRIPT_DIR / "extract_chirps_daily_targets.py",
        "smap": SCRIPT_DIR / "extract_smap_daily_targets.py",
        "era5": SCRIPT_DIR / "extract_era5_land_daily_targets.py",
    }
    selected_products = parse_products(args.products)
    outputs: dict[str, Path] = {}
    metadata_outputs: dict[str, Path] = {}
    product_jobs: list[tuple[str, list[str], Path, Path]] = []
    profile: dict[str, object] = {
        "stage": "decoder_target_benchmark_v1",
        "selected_products": selected_products,
        "max_parallel_products": int(max(1, args.max_parallel_products)),
        "product_runs": {},
    }
    for product_name in selected_products:
        script_path = product_defs[product_name]
        parquet_path = output_dir / f"{product_name}_daily.parquet"
        metadata_path = output_dir / f"{product_name}_daily_metadata.json"
        outputs[product_name] = parquet_path
        metadata_outputs[product_name] = metadata_path
        if args.skip_existing and parquet_path.exists() and metadata_path.exists():
            print(f"Skipping {product_name}: existing outputs found.", flush=True)
            profile["product_runs"][product_name] = {
                "status": "skipped_existing",
                "output_parquet": str(parquet_path),
                "output_metadata_json": str(metadata_path),
            }
            continue
        product_jobs.append(
            (
                product_name,
                [
                    sys.executable,
                    str(script_path),
                    *common,
                    "--output-parquet", str(parquet_path),
                    "--output-metadata-json", str(metadata_path),
                ],
                parquet_path,
                metadata_path,
            )
        )

    max_workers = max(1, int(args.max_parallel_products))
    if product_jobs:
        if max_workers == 1 or len(product_jobs) == 1:
            for product_name, argv, parquet_path, metadata_path in product_jobs:
                result = run_timed_command(argv)
                profile["product_runs"][product_name] = {
                    "status": "completed",
                    "elapsed_sec": result["elapsed_sec"],
                    "output_parquet": str(parquet_path),
                    "output_metadata_json": str(metadata_path),
                }
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(product_jobs))) as executor:
                future_to_job = {
                    executor.submit(run_timed_command, argv): (product_name, parquet_path, metadata_path)
                    for product_name, argv, parquet_path, metadata_path in product_jobs
                }
                for future in as_completed(future_to_job):
                    product_name, parquet_path, metadata_path = future_to_job[future]
                    result = future.result()
                    profile["product_runs"][product_name] = {
                        "status": "completed",
                        "elapsed_sec": result["elapsed_sec"],
                        "output_parquet": str(parquet_path),
                        "output_metadata_json": str(metadata_path),
                    }

    benchmark_parquet = benchmark_output_dir / "decoder_target_benchmark_v1.parquet"
    benchmark_metadata = benchmark_output_dir / "decoder_target_benchmark_v1_metadata.json"
    if args.skip_benchmark_if_exists and benchmark_parquet.exists() and benchmark_metadata.exists():
        print("Skipping benchmark assembly: existing benchmark outputs found.", flush=True)
        profile["benchmark"] = {
            "status": "skipped_existing",
            "output_parquet": str(benchmark_parquet),
            "output_metadata_json": str(benchmark_metadata),
        }
        profile["total_elapsed_sec"] = float(time.perf_counter() - started_at)
        profile_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return

    selected_output_args: list[str] = []
    if "modis" in outputs:
        selected_output_args.extend(["--modis-path", str(outputs["modis"])])
    if "chirps" in outputs:
        selected_output_args.extend(["--chirps-path", str(outputs["chirps"])])
    if "smap" in outputs:
        selected_output_args.extend(["--smap-path", str(outputs["smap"])])
    if "era5" in outputs:
        selected_output_args.extend(["--era5-path", str(outputs["era5"])])
    benchmark_started = time.perf_counter()
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_decoder_target_benchmark.py"),
            *selected_output_args,
            "--require-sources", args.require_sources,
            "--forecast-horizons-days", args.forecast_horizons_days,
            "--output-parquet", str(benchmark_parquet),
            "--output-metadata-json", str(benchmark_metadata),
        ]
    )
    profile["benchmark"] = {
        "status": "completed",
        "elapsed_sec": float(time.perf_counter() - benchmark_started),
        "output_parquet": str(benchmark_parquet),
        "output_metadata_json": str(benchmark_metadata),
    }
    profile["total_elapsed_sec"] = float(time.perf_counter() - started_at)
    profile_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
