#!/usr/bin/env python3
"""Download grouped exact-chip Earth Engine stacks for multiscale regional requests."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ee_stage_utils import initialize_ee, require_ee
import run_earth_engine_exact_chip_interactive_benchmark as single


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pandas is required for grouped Earth Engine collection.") from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped exact-chip Earth Engine interactive downloads for local-grid and regional-context requests.",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--requests-path", required=True, help="CSV or parquet with grouped request rows.")
    parser.add_argument("--request-id-column", default="request_id")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--default-chip-size", type=int, default=256)
    parser.add_argument("--default-resolution-meters", type=float, default=10.0)
    parser.add_argument("--default-region-side-meters", type=float, default=2560.0)
    parser.add_argument("--chip-size-column", default="chip_size")
    parser.add_argument("--resolution-column", default="resolution_meters")
    parser.add_argument("--region-side-column", default="region_side_meters")
    parser.add_argument("--week-step", type=int, default=4)
    parser.add_argument("--total-weeks", type=int, default=52)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=5.0)
    parser.add_argument("--ee-max-retries", type=int, default=6)
    parser.add_argument("--ee-retry-backoff-sec", type=float, default=10.0)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-bucket", default="")
    parser.add_argument("--gcs-prefix", default="")
    parser.add_argument("--upload-completed-samples-to-gcs", action="store_true")
    parser.add_argument("--delete-local-after-gcs-upload", action="store_true")
    return parser.parse_args()


def _read_table(path: Path):
    pd = require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported grouped request format for {path}. Expected .csv or .parquet")


def load_group_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pd = require_pandas()
    frame = _read_table(Path(args.requests_path)).copy()
    required = [
        args.request_id_column,
        args.sample_id_column,
        args.latitude_column,
        args.longitude_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"Grouped request table missing required columns {missing}: {args.requests_path}")
    frame = frame.rename(
        columns={
            args.request_id_column: "request_id",
            args.sample_id_column: "sample_id",
            args.latitude_column: "latitude",
            args.longitude_column: "longitude",
        }
    )
    frame["request_id"] = frame["request_id"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["request_id", "sample_id", "latitude", "longitude"]).copy()
    request_ids = sorted(frame["request_id"].drop_duplicates().tolist())
    if int(args.sample_offset) > 0:
        request_ids = request_ids[int(args.sample_offset) :]
    if int(args.sample_limit) > 0:
        request_ids = request_ids[: int(args.sample_limit)]
    if not request_ids:
        raise SystemExit("No grouped requests selected.")
    frame = frame[frame["request_id"].isin(set(request_ids))].copy()

    rows: list[dict[str, Any]] = []
    for request_id, group_frame in frame.groupby("request_id", sort=True):
        first = group_frame.iloc[0]
        chip_size = int(first[args.chip_size_column]) if args.chip_size_column in group_frame.columns else int(args.default_chip_size)
        resolution_m = float(first[args.resolution_column]) if args.resolution_column in group_frame.columns else float(args.default_resolution_meters)
        region_side_m = float(first[args.region_side_column]) if args.region_side_column in group_frame.columns else float(args.default_region_side_meters)
        points = [
            {
                "sample_id": str(point_row.sample_id),
                "lat": float(point_row.latitude),
                "lon": float(point_row.longitude),
            }
            for point_row in group_frame[["sample_id", "latitude", "longitude"]].itertuples(index=False)
        ]
        metadata = {
            column: first[column]
            for column in group_frame.columns
            if column not in {"request_id", "sample_id", "latitude", "longitude"}
        }
        rows.append(
            {
                "request_id": str(request_id),
                "chip_size": chip_size,
                "resolution_meters": resolution_m,
                "region_side_meters": region_side_m,
                "points": points,
                "metadata": metadata,
            }
        )
    return rows


def process_group(
    *,
    ee: Any,
    export_module: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    request_id = str(row["request_id"])
    points = list(row["points"])
    chip_size = int(row["chip_size"])
    resolution_meters = float(row["resolution_meters"])
    region_side_meters = float(row["region_side_meters"])
    sample_dir = output_dir / request_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    existing_payload = single.load_json(sample_dir / f"{request_id}.json")
    if single.sample_payload_is_complete(existing_payload):
        payload = dict(existing_payload)
        payload["reused_local_complete_sample"] = True
        if args.upload_completed_samples_to_gcs and not payload.get("gcs_upload"):
            try:
                upload_meta = single.upload_sample_dir_to_gcs(
                    sample_dir=sample_dir,
                    bucket=str(args.gcs_bucket),
                    prefix=str(args.gcs_prefix),
                    sample_id=request_id,
                )
                payload["gcs_upload"] = upload_meta
                if args.delete_local_after_gcs_upload:
                    shutil.rmtree(sample_dir)
                    payload["deleted_local_after_gcs_upload"] = True
            except Exception as exc:
                payload["gcs_upload_error"] = str(exc)
            else:
                if sample_dir.exists():
                    single.write_json(sample_dir / f"{request_id}.json", payload)
        return payload

    group_args = argparse.Namespace(**vars(args))
    group_args.chip_size = chip_size
    group_args.resolution_meters = resolution_meters
    group_args.region_side_meters = region_side_meters

    resolved_points: list[dict[str, Any]] = []
    region_info: dict[str, Any] = {}
    region = None
    chunk_results: list[dict[str, Any]] = []
    group_started = time.time()
    group_error = None
    try:
        region, resolved_points, region_info = export_module._build_region(
            ee,
            lat=None,
            lon=None,
            region_side_meters=region_side_meters,
            chip_size=chip_size,
            resolution_meters=resolution_meters,
            region_padding_meters=0.0,
            points=points,
        )
        for week_start, week_end in single.chunk_ranges(int(args.total_weeks), int(args.week_step)):
            chunk_results.extend(
                single.download_week_range(
                    ee=ee,
                    export_module=export_module,
                    region=region,
                    sample_id=request_id,
                    args=group_args,
                    sample_dir=sample_dir,
                    week_start=int(week_start),
                    week_end=int(week_end),
                )
            )
    except Exception as exc:
        group_error = str(exc)

    payload = {
        "sample_id": request_id,
        "points": resolved_points,
        "region_info": region_info,
        "chip_size": chip_size,
        "resolution_meters": resolution_meters,
        "region_side_meters": region_side_meters,
        "request_metadata": row.get("metadata", {}),
        "chunk_results": chunk_results,
        "sample_wall_sec": round(time.time() - group_started, 2),
    }
    if group_error is not None:
        payload["fatal_error"] = group_error
    single.write_json(sample_dir / f"{request_id}.json", payload)

    if args.upload_completed_samples_to_gcs and single.sample_payload_is_complete(payload):
        try:
            upload_meta = single.upload_sample_dir_to_gcs(
                sample_dir=sample_dir,
                bucket=str(args.gcs_bucket),
                prefix=str(args.gcs_prefix),
                sample_id=request_id,
            )
            payload["gcs_upload"] = upload_meta
            if args.delete_local_after_gcs_upload:
                shutil.rmtree(sample_dir)
                payload["deleted_local_after_gcs_upload"] = True
        except Exception as exc:
            payload["gcs_upload_error"] = str(exc)
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ee = require_ee()
    initialize_ee(ee, project=args.project, authenticate=bool(args.authenticate))
    export_module = single.load_export_module()
    rows = load_group_rows(args)

    start = time.time()
    sample_payloads: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {
            executor.submit(
                process_group,
                ee=ee,
                export_module=export_module,
                row=row,
                args=args,
                output_dir=output_dir,
            ): row
            for row in rows
        }
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                sample_payloads.append(future.result())
            except Exception as exc:
                sample_payloads.append(
                    {
                        "sample_id": str(row["request_id"]),
                        "points": list(row["points"]),
                        "region_info": {},
                        "chunk_results": [],
                        "sample_wall_sec": 0.0,
                        "fatal_error": str(exc),
                    }
                )

    sample_payloads.sort(key=lambda item: str(item["sample_id"]))
    chunk_rows = [chunk for payload in sample_payloads for chunk in payload["chunk_results"]]
    status_ok = [row for row in chunk_rows if row.get("status_code") == 200]
    status_fail = [row for row in chunk_rows if row.get("status_code") != 200]
    failed_samples = [payload for payload in sample_payloads if payload.get("fatal_error")]
    summary = {
        "project": args.project,
        "requests_path": str(args.requests_path),
        "sample_offset": int(args.sample_offset),
        "sample_limit": int(args.sample_limit),
        "sample_count": int(len(sample_payloads)),
        "parallelism": int(args.parallelism),
        "week_step": int(args.week_step),
        "total_weeks": int(args.total_weeks),
        "chunk_count": int(len(chunk_rows)),
        "successful_chunk_count": int(len(status_ok)),
        "failed_chunk_count": int(len(status_fail)),
        "failed_sample_count": int(len(failed_samples)),
        "wall_elapsed_sec": round(time.time() - start, 2),
        "total_downloaded_bytes": int(sum(int(row.get("bytes") or 0) for row in status_ok)),
        "samples": sample_payloads,
    }
    single.write_json(output_dir / "summary.json", summary)
    print(json.dumps(single.to_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
