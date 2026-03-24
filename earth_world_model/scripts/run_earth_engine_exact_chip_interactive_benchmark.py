#!/usr/bin/env python3
"""Download exact-chip Earth Engine yearly stacks via interactive 4-week chunks."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("/tmp/ee_exact_chip_interactive_benchmark")
EE_REQUEST_SIZE_LIMIT_PATTERN = re.compile(
    r"Total request size \((?P<requested>\d+) bytes\) must be less than or equal to (?P<limit>\d+) bytes"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact-chip Earth Engine interactive downloads for yearly 4-week chunk stacks."
    )
    parser.add_argument("--project", required=True, help="Google Cloud project for Earth Engine.")
    parser.add_argument("--locations-path", required=True, help="CSV with sample_id,latitude,longitude.")
    parser.add_argument("--sample-offset", type=int, default=0, help="0-based row offset into the locations CSV.")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of sample rows to process.")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--resolution-meters", type=float, default=10.0)
    parser.add_argument("--region-side-meters", type=float, default=2560.0)
    parser.add_argument("--week-step", type=int, default=4, help="Weeks per interactive request.")
    parser.add_argument("--total-weeks", type=int, default=52, help="Total yearly weeks to download.")
    parser.add_argument("--parallelism", type=int, default=5, help="Parallel sample downloads.")
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=5.0)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gcs-bucket", default="", help="Optional GCS bucket for per-sample uploads.")
    parser.add_argument("--gcs-prefix", default="", help="Optional GCS prefix for per-sample uploads.")
    parser.add_argument(
        "--upload-completed-samples-to-gcs",
        action="store_true",
        help="Upload each fully successful sample directory to GCS as soon as it completes.",
    )
    parser.add_argument(
        "--delete-local-after-gcs-upload",
        action="store_true",
        help="Delete local sample directories after a successful per-sample GCS upload.",
    )
    return parser.parse_args()


def require_ee() -> Any:
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("earthengine-api is not installed. Install requirements first.") from exc
    return ee


def load_export_module() -> Any:
    script_path = REPO_ROOT / "earth_world_model" / "scripts" / "run_earth_engine_shard_export.py"
    spec = importlib.util.spec_from_file_location("ee_shard_export", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_locations(path: Path, *, sample_offset: int, sample_limit: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[sample_offset:]
    if sample_limit > 0:
        rows = rows[:sample_limit]
    if not rows:
        raise SystemExit(f"No rows selected from {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def chunk_ranges(total_weeks: int, week_step: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_weeks:
        end = min(total_weeks, start + week_step)
        ranges.append((start, end))
        start = end
    return ranges


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sample_payload_is_complete(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    chunk_results = payload.get("chunk_results") or []
    if not chunk_results:
        return False
    return all(int(item.get("status_code") or 0) == 200 for item in chunk_results)


def build_gcs_sample_uri(*, bucket: str, prefix: str, sample_id: str) -> str:
    clean_prefix = prefix.strip("/")
    if clean_prefix:
        return f"gs://{bucket}/{clean_prefix}/raw/{sample_id}"
    return f"gs://{bucket}/raw/{sample_id}"


def upload_sample_dir_to_gcs(*, sample_dir: Path, bucket: str, prefix: str, sample_id: str) -> dict[str, Any]:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-storage is required for per-sample GCS upload") from exc

    destination = build_gcs_sample_uri(bucket=bucket, prefix=prefix, sample_id=sample_id)
    clean_prefix = prefix.strip("/")
    started = time.time()
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    uploaded_files = 0
    uploaded_bytes = 0
    for path in sorted(sample_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(sample_dir).as_posix()
        blob_name = f"raw/{sample_id}/{relative}"
        if clean_prefix:
            blob_name = f"{clean_prefix}/{blob_name}"
        blob = bucket_obj.blob(blob_name)
        blob.upload_from_filename(str(path))
        uploaded_files += 1
        uploaded_bytes += int(path.stat().st_size)
    return {
        "destination": destination,
        "elapsed_sec": round(time.time() - started, 2),
        "uploaded_files": uploaded_files,
        "uploaded_bytes": uploaded_bytes,
    }


def download_with_retries(
    *,
    url: str,
    output_path: Path,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        started = time.time()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "codex-ee-interactive"})
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                payload = response.read()
                status_code = int(getattr(response, "status", 200))
                headers = dict(response.headers)
            output_path.write_bytes(payload)
            return {
                "status_code": status_code,
                "bytes": len(payload),
                "elapsed_sec": round(time.time() - started, 2),
                "attempt": attempt,
                "headers": headers,
            }
        except Exception as exc:
            error = str(exc)
            if attempt >= max_retries:
                return {
                    "status_code": None,
                    "bytes": 0,
                    "elapsed_sec": round(time.time() - started, 2),
                    "attempt": attempt,
                    "error": error,
                }
            time.sleep(max(0.0, retry_backoff_sec) * attempt)
    raise AssertionError("unreachable")


def parse_request_size_limit_error(exc: Exception) -> dict[str, int] | None:
    match = EE_REQUEST_SIZE_LIMIT_PATTERN.search(str(exc))
    if not match:
        return None
    return {
        "requested_bytes": int(match.group("requested")),
        "limit_bytes": int(match.group("limit")),
    }


def download_week_range(
    *,
    ee: Any,
    export_module: Any,
    region: Any,
    sample_id: str,
    args: argparse.Namespace,
    sample_dir: Path,
    week_start: int,
    week_end: int,
    split_depth: int = 0,
    split_parent: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    week_count = int(week_end - week_start)
    image, weekly_manifest = export_module._build_export_image(
        ee=ee,
        region=region,
        year=int(args.year),
        week_start_index=int(week_start),
        week_count=week_count,
    )
    image = image.toFloat()
    params = {
        "region": region,
        "scale": float(args.resolution_meters),
        "format": "GEO_TIFF",
    }
    url_started = time.time()
    try:
        url = image.getDownloadURL(params)
    except Exception as exc:
        size_error = parse_request_size_limit_error(exc)
        if size_error is not None and week_count > 1:
            mid = int(week_start + max(1, week_count // 2))
            left = download_week_range(
                ee=ee,
                export_module=export_module,
                region=region,
                sample_id=sample_id,
                args=args,
                sample_dir=sample_dir,
                week_start=week_start,
                week_end=mid,
                split_depth=split_depth + 1,
                split_parent=(week_start, week_end),
            )
            right = download_week_range(
                ee=ee,
                export_module=export_module,
                region=region,
                sample_id=sample_id,
                args=args,
                sample_dir=sample_dir,
                week_start=mid,
                week_end=week_end,
                split_depth=split_depth + 1,
                split_parent=(week_start, week_end),
            )
            return left + right
        return [
            {
                "sample_id": sample_id,
                "week_start_index": int(week_start),
                "week_end_index": int(week_end),
                "output_path": str(sample_dir / f"w{week_start:02d}_{week_end:02d}.tif"),
                "url_elapsed_sec": round(time.time() - url_started, 2),
                "weekly_manifest": weekly_manifest,
                "status_code": None,
                "bytes": 0,
                "elapsed_sec": round(time.time() - url_started, 2),
                "attempt": 0,
                "error": str(exc),
                "split_depth": int(split_depth),
                "split_parent": list(split_parent) if split_parent is not None else None,
                "request_size_error": size_error,
            }
        ]
    url_elapsed = round(time.time() - url_started, 2)

    output_path = sample_dir / f"w{week_start:02d}_{week_end:02d}.tif"
    download_meta = download_with_retries(
        url=url,
        output_path=output_path,
        timeout_sec=float(args.request_timeout_sec),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )
    return [
        {
            "sample_id": sample_id,
            "week_start_index": int(week_start),
            "week_end_index": int(week_end),
            "output_path": str(output_path),
            "url_elapsed_sec": url_elapsed,
            "weekly_manifest": weekly_manifest,
            "split_depth": int(split_depth),
            "split_parent": list(split_parent) if split_parent is not None else None,
            **download_meta,
        }
    ]


def process_sample(
    *,
    ee: Any,
    export_module: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    existing_payload = load_json(sample_dir / f"{sample_id}.json")
    if sample_payload_is_complete(existing_payload):
        payload = dict(existing_payload)
        payload["reused_local_complete_sample"] = True
        return payload

    region, resolved_points, region_info = export_module._build_region(
        ee,
        lat=lat,
        lon=lon,
        region_side_meters=float(args.region_side_meters),
        chip_size=int(args.chip_size),
        resolution_meters=float(args.resolution_meters),
        region_padding_meters=0.0,
        points=None,
    )

    chunk_results: list[dict[str, Any]] = []
    sample_started = time.time()
    for week_start, week_end in chunk_ranges(int(args.total_weeks), int(args.week_step)):
        chunk_results.extend(
            download_week_range(
                ee=ee,
                export_module=export_module,
                region=region,
                sample_id=sample_id,
                args=args,
                sample_dir=sample_dir,
                week_start=int(week_start),
                week_end=int(week_end),
            )
        )

    sample_payload = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "points": resolved_points,
        "region_info": region_info,
        "chunk_results": chunk_results,
        "sample_wall_sec": round(time.time() - sample_started, 2),
    }
    write_json(sample_dir / f"{sample_id}.json", sample_payload)

    if args.upload_completed_samples_to_gcs and sample_payload_is_complete(sample_payload):
        try:
            upload_meta = upload_sample_dir_to_gcs(
                sample_dir=sample_dir,
                bucket=str(args.gcs_bucket),
                prefix=str(args.gcs_prefix),
                sample_id=sample_id,
            )
            sample_payload["gcs_upload"] = upload_meta
            if args.delete_local_after_gcs_upload:
                shutil.rmtree(sample_dir)
                sample_payload["deleted_local_after_gcs_upload"] = True
        except Exception as exc:
            sample_payload["gcs_upload_error"] = str(exc)
    return sample_payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ee = require_ee()
    if args.authenticate:  # pragma: no cover
        ee.Authenticate()
    ee.Initialize(project=args.project)
    export_module = load_export_module()

    rows = load_locations(
        Path(args.locations_path),
        sample_offset=int(args.sample_offset),
        sample_limit=int(args.sample_limit),
    )

    start = time.time()
    sample_payloads: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {
            executor.submit(
                process_sample,
                ee=ee,
                export_module=export_module,
                row=row,
                args=args,
                output_dir=output_dir,
            ): row
            for row in rows
        }
        for future in as_completed(future_map):
            sample_payloads.append(future.result())

    sample_payloads.sort(key=lambda item: str(item["sample_id"]))
    chunk_rows = [chunk for payload in sample_payloads for chunk in payload["chunk_results"]]
    status_ok = [row for row in chunk_rows if row.get("status_code") == 200]
    status_fail = [row for row in chunk_rows if row.get("status_code") != 200]
    summary = {
        "project": args.project,
        "locations_path": str(args.locations_path),
        "sample_offset": int(args.sample_offset),
        "sample_limit": int(args.sample_limit),
        "sample_count": int(len(sample_payloads)),
        "parallelism": int(args.parallelism),
        "week_step": int(args.week_step),
        "total_weeks": int(args.total_weeks),
        "chunk_count": int(len(chunk_rows)),
        "successful_chunk_count": int(len(status_ok)),
        "failed_chunk_count": int(len(status_fail)),
        "wall_elapsed_sec": round(time.time() - start, 2),
        "avg_request_total_sec": round(
            sum(float(row.get("elapsed_sec") or 0.0) for row in chunk_rows) / max(1, len(chunk_rows)),
            2,
        ),
        "avg_request_download_sec": round(
            sum(float(row.get("elapsed_sec") or 0.0) for row in status_ok) / max(1, len(status_ok)),
            2,
        ),
        "avg_request_url_sec": round(
            sum(float(row.get("url_elapsed_sec") or 0.0) for row in chunk_rows) / max(1, len(chunk_rows)),
            2,
        ),
        "total_downloaded_bytes": int(sum(int(row.get("bytes") or 0) for row in status_ok)),
        "samples": sample_payloads,
        "failed_chunks": status_fail,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary_path": str(output_dir / "summary.json"),
                "sample_count": summary["sample_count"],
                "chunk_count": summary["chunk_count"],
                "successful_chunk_count": summary["successful_chunk_count"],
                "failed_chunk_count": summary["failed_chunk_count"],
                "wall_elapsed_sec": summary["wall_elapsed_sec"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
