#!/usr/bin/env python3
"""Launch and track Earth Engine exact-chip batch exports for yearly benchmarks."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCATIONS_PATH = REPO_ROOT / "data" / "raw" / "dense_temporal_seed_locations_pilot10.csv"
DEFAULT_OUTPUT_DIR = Path("/tmp/ee_exact_chip_batch_benchmark")
FINAL_TASK_STATES = {"COMPLETED", "FAILED", "CANCELLED"}
ACTIVE_TASK_STATES = {"READY", "RUNNING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and track exact-chip Earth Engine batch exports for yearly S1/S2 stacks."
    )
    parser.add_argument("--project", required=True, help="Google Cloud project for Earth Engine.")
    parser.add_argument("--bucket", required=True, help="Target GCS bucket.")
    parser.add_argument("--locations-path", default=str(DEFAULT_LOCATIONS_PATH))
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--week-start-index", type=int, default=0)
    parser.add_argument("--week-count", type=int, default=52)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--resolution-meters", type=float, default=10.0)
    parser.add_argument("--region-side-meters", type=float, default=2560.0)
    parser.add_argument("--description-prefix", default="ee_exact_chip52w_batch")
    parser.add_argument("--file-name-prefix", default="earth_engine_exact_chip52w_batch")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--file-dimensions", type=int, default=256)
    parser.add_argument("--priority", type=int, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=30.0)
    parser.add_argument("--max-poll-minutes", type=float, default=20.0)
    parser.add_argument("--cancel-active-first", action="store_true")
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--interactive-reference-path",
        default="/tmp/ee_interactive_parallel5_full_year_4w/summary.json",
        help="Optional existing interactive benchmark summary to include for comparison.",
    )
    return parser.parse_args()


def require_ee() -> Any:
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
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


def load_locations(path: Path, sample_limit: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if sample_limit > 0:
        rows = rows[:sample_limit]
    if not rows:
        raise SystemExit(f"No sample rows found in {path}")
    return rows


def cancel_active_tasks(ee: Any) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for task in ee.data.getTaskList():
        state = str(task.get("state") or "")
        if state not in ACTIVE_TASK_STATES:
            continue
        task_id = str(task.get("id"))
        description = str(task.get("description") or "")
        try:
            ee.data.cancelTask(task_id)
            action = "cancel_requested"
        except Exception as exc:  # pragma: no cover - network/service dependent
            action = f"cancel_failed: {exc}"
        actions.append(
            {
                "id": task_id,
                "description": description,
                "from_state": state,
                "action": action,
            }
        )
    return actions


def launch_exact_chip_task(
    *,
    ee: Any,
    export_module: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    lat = float(row["latitude"])
    lon = float(row["longitude"])

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
    image, weekly_manifest = export_module._build_export_image(
        ee=ee,
        region=region,
        year=int(args.year),
        week_start_index=int(args.week_start_index),
        week_count=int(args.week_count),
    )
    image = image.toFloat()

    description = (
        f"{args.description_prefix}_{sample_id}_{args.year}_"
        f"w{int(args.week_start_index):02d}_n001"
    )
    file_name_prefix = f"{args.file_name_prefix}/{sample_id}/{description}"
    task_kwargs: dict[str, Any] = {
        "image": image,
        "description": description,
        "bucket": args.bucket,
        "fileNamePrefix": file_name_prefix,
        "region": region,
        "scale": float(args.resolution_meters),
        "maxPixels": 1e10,
        "shardSize": int(args.shard_size),
        "fileDimensions": int(args.file_dimensions),
        "fileFormat": "GeoTIFF",
        "formatOptions": {"cloudOptimized": True},
    }
    if args.priority is not None:
        task_kwargs["priority"] = int(args.priority)

    task = ee.batch.Export.image.toCloudStorage(**task_kwargs)
    task.start()
    return {
        "sample_id": sample_id,
        "task_id": str(task.id),
        "description": description,
        "bucket": args.bucket,
        "file_name_prefix": file_name_prefix,
        "lat": lat,
        "lon": lon,
        "chip_size": int(args.chip_size),
        "region_side_meters": float(args.region_side_meters),
        "week_start_index": int(args.week_start_index),
        "week_count": int(args.week_count),
        "points": resolved_points,
        "region_info": region_info,
        "weekly_manifest": weekly_manifest,
    }


def fetch_task_status(ee: Any, task_id: str) -> dict[str, Any]:
    status = ee.data.getTaskStatus(task_id)[0]
    return {
        "task_id": task_id,
        "state": status.get("state"),
        "description": status.get("description"),
        "creation_timestamp_ms": status.get("creation_timestamp_ms"),
        "start_timestamp_ms": status.get("start_timestamp_ms"),
        "update_timestamp_ms": status.get("update_timestamp_ms"),
        "error_message": status.get("error_message"),
    }


def load_interactive_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "path": str(path),
        "wall_elapsed_sec": payload.get("wall_elapsed_sec"),
        "sample_count": payload.get("sample_count"),
        "chunk_count": payload.get("chunk_count"),
        "avg_chunk_wall_sec": payload.get("avg_chunk_wall_sec"),
        "avg_request_total_sec": payload.get("avg_request_total_sec"),
        "avg_request_download_sec": payload.get("avg_request_download_sec"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ee = require_ee()
    if args.authenticate:  # pragma: no cover - interactive auth
        ee.Authenticate()
    ee.Initialize(project=args.project)
    export_module = load_export_module()

    cancel_actions: list[dict[str, Any]] = []
    if args.cancel_active_first:
        cancel_actions = cancel_active_tasks(ee)

    locations = load_locations(Path(args.locations_path), int(args.sample_limit))
    launches: list[dict[str, Any]] = []
    launch_start = time.time()
    for row in locations:
        launch = launch_exact_chip_task(ee=ee, export_module=export_module, row=row, args=args)
        launches.append(launch)
        write_json(output_dir / f"{launch['sample_id']}.json", launch)
    launch_wall_sec = round(time.time() - launch_start, 2)

    events: list[dict[str, Any]] = []
    latest_status: dict[str, dict[str, Any]] = {}
    last_state: dict[str, str | None] = {}
    deadline = time.time() + max(0.0, float(args.max_poll_minutes)) * 60.0
    while True:
        statuses = []
        all_final = True
        for launch in launches:
            status = fetch_task_status(ee, str(launch["task_id"]))
            statuses.append(status)
            latest_status[str(launch["task_id"])] = status
            state = str(status.get("state") or "")
            if last_state.get(str(launch["task_id"])) != state:
                last_state[str(launch["task_id"])] = state
                events.append(
                    {
                        "ts_unix": time.time(),
                        "task_id": str(launch["task_id"]),
                        "sample_id": str(launch["sample_id"]),
                        "state": state,
                    }
                )
            if state not in FINAL_TASK_STATES:
                all_final = False

        snapshot = {
            "project": args.project,
            "bucket": args.bucket,
            "launch_wall_sec": launch_wall_sec,
            "cancel_actions": cancel_actions,
            "launches": launches,
            "statuses": statuses,
            "events": events,
            "interactive_reference": load_interactive_reference(Path(args.interactive_reference_path)),
            "polled_at_unix": time.time(),
        }
        write_json(output_dir / "live_status.json", snapshot)

        if all_final or time.time() >= deadline:
            break
        time.sleep(max(1.0, float(args.poll_interval_sec)))

    final_statuses = [latest_status[str(launch["task_id"])] for launch in launches]
    state_counts: dict[str, int] = {}
    for status in final_statuses:
        state = str(status.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1

    summary = {
        "project": args.project,
        "bucket": args.bucket,
        "year": int(args.year),
        "week_count": int(args.week_count),
        "chip_size": int(args.chip_size),
        "launch_wall_sec": launch_wall_sec,
        "cancel_actions": cancel_actions,
        "launch_count": len(launches),
        "state_counts": state_counts,
        "launches": launches,
        "final_statuses": final_statuses,
        "events": events,
        "interactive_reference": load_interactive_reference(Path(args.interactive_reference_path)),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary_path": str(output_dir / "summary.json"),
                "live_status_path": str(output_dir / "live_status.json"),
                "launch_count": len(launches),
                "launch_wall_sec": launch_wall_sec,
                "state_counts": state_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
