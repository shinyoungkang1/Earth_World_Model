#!/usr/bin/env python3
"""Launch a grouped Earth Engine shard export for weekly Sentinel-1/Sentinel-2 data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


def _require_ee() -> Any:
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "earthengine-api is not installed. Install requirements first."
        ) from exc
    return ee


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a grouped weekly shard from Earth Engine to GCS."
    )
    parser.add_argument("--project", required=True, help="Google Cloud project for EE.")
    parser.add_argument("--bucket", required=True, help="Target GCS bucket.")
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Anchor year for weekly windows.",
    )
    parser.add_argument(
        "--week-start-index",
        type=int,
        default=0,
        help="0-based weekly window start index.",
    )
    parser.add_argument(
        "--week-count",
        type=int,
        default=8,
        help="Number of weekly windows to pack into one shard export.",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=256,
        help="Chip size in pixels at 10m resolution.",
    )
    parser.add_argument(
        "--resolution-meters",
        type=float,
        default=10.0,
        help="Nominal output resolution in meters.",
    )
    parser.add_argument(
        "--region-padding-meters",
        type=float,
        default=2048.0,
        help="Extra padding around multi-point bounds.",
    )
    parser.add_argument(
        "--region-side-meters",
        type=float,
        default=20480.0,
        help="Square region side length for single-center mode.",
    )
    parser.add_argument("--lat", type=float, help="Latitude for single-center mode.")
    parser.add_argument("--lon", type=float, help="Longitude for single-center mode.")
    parser.add_argument(
        "--locations-path",
        help="CSV with sample_id,latitude,longitude for grouped export mode.",
    )
    parser.add_argument(
        "--sample-ids",
        help="Comma-separated sample_id subset from --locations-path.",
    )
    parser.add_argument(
        "--description-prefix",
        default="ee_weekly_shard",
        help="Task description prefix.",
    )
    parser.add_argument(
        "--file-name-prefix",
        default="earth_engine_shards",
        help="GCS object prefix.",
    )
    parser.add_argument(
        "--file-dimensions",
        type=int,
        default=2048,
        help="GeoTIFF tile dimensions for export splitting.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Earth Engine compute shardSize.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        help="Optional EE batch task priority for paid projects.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Run ee.Authenticate() before ee.Initialize().",
    )
    parser.add_argument(
        "--output-json",
        help="Optional local path to save export manifest JSON.",
    )
    return parser


def _weekly_windows(year: int) -> list[tuple[dt.date, dt.date]]:
    start = dt.date(year, 1, 1)
    end = dt.date(year + 1, 1, 1)
    windows: list[tuple[dt.date, dt.date]] = []
    current = start
    while current < end:
        nxt = min(current + dt.timedelta(days=7), end)
        windows.append((current, nxt))
        current = nxt
    return windows


def _load_points(locations_path: str, sample_ids: str | None) -> list[dict[str, Any]]:
    wanted = None
    if sample_ids:
        wanted = {token.strip() for token in sample_ids.split(",") if token.strip()}
    with open(locations_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        sample_id = row["sample_id"]
        if wanted is not None and sample_id not in wanted:
            continue
        points.append(
            {
                "sample_id": sample_id,
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }
        )
    if not points:
        raise SystemExit("no points matched the requested locations")
    return points


def _build_region(
    ee: Any,
    *,
    lat: float | None,
    lon: float | None,
    region_side_meters: float,
    chip_size: int,
    resolution_meters: float,
    region_padding_meters: float,
    points: list[dict[str, Any]] | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if points:
        coords = [[point["lon"], point["lat"]] for point in points]
        chip_half_side = chip_size * resolution_meters / 2.0
        region = (
            ee.Geometry.MultiPoint(coords)
            .bounds()
            .buffer(chip_half_side + region_padding_meters)
            .bounds()
        )
        region_info = {
            "mode": "points_bbox",
            "point_count": len(points),
            "sample_ids": [point["sample_id"] for point in points],
        }
        return region, points, region_info
    if lat is None or lon is None:
        raise SystemExit("single-center mode requires --lat and --lon")
    half_side = region_side_meters / 2.0
    region = ee.Geometry.Point([lon, lat]).buffer(half_side).bounds()
    point = {"sample_id": "center", "lat": lat, "lon": lon}
    region_info = {
        "mode": "square",
        "center_lat": lat,
        "center_lon": lon,
        "region_side_meters": region_side_meters,
    }
    return region, [point], region_info


def _constant_band_image(ee: Any, band_names: list[str], value: float) -> Any:
    return ee.Image.constant([value] * len(band_names)).rename(band_names)


def _rename_for_week(week_index: int, band_names: list[str]) -> list[str]:
    return [f"w{week_index:02d}_{band}" for band in band_names]


def _best_s2_collection(ee: Any, region: Any, start: dt.date, end: dt.date) -> Any:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )


def _best_s1_collection(ee: Any, region: Any, start: dt.date, end: dt.date) -> Any:
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def _dict_info(feature: Any, properties: list[str]) -> dict[str, Any]:
    return feature.toDictionary(properties).getInfo()


def _week_stack(ee: Any, region: Any, start: dt.date, end: dt.date, week_index: int) -> tuple[Any, dict[str, Any]]:
    s2_collection = _best_s2_collection(ee, region, start, end)
    s1_collection = _best_s1_collection(ee, region, start, end)

    has_s2 = int(s2_collection.size().getInfo()) > 0
    has_s1 = int(s1_collection.size().getInfo()) > 0

    metadata: dict[str, Any] = {
        "week_index": week_index,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "has_s2": has_s2,
        "has_s1": has_s1,
    }

    images = []

    s2_band_names = _rename_for_week(week_index, S2_BANDS)
    scl_band_name = _rename_for_week(week_index, ["SCL"])
    s1_band_names = _rename_for_week(week_index, S1_BANDS)
    presence_band_names = _rename_for_week(week_index, ["S2_PRESENT", "S1_PRESENT"])

    if has_s2:
        s2_first = s2_collection.first()
        s2_props = _dict_info(
            s2_first,
            ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"],
        )
        metadata["s2"] = {
            "item_id": s2_props.get("system:index"),
            "time_start": str(s2_props.get("system:time_start")),
            "cloud_pct": s2_props.get("CLOUDY_PIXEL_PERCENTAGE"),
            "tile": s2_props.get("MGRS_TILE"),
        }
        images.append(s2_first.select(S2_BANDS, s2_band_names))
        images.append(s2_first.select(["SCL"], scl_band_name))
    else:
        metadata["s2"] = None
        images.append(_constant_band_image(ee, s2_band_names, 0))
        images.append(_constant_band_image(ee, scl_band_name, 0))

    if has_s1:
        s1_first = s1_collection.first()
        s1_props = _dict_info(
            s1_first,
            ["system:index", "system:time_start", "orbitProperties_pass", "instrumentMode"],
        )
        metadata["s1"] = {
            "item_id": s1_props.get("system:index"),
            "time_start": str(s1_props.get("system:time_start")),
            "orbit_pass": s1_props.get("orbitProperties_pass"),
            "instrument_mode": s1_props.get("instrumentMode"),
        }
        images.append(s1_first.select(S1_BANDS, s1_band_names))
    else:
        metadata["s1"] = None
        images.append(_constant_band_image(ee, s1_band_names, 0))

    images.append(
        ee.Image.constant([int(has_s2), int(has_s1)]).rename(presence_band_names)
    )

    stack = images[0]
    for image in images[1:]:
        stack = stack.addBands(image)
    return stack, metadata


def _build_export_image(
    ee: Any,
    region: Any,
    year: int,
    week_start_index: int,
    week_count: int,
) -> tuple[Any, list[dict[str, Any]]]:
    windows = _weekly_windows(year)
    if week_start_index < 0 or week_start_index >= len(windows):
        raise SystemExit("week-start-index is out of range")
    end_index = min(week_start_index + week_count, len(windows))
    selected = windows[week_start_index:end_index]
    if not selected:
        raise SystemExit("no weekly windows selected")

    manifest: list[dict[str, Any]] = []
    image = None
    for offset, (start, end) in enumerate(selected):
        global_week_index = week_start_index + offset
        week_image, metadata = _week_stack(ee, region, start, end, global_week_index)
        manifest.append(metadata)
        image = week_image if image is None else image.addBands(week_image)
    assert image is not None
    return image, manifest


def main() -> None:
    args = _build_parser().parse_args()
    ee = _require_ee()
    if args.authenticate:  # pragma: no cover - interactive auth
        ee.Authenticate()
    ee.Initialize(project=args.project)

    points = None
    if args.locations_path:
        points = _load_points(args.locations_path, args.sample_ids)

    region, resolved_points, region_info = _build_region(
        ee,
        lat=args.lat,
        lon=args.lon,
        region_side_meters=args.region_side_meters,
        chip_size=args.chip_size,
        resolution_meters=args.resolution_meters,
        region_padding_meters=args.region_padding_meters,
        points=points,
    )

    image, weekly_manifest = _build_export_image(
        ee=ee,
        region=region,
        year=args.year,
        week_start_index=args.week_start_index,
        week_count=args.week_count,
    )

    region_geojson = region.getInfo()
    description = (
        f"{args.description_prefix}_{args.year}_w{args.week_start_index:02d}"
        f"_n{len(resolved_points):03d}"
    )
    task_kwargs = {
        "image": image.toFloat(),
        "description": description,
        "bucket": args.bucket,
        "fileNamePrefix": f"{args.file_name_prefix}/{description}",
        "region": region,
        "scale": args.resolution_meters,
        "maxPixels": 1e10,
        "shardSize": args.shard_size,
        "fileDimensions": args.file_dimensions,
        "fileFormat": "GeoTIFF",
        "formatOptions": {"cloudOptimized": True},
    }
    if args.priority is not None:
        task_kwargs["priority"] = args.priority
    task = ee.batch.Export.image.toCloudStorage(**task_kwargs)
    task.start()

    payload = {
        "project": args.project,
        "bucket": args.bucket,
        "task_id": task.id,
        "description": description,
        "year": args.year,
        "week_start_index": args.week_start_index,
        "week_count": args.week_count,
        "resolution_meters": args.resolution_meters,
        "chip_size": args.chip_size,
        "region_info": region_info,
        "points": resolved_points,
        "region": region_geojson,
        "weekly_manifest": weekly_manifest,
        "file_name_prefix": f"{args.file_name_prefix}/{description}",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
