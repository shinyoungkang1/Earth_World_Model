#!/usr/bin/env python3
"""Minimal Earth Engine smoke test for weekly EO availability and one optional export."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any


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
        description="Summarize weekly Sentinel-1/Sentinel-2 availability in Earth Engine."
    )
    parser.add_argument("--project", required=True, help="Google Cloud project for EE.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of chip center.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of chip center.")
    parser.add_argument("--year", type=int, default=2020, help="Anchor year.")
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
        help="Nominal chip resolution in meters.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Run ee.Authenticate() before ee.Initialize().",
    )
    parser.add_argument(
        "--export-gcs-bucket",
        help="If set, export one weekly RGB+S1 image stack to this GCS bucket.",
    )
    parser.add_argument(
        "--export-week-index",
        type=int,
        default=0,
        help="0-based week index to export if --export-gcs-bucket is set.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the weekly summary JSON.",
    )
    return parser


def _square_region(ee: Any, lon: float, lat: float, chip_size: int, resolution_m: float) -> Any:
    half_side_m = chip_size * resolution_m / 2.0
    return ee.Geometry.Point([lon, lat]).buffer(half_side_m).bounds()


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


def _serialize_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _best_s2_props(
    collection: Any, region: Any, start: dt.date, end: dt.date
) -> dict[str, Any] | None:
    weekly = (
        collection.filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    if int(weekly.size().getInfo()) == 0:
        return None
    feature = weekly.first().toDictionary(
        ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"]
    )
    props = feature.getInfo()
    return {
        "item_id": props.get("system:index"),
        "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
        "tile": props.get("MGRS_TILE"),
        "time_start": _serialize_date(props.get("system:time_start")),
    }


def _best_s1_props(
    collection: Any, region: Any, start: dt.date, end: dt.date
) -> dict[str, Any] | None:
    weekly = collection.filterDate(start.isoformat(), end.isoformat()).filterBounds(region)
    if int(weekly.size().getInfo()) == 0:
        return None
    feature = weekly.first().toDictionary(
        ["system:index", "system:time_start", "orbitProperties_pass", "instrumentMode"]
    )
    props = feature.getInfo()
    return {
        "item_id": props.get("system:index"),
        "orbit_pass": props.get("orbitProperties_pass"),
        "instrument_mode": props.get("instrumentMode"),
        "time_start": _serialize_date(props.get("system:time_start")),
    }


def _rgb_s2_image(ee: Any, region: Any, start: dt.date, end: dt.date) -> Any | None:
    weekly = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    if int(weekly.size().getInfo()) == 0:
        return None
    image = weekly.first().select(["B4", "B3", "B2"], ["red", "green", "blue"])
    return image


def _s1_vv_vh_image(ee: Any, region: Any, start: dt.date, end: dt.date) -> Any | None:
    weekly = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if int(weekly.size().getInfo()) == 0:
        return None
    return weekly.first().select(["VV", "VH"], ["vv", "vh"])


def _export_week(
    ee: Any,
    region: Any,
    lat: float,
    lon: float,
    year: int,
    week_index: int,
    bucket: str,
    chip_size: int,
    resolution_m: float,
) -> dict[str, Any]:
    windows = _weekly_windows(year)
    if not 0 <= week_index < len(windows):
        raise SystemExit(f"export week index {week_index} out of range for year {year}")
    start, end = windows[week_index]
    s2 = _rgb_s2_image(ee, region, start, end)
    s1 = _s1_vv_vh_image(ee, region, start, end)
    if s2 is None and s1 is None:
        raise SystemExit("selected export week has neither S1 nor S2 imagery")
    if s2 is None:
        image = s1
    elif s1 is None:
        image = s2
    else:
        image = s2.addBands(s1)
    description = f"ee_weekly_smoke_{year}_w{week_index:02d}_{abs(lat):.3f}_{abs(lon):.3f}"
    task = ee.batch.Export.image.toCloudStorage(
        image=image.toFloat(),
        description=description,
        bucket=bucket,
        fileNamePrefix=description,
        region=region,
        scale=resolution_m,
        maxPixels=max(1e8, chip_size * chip_size * 100),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return {
        "task_id": task.id,
        "description": description,
        "bucket": bucket,
        "week_index": week_index,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def main() -> None:
    args = _build_parser().parse_args()
    ee = _require_ee()
    if args.authenticate:  # pragma: no cover - interactive auth
        ee.Authenticate()
    ee.Initialize(project=args.project)

    region = _square_region(
        ee,
        lon=args.lon,
        lat=args.lat,
        chip_size=args.chip_size,
        resolution_m=args.resolution_meters,
    )
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    weekly = []
    windows = _weekly_windows(args.year)
    paired = 0
    s1_only = 0
    s2_only = 0
    empty = 0
    for idx, (start, end) in enumerate(windows):
        s2_props = _best_s2_props(s2, region, start, end)
        s1_props = _best_s1_props(s1, region, start, end)
        has_s2 = s2_props is not None
        has_s1 = s1_props is not None
        if has_s1 and has_s2:
            paired += 1
        elif has_s1:
            s1_only += 1
        elif has_s2:
            s2_only += 1
        else:
            empty += 1
        weekly.append(
            {
                "week_index": idx,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "has_s2": has_s2,
                "has_s1": has_s1,
                "s2": s2_props,
                "s1": s1_props,
            }
        )

    payload = {
        "project": args.project,
        "lat": args.lat,
        "lon": args.lon,
        "year": args.year,
        "chip_size": args.chip_size,
        "resolution_meters": args.resolution_meters,
        "weeks_total": len(windows),
        "weeks_paired": paired,
        "weeks_s1_only": s1_only,
        "weeks_s2_only": s2_only,
        "weeks_empty": empty,
        "weekly": weekly,
    }

    if args.export_gcs_bucket:
        payload["export"] = _export_week(
            ee=ee,
            region=region,
            lat=args.lat,
            lon=args.lon,
            year=args.year,
            week_index=args.export_week_index,
            bucket=args.export_gcs_bucket,
            chip_size=args.chip_size,
            resolution_m=args.resolution_meters,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")


if __name__ == "__main__":
    main()
