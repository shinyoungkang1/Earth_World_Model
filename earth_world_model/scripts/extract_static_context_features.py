#!/usr/bin/env python3
"""Extract one-time static context features for each sample_id from Earth Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ee_stage_utils import initialize_ee, require_ee
from gee_decoder_target_utils import build_feature_collection, chunk_frame, load_locations, require_pandas


PRODUCT_NAME = "static_context_v1"
DEM_DATASET_ID = "NASA/NASADEM_HGT/001"
WORLDCOVER_DATASET_ID = "ESA/WorldCover/v200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract static DEM/terrain/land-cover context from Earth Engine.")
    parser.add_argument("--project", required=True, help="GCP project for Earth Engine.")
    parser.add_argument("--locations-path", required=True, help="CSV or parquet with sample_id and coordinates.")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument("--batch-size", type=int, default=256, help="Samples per EE reduceRegions request.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit loaded sample rows.")
    parser.add_argument(
        "--aggregation",
        choices=("point", "square_mean"),
        default="point",
        help="Spatial aggregation mode. point=single-pixel sample, square_mean=windowed mean/mode.",
    )
    parser.add_argument(
        "--region-side-meters",
        type=float,
        default=2560.0,
        help="Square side length used when aggregation=square_mean.",
    )
    parser.add_argument("--continuous-scale-meters", type=float, default=30.0, help="DEM/terrain reduce scale.")
    parser.add_argument("--landcover-scale-meters", type=float, default=10.0, help="WorldCover reduce scale.")
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--authenticate", action="store_true")
    return parser.parse_args()


def _collect_feature_rows(reduced_fc: Any) -> list[dict[str, Any]]:
    payload = reduced_fc.getInfo()
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = dict(feature.get("properties") or {})
        sample_id = props.pop("sample_id", None)
        if sample_id is None:
            continue
        row: dict[str, Any] = {"sample_id": str(sample_id)}
        row.update(props)
        rows.append(row)
    return rows


def _merge_rows_by_sample_id(*row_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rows in row_sets:
        for row in rows:
            sample_id = str(row["sample_id"])
            target = merged.setdefault(sample_id, {"sample_id": sample_id})
            target.update(row)
    result = []
    for sample_id in sorted(merged):
        row = merged[sample_id]
        continuous_valid = 0.0
        worldcover_valid = 0.0
        if row.get("elevation_m") is not None or row.get("slope_deg") is not None or row.get("aspect_deg") is not None:
            continuous_valid = 1.0
        if row.get("worldcover_class") is not None:
            worldcover_valid = 1.0
        row["continuous_valid"] = continuous_valid
        row["worldcover_valid"] = worldcover_valid
        row["static_valid"] = 1.0 if continuous_valid > 0.0 or worldcover_valid > 0.0 else 0.0
        result.append(row)
    return result


def _normalize_worldcover_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if "worldcover_class" not in updated:
            for fallback_key in ("first", "mode", "Map"):
                if fallback_key in updated:
                    updated["worldcover_class"] = updated.pop(fallback_key)
                    break
        normalized.append(updated)
    return normalized


def main() -> None:
    args = parse_args()
    ee = require_ee()
    initialize_ee(ee, project=args.project, authenticate=bool(args.authenticate))

    samples = load_locations(
        path=Path(args.locations_path),
        sample_id_column=args.sample_id_column,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        max_samples=int(args.max_samples),
    )

    dem_image = ee.Image(DEM_DATASET_ID).select("elevation").rename("elevation_m").toFloat()
    terrain_image = ee.Algorithms.Terrain(dem_image)
    continuous_image = (
        dem_image
        .addBands(terrain_image.select("slope").rename("slope_deg").toFloat())
        .addBands(terrain_image.select("aspect").rename("aspect_deg").toFloat())
    )
    worldcover_image = (
        ee.ImageCollection(WORLDCOVER_DATASET_ID)
        .first()
        .select("Map")
        .rename("worldcover_class")
        .toFloat()
    )

    continuous_reducer = ee.Reducer.first() if args.aggregation == "point" else ee.Reducer.mean()
    worldcover_reducer = ee.Reducer.first() if args.aggregation == "point" else ee.Reducer.mode()

    continuous_rows: list[dict[str, Any]] = []
    worldcover_rows: list[dict[str, Any]] = []
    for sample_batch in chunk_frame(samples, int(args.batch_size)):
        fc = build_feature_collection(
            ee,
            frame=sample_batch,
            aggregation=args.aggregation,
            region_side_meters=float(args.region_side_meters),
        )
        continuous_fc = continuous_image.reduceRegions(
            collection=fc,
            reducer=continuous_reducer,
            scale=float(args.continuous_scale_meters),
        )
        worldcover_fc = worldcover_image.reduceRegions(
            collection=fc,
            reducer=worldcover_reducer,
            scale=float(args.landcover_scale_meters),
        )
        continuous_rows.extend(_collect_feature_rows(continuous_fc))
        worldcover_rows.extend(_normalize_worldcover_rows(_collect_feature_rows(worldcover_fc)))

    rows = _merge_rows_by_sample_id(continuous_rows, worldcover_rows)
    pd = require_pandas()
    frame = pd.DataFrame(rows)
    output_parquet = Path(args.output_parquet)
    output_metadata_json = Path(args.output_metadata_json)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)

    metadata = {
        "dataset": PRODUCT_NAME,
        "source_dataset_ids": {
            "dem": DEM_DATASET_ID,
            "worldcover": WORLDCOVER_DATASET_ID,
        },
        "locations_path": str(Path(args.locations_path)),
        "sample_count": int(len(samples)),
        "row_count_output": int(len(frame)),
        "aggregation": str(args.aggregation),
        "region_side_meters": float(args.region_side_meters),
        "continuous_scale_meters": float(args.continuous_scale_meters),
        "landcover_scale_meters": float(args.landcover_scale_meters),
        "output_parquet": str(output_parquet),
        "output_metadata_json": str(output_metadata_json),
    }
    output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": PRODUCT_NAME, "rows": int(len(frame)), "output_parquet": str(output_parquet)}, indent=2))


if __name__ == "__main__":
    main()
