#!/usr/bin/env python3
"""Extract pre-drill geospatial features from Google Earth Engine.

For each well location, extracts terrain, land cover, and surface state
features from GEE datasets. All features are available BEFORE drilling
(no well/permit/production data used).

Usage:
    python scripts/extract_gee_features.py \
        --cohort-path data/features/multiregion/cohort.parquet \
        --output-path data/features/multiregion/gee_features_v1.parquet \
        --project omois-483220

    # Test with 100 wells first:
    python scripts/extract_gee_features.py \
        --cohort-path data/features/multiregion/cohort.parquet \
        --output-path data/features/multiregion/gee_features_v1_test.parquet \
        --project omois-483220 \
        --max-wells 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GEE features for well locations.")
    parser.add_argument("--cohort-path", required=True, help="Parquet with well locations")
    parser.add_argument("--output-path", required=True, help="Output parquet path")
    parser.add_argument("--project", default="omois-483220", help="GCP project for GEE")
    parser.add_argument("--batch-size", type=int, default=500, help="Wells per GEE batch request")
    parser.add_argument("--max-wells", type=int, default=0, help="Limit wells (0=all)")
    parser.add_argument("--pre-drill-offset-years", type=int, default=2, help="Years before spud for temporal features")
    return parser.parse_args()


def build_static_terrain_image() -> ee.Image:
    """SRTM-derived terrain features (static, global)."""
    srtm = ee.Image("USGS/SRTMGL1_003")
    terrain = ee.Terrain.products(srtm)
    return terrain.select(
        ["elevation", "slope", "aspect"],
        ["gee_elevation_m", "gee_slope_deg", "gee_aspect_deg"],
    )


def build_land_cover_image() -> ee.Image:
    """ESA WorldCover 10m land cover (2021)."""
    return (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select(["Map"], ["gee_land_cover"])
    )


def build_tree_cover_image() -> ee.Image:
    """Hansen Global Forest Change — tree cover percentage."""
    return (
        ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
        .select(["treecover2000"], ["gee_tree_cover_pct"])
    )


def build_predrill_surface_image(year: int) -> ee.Image:
    """Pre-drill surface state features for a given year.

    Computes annual composites for NDVI, NDWI, surface temperature, nightlights.
    """
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    # NDVI and NDWI from Sentinel-2 annual composite
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("gee_ndvi_mean")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("gee_ndwi_mean")

    # Surface temperature from MODIS (Kelvin, scale 0.02)
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)  # Convert to Celsius
        .rename("gee_surface_temp_c")
    )

    # Nighttime lights from VIIRS
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(start, end)
        .select("avg_rad")
        .mean()
        .rename("gee_nightlight_rad")
    )

    return ndvi.addBands(ndwi).addBands(lst).addBands(viirs)


def extract_batch(
    wells_batch: pd.DataFrame,
    static_image: ee.Image,
    land_cover_image: ee.Image,
    tree_cover_image: ee.Image,
    pre_drill_years: dict[int, ee.Image],
) -> list[dict]:
    """Extract features for a batch of wells."""
    features = []
    for _, row in wells_batch.iterrows():
        lon = float(row["sample_longitude"])
        lat = float(row["sample_latitude"])
        props = {"sample_id": str(row["sample_id"])}
        anchor_year = int(row.get("pre_drill_year", 2020))
        props["pre_drill_year"] = anchor_year
        features.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))

    fc = ee.FeatureCollection(features)

    # Extract static terrain
    static_result = static_image.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=30,
    )

    # Extract land cover
    lc_result = land_cover_image.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=10,
    )

    # Extract tree cover
    tc_result = tree_cover_image.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=30,
    )

    # Extract temporal features per year
    temporal_results = {}
    for year, image in pre_drill_years.items():
        temporal_results[year] = image.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=100,  # coarser for faster extraction
        )

    # Fetch all results
    static_data = {f["properties"]["sample_id"]: f["properties"] for f in static_result.getInfo()["features"]}
    lc_data = {f["properties"]["sample_id"]: f["properties"] for f in lc_result.getInfo()["features"]}
    tc_data = {f["properties"]["sample_id"]: f["properties"] for f in tc_result.getInfo()["features"]}

    temporal_data = {}
    for year, result in temporal_results.items():
        temporal_data[year] = {f["properties"]["sample_id"]: f["properties"] for f in result.getInfo()["features"]}

    # Merge into rows
    rows = []
    for _, row in wells_batch.iterrows():
        sid = str(row["sample_id"])
        anchor_year = int(row.get("pre_drill_year", 2020))
        record = {"sample_id": sid}

        # Static terrain
        s = static_data.get(sid, {})
        record["gee_elevation_m"] = s.get("gee_elevation_m")
        record["gee_slope_deg"] = s.get("gee_slope_deg")
        record["gee_aspect_deg"] = s.get("gee_aspect_deg")

        # Land cover
        lc = lc_data.get(sid, {})
        record["gee_land_cover"] = lc.get("gee_land_cover")

        # Tree cover
        tc = tc_data.get(sid, {})
        record["gee_tree_cover_pct"] = tc.get("gee_tree_cover_pct")

        # Temporal (pre-drill year)
        t = temporal_data.get(anchor_year, {}).get(sid, {})
        record["gee_ndvi_mean"] = t.get("gee_ndvi_mean")
        record["gee_ndwi_mean"] = t.get("gee_ndwi_mean")
        record["gee_surface_temp_c"] = t.get("gee_surface_temp_c")
        record["gee_nightlight_rad"] = t.get("gee_nightlight_rad")
        record["gee_pre_drill_year"] = anchor_year

        rows.append(record)

    return rows


def main():
    args = parse_args()

    print("Initializing GEE...", flush=True)
    ee.Initialize(project=args.project)

    print("Loading cohort...", flush=True)
    cohort = pd.read_parquet(args.cohort_path)
    wells = cohort[cohort["sample_type"] == "well"].copy()

    # Compute pre-drill year: spud_date - offset, or first_prod_date - (offset+1)
    spud = pd.to_datetime(wells["spud_date"], errors="coerce")
    first_prod = pd.to_datetime(wells["first_prod_date"], errors="coerce")
    pre_drill_year = spud.dt.year - args.pre_drill_offset_years
    # Fallback: use first_prod_date - (offset+1) for wells without spud_date
    fallback = first_prod.dt.year - (args.pre_drill_offset_years + 1)
    wells["pre_drill_year"] = pre_drill_year.fillna(fallback).fillna(2020).astype(int)

    # Filter to wells with S1+S2 available (pre_drill_year >= 2015)
    wells = wells[wells["pre_drill_year"] >= 2015].copy().reset_index(drop=True)
    print(f"  Wells with S1+S2 pre-drill imagery: {len(wells)}")

    if args.max_wells > 0:
        wells = wells.head(args.max_wells).copy()
        print(f"  Limited to {len(wells)} wells")

    # Unique pre-drill years we need
    unique_years = sorted(wells["pre_drill_year"].unique())
    print(f"  Pre-drill years needed: {unique_years}")

    # Build GEE images
    print("Building GEE image stack...", flush=True)
    static_image = build_static_terrain_image()
    land_cover_image = build_land_cover_image()
    tree_cover_image = build_tree_cover_image()

    pre_drill_images = {}
    for year in unique_years:
        pre_drill_images[year] = build_predrill_surface_image(year)
    print(f"  Built {len(pre_drill_images)} pre-drill annual composites")

    # Extract in batches
    print(f"\nExtracting features ({len(wells)} wells, batch={args.batch_size})...", flush=True)
    all_rows = []
    for start in range(0, len(wells), args.batch_size):
        end = min(start + args.batch_size, len(wells))
        batch = wells.iloc[start:end]

        # Only send images for years in this batch
        batch_years = set(batch["pre_drill_year"].unique())
        batch_images = {y: img for y, img in pre_drill_images.items() if y in batch_years}

        t0 = time.time()
        try:
            rows = extract_batch(batch, static_image, land_cover_image, tree_cover_image, batch_images)
            all_rows.extend(rows)
            elapsed = time.time() - t0
            print(f"  Batch {start//args.batch_size + 1}: {len(rows)} wells in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"  Batch {start//args.batch_size + 1}: ERROR — {e}", flush=True)
            # Try smaller sub-batches
            for sub_start in range(start, end, 50):
                sub_end = min(sub_start + 50, end)
                sub_batch = wells.iloc[sub_start:sub_end]
                sub_years = set(sub_batch["pre_drill_year"].unique())
                sub_images = {y: img for y, img in pre_drill_images.items() if y in sub_years}
                try:
                    rows = extract_batch(sub_batch, static_image, land_cover_image, tree_cover_image, sub_images)
                    all_rows.extend(rows)
                except Exception as sub_e:
                    print(f"    Sub-batch {sub_start}-{sub_end}: ERROR — {sub_e}", flush=True)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(all_rows)
    result.to_parquet(output_path, index=False)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gee_features_v1",
        "well_count": len(result),
        "features": list(result.columns),
        "pre_drill_offset_years": args.pre_drill_offset_years,
        "pre_drill_years": [int(y) for y in unique_years],
        "output_path": str(output_path),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDone! {len(result)} wells → {output_path}")
    print(f"Features: {list(result.columns)}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
