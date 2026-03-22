#!/usr/bin/env python3
"""Score the basin-scoped gas v1 prospect grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gas_v1_common import (
    assign_geology,
    build_balltree,
    build_geology_lookup,
    close_dem_datasets,
    count_neighbors_current,
    load_basin_config,
    load_dem_datasets,
    load_fault_index,
    nearest_fault_distance_km,
    nearest_neighbor_current_km,
    polygon_from_bbox,
    prospect_tier,
    sample_dem_features,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path override for the trained v1 model.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    counties = set(basin["counties"])

    canonical_dir = repo_root / "data" / "canonical" / "pa_mvp"
    features_dir = repo_root / "data" / "features" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    derived_dir = repo_root / "data" / "derived" / args.basin_id
    derived_dir.mkdir(parents=True, exist_ok=True)

    training_metadata = json.loads((features_dir / "gas_training_table_v1_metadata.json").read_text(encoding="utf-8"))
    model_path = (
        Path(args.model_path)
        if args.model_path
        else models_dir / "gas_baseline_v1_label_recent12_ge_250000.joblib"
    )
    pipeline = joblib.load(model_path)
    metrics_path = model_path.with_name(model_path.stem + "_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    wells = pd.read_csv(canonical_dir / "wells.csv")
    wells = wells[(wells["unconventional"] == True) & (wells["county_name"].isin(counties))].copy()
    wells = wells.dropna(subset=["latitude_decimal", "longitude_decimal"]).copy()
    permits = pd.read_csv(canonical_dir / "permits.csv")
    permits = permits[permits["county"].isin(counties)].dropna(subset=["latitude", "longitude"]).copy()

    well_frame, well_tree = build_balltree(wells, "latitude_decimal", "longitude_decimal")
    permit_frame, permit_tree = build_balltree(permits, "latitude", "longitude")

    min_lon = float(well_frame["longitude_decimal"].min()) - 0.05
    max_lon = float(well_frame["longitude_decimal"].max()) + 0.05
    min_lat = float(well_frame["latitude_decimal"].min()) - 0.05
    max_lat = float(well_frame["latitude_decimal"].max()) + 0.05
    step = float(basin["default_grid_step_deg"])
    max_dist_km = float(basin["default_grid_max_distance_km"])

    xs = np.arange(math.floor(min_lon / step) * step, math.ceil(max_lon / step) * step, step)
    ys = np.arange(math.floor(min_lat / step) * step, math.ceil(max_lat / step) * step, step)

    geology_geoms, geology_props, geology_tree = build_geology_lookup(canonical_dir / "bedrock_geology.geojson")
    dem_datasets = load_dem_datasets(repo_root / "data" / "raw" / "usgs" / "elevation_1")
    fault_geometries, fault_tree, fault_transformer = load_fault_index(repo_root / "data" / "raw" / "usgs" / "PAfaults_lcc.zip")

    rows = []
    for row_idx, west in enumerate(xs):
        for col_idx, south in enumerate(ys):
            east = west + step
            north = south + step
            center_lon = west + (step / 2.0)
            center_lat = south + (step / 2.0)
            nearest_well_km = nearest_neighbor_current_km(center_lat, center_lon, well_tree, exclude_self=False)
            if np.isnan(nearest_well_km) or nearest_well_km > max_dist_km:
                continue
            geology = assign_geology(center_lon, center_lat, geology_geoms, geology_props, geology_tree)
            terrain = sample_dem_features(center_lon, center_lat, dem_datasets)
            county_guess = well_frame.iloc[
                int(
                    well_tree.query(np.deg2rad([[center_lat, center_lon]]), k=1, return_distance=False)[0][0]
                )
            ]["county_name"]
            rows.append(
                {
                    "cell_id": f"{args.basin_id}_v1_{row_idx}_{col_idx}",
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "bbox_north": north,
                    "center_longitude": center_lon,
                    "center_latitude": center_lat,
                    "county_name": county_guess,
                    **geology,
                    **terrain,
                    "fault_distance_km": nearest_fault_distance_km(
                        center_lon,
                        center_lat,
                        fault_geometries,
                        fault_tree,
                        fault_transformer,
                    ),
                    "well_count_2km": count_neighbors_current(center_lat, center_lon, well_tree, 2.0),
                    "well_count_5km": count_neighbors_current(center_lat, center_lon, well_tree, 5.0),
                    "nearest_well_distance_km": nearest_well_km,
                    "permit_count_2km": count_neighbors_current(center_lat, center_lon, permit_tree, 2.0),
                    "permit_count_5km": count_neighbors_current(center_lat, center_lon, permit_tree, 5.0),
                    "nearest_permit_distance_km": nearest_neighbor_current_km(center_lat, center_lon, permit_tree),
                }
            )

    close_dem_datasets(dem_datasets)

    score_frame = pd.DataFrame(rows)
    feature_frame = score_frame.copy().rename(
        columns={
            "center_latitude": "latitude_decimal",
            "center_longitude": "longitude_decimal",
        }
    )[
        training_metadata["feature_columns_numeric"] + training_metadata["feature_columns_categorical"]
    ].copy()
    scores = pipeline.predict_proba(feature_frame)[:, 1]
    score_frame["score"] = scores
    score_frame["score_percentile"] = score_frame["score"].rank(method="average", pct=True)
    score_frame["prospect_tier"] = score_frame["score_percentile"].map(prospect_tier)
    score_frame["model_label_column"] = metrics.get("label_column", "label_recent12_ge_250000")
    score_frame["model_type"] = metrics.get("model_type", "lightgbm_classifier")
    score_frame["basin_id"] = args.basin_id
    score_frame = score_frame.sort_values(["score", "nearest_well_distance_km"], ascending=[False, True]).reset_index(
        drop=True
    )
    score_frame["score_rank"] = np.arange(1, len(score_frame) + 1)

    out_csv = derived_dir / "gas_prospect_cells_v1.csv"
    write_csv(score_frame, out_csv)

    features = []
    for row in score_frame.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "geometry": polygon_from_bbox(row.bbox_west, row.bbox_south, row.bbox_east, row.bbox_north),
                "properties": {
                    "cell_id": row.cell_id,
                    "score": round(float(row.score), 6),
                    "score_percentile": round(float(row.score_percentile), 6),
                    "score_rank": int(row.score_rank),
                    "prospect_tier": row.prospect_tier,
                    "county_name": row.county_name,
                    "geology_name": row.geology_name,
                    "geology_lith1": row.geology_lith1,
                    "elevation_m": round(float(row.elevation_m), 3) if pd.notna(row.elevation_m) else None,
                    "slope_deg": round(float(row.slope_deg), 3) if pd.notna(row.slope_deg) else None,
                    "fault_distance_km": round(float(row.fault_distance_km), 3) if pd.notna(row.fault_distance_km) else None,
                    "well_count_5km": int(row.well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                    "nearest_well_distance_km": round(float(row.nearest_well_distance_km), 3)
                    if pd.notna(row.nearest_well_distance_km)
                    else None,
                },
            }
        )
    geojson_payload = {"type": "FeatureCollection", "features": features}
    out_geojson = derived_dir / "gas_prospect_cells_v1.geojson"
    out_geojson.write_text(json.dumps(geojson_payload), encoding="utf-8")

    top_cols = [
        "cell_id",
        "score_rank",
        "score",
        "score_percentile",
        "prospect_tier",
        "county_name",
        "geology_name",
        "geology_lith1",
        "fault_distance_km",
        "well_count_5km",
        "permit_count_5km",
        "center_longitude",
        "center_latitude",
    ]
    top_path = derived_dir / "top_prospects_v1.csv"
    write_csv(score_frame[top_cols].head(100), top_path)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prospect_cells_v1",
        "basin_id": args.basin_id,
        "basin_name": basin["display_name"],
        "row_count": int(len(score_frame)),
        "grid_step_deg": step,
        "max_distance_km": max_dist_km,
        "score_min": float(score_frame["score"].min()),
        "score_p50": float(score_frame["score"].median()),
        "score_p90": float(score_frame["score"].quantile(0.9)),
        "score_max": float(score_frame["score"].max()),
        "tier_counts": {tier: int(count) for tier, count in score_frame["prospect_tier"].value_counts().sort_index().items()},
        "model_path": str(model_path),
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "top_prospects_path": str(top_path),
        "output_csv_path": str(out_csv),
        "output_geojson_path": str(out_geojson),
    }
    meta_path = derived_dir / "gas_prospect_cells_v1_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
