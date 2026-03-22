#!/usr/bin/env python3
"""Score the first WV horizontal gas prospect grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wv_gas_common import (
    build_balltree,
    clean_formation,
    clean_flag,
    count_neighbors_current,
    county_code_string,
    deduplicate_wells,
    load_basin_config,
    nearest_neighbor_current_km,
    nearest_row,
    neighbor_value_stats_current,
    polygon_from_bbox,
    prospect_tier,
    write_csv,
)


def json_value(value):
    return None if pd.isna(value) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="wv_horizontal_statewide")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    canonical_dir = repo_root / "data" / "canonical" / "wv_mvp"
    features_dir = repo_root / "data" / "features" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    derived_dir = repo_root / "data" / "derived" / args.basin_id
    derived_dir.mkdir(parents=True, exist_ok=True)

    training_metadata = json.loads((features_dir / "gas_training_table_wv_v0_metadata.json").read_text(encoding="utf-8"))
    model_path = (
        Path(args.model_path)
        if args.model_path
        else models_dir / "gas_baseline_wv_v0_label_f12_ge_2000000.joblib"
    )
    pipeline = joblib.load(model_path)
    metrics_path = model_path.with_name(model_path.stem + "_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    wells = deduplicate_wells(pd.read_csv(canonical_dir / "wells.csv"))
    wells["formation_name"] = clean_formation(wells["formation_name"])
    wells["marcellus_flag"] = clean_flag(wells["marcellus_flag"])
    training_table = pd.read_csv(features_dir / "gas_training_table_wv_v0.csv")
    producing = training_table[training_table["first_prod_date"].notna()].copy()
    mature = training_table[training_table["label_f12_available"] == True].copy()

    permits = pd.read_csv(canonical_dir / "permits.csv")
    permits["latitude"] = pd.to_numeric(permits["latitude"], errors="coerce")
    permits["longitude"] = pd.to_numeric(permits["longitude"], errors="coerce")
    permits = permits.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["permit_id"]).copy()

    well_frame, well_tree = build_balltree(wells, "latitude", "longitude")
    producing_frame, producing_tree = build_balltree(producing.dropna(subset=["latitude", "longitude"]).copy(), "latitude", "longitude")
    mature_frame, mature_tree = build_balltree(mature.dropna(subset=["latitude", "longitude"]).copy(), "latitude", "longitude")
    permit_frame, permit_tree = build_balltree(permits, "latitude", "longitude")

    min_lon = float(well_frame["longitude"].min()) - 0.05
    max_lon = float(well_frame["longitude"].max()) + 0.05
    min_lat = float(well_frame["latitude"].min()) - 0.05
    max_lat = float(well_frame["latitude"].max()) + 0.05
    step = float(basin["default_grid_step_deg"])
    max_dist_km = float(basin["default_grid_max_distance_km"])

    xs = np.arange(math.floor(min_lon / step) * step, math.ceil(max_lon / step) * step, step)
    ys = np.arange(math.floor(min_lat / step) * step, math.ceil(max_lat / step) * step, step)

    rows = []
    for row_idx, west in enumerate(xs):
        for col_idx, south in enumerate(ys):
            east = west + step
            north = south + step
            center_lon = west + (step / 2.0)
            center_lat = south + (step / 2.0)
            nearest_well_km = nearest_neighbor_current_km(center_lat, center_lon, producing_tree, exclude_self=False)
            if np.isnan(nearest_well_km) or nearest_well_km > max_dist_km:
                continue

            nearest = nearest_row(well_frame, well_tree, center_lat, center_lon)
            mature_stats_5 = neighbor_value_stats_current(
                center_lat,
                center_lon,
                mature_frame,
                mature_tree,
                "f12_gas",
                5.0,
            )
            mature_stats_10 = neighbor_value_stats_current(
                center_lat,
                center_lon,
                mature_frame,
                mature_tree,
                "f12_gas",
                10.0,
            )
            rows.append(
                {
                    "cell_id": f"{args.basin_id}_v0_{row_idx}_{col_idx}",
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "bbox_north": north,
                    "center_longitude": center_lon,
                    "center_latitude": center_lat,
                    "county_code": county_code_string(pd.Series([nearest["county_code"]])).iloc[0],
                    "formation_name": nearest["formation_name"],
                    "marcellus_flag": nearest["marcellus_flag"],
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "producing_well_count_2km": count_neighbors_current(center_lat, center_lon, producing_tree, 2.0),
                    "producing_well_count_5km": count_neighbors_current(center_lat, center_lon, producing_tree, 5.0),
                    "nearest_producing_well_distance_km": nearest_well_km,
                    "permit_count_2km": count_neighbors_current(center_lat, center_lon, permit_tree, 2.0),
                    "permit_count_5km": count_neighbors_current(center_lat, center_lon, permit_tree, 5.0),
                    "nearest_permit_distance_km": nearest_neighbor_current_km(center_lat, center_lon, permit_tree),
                    "mature_f12_neighbor_count_5km": mature_stats_5["count"],
                    "mature_f12_neighbor_median_gas_5km": mature_stats_5["median"],
                    "mature_f12_neighbor_p90_gas_5km": mature_stats_5["p90"],
                    "mature_f12_neighbor_count_10km": mature_stats_10["count"],
                    "mature_f12_neighbor_median_gas_10km": mature_stats_10["median"],
                    "mature_f12_neighbor_p90_gas_10km": mature_stats_10["p90"],
                }
            )

    score_frame = pd.DataFrame(rows)
    feature_frame = score_frame[
        training_metadata["feature_columns_numeric"] + training_metadata["feature_columns_categorical"]
    ].copy()
    feature_frame = feature_frame.replace({pd.NA: np.nan})
    scores = pipeline.predict_proba(feature_frame)[:, 1]
    score_frame["score"] = scores
    score_frame["score_percentile"] = score_frame["score"].rank(method="average", pct=True)
    score_frame["prospect_tier"] = score_frame["score_percentile"].map(prospect_tier)
    score_frame["model_label_column"] = metrics.get("label_column", "label_f12_ge_2000000")
    score_frame["model_type"] = metrics.get("model_type", "xgboost_classifier_gpu")
    score_frame["basin_id"] = args.basin_id
    score_frame = score_frame.sort_values(["score", "nearest_producing_well_distance_km"], ascending=[False, True]).reset_index(drop=True)
    score_frame["score_rank"] = np.arange(1, len(score_frame) + 1)

    out_csv = derived_dir / "gas_prospect_cells_wv_v0.csv"
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
                    "county_code": json_value(row.county_code),
                    "formation_name": json_value(row.formation_name),
                    "marcellus_flag": json_value(row.marcellus_flag),
                    "producing_well_count_5km": int(row.producing_well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                    "mature_f12_neighbor_median_gas_5km": round(float(row.mature_f12_neighbor_median_gas_5km), 2)
                    if pd.notna(row.mature_f12_neighbor_median_gas_5km)
                    else None,
                    "nearest_producing_well_distance_km": round(float(row.nearest_producing_well_distance_km), 3)
                    if pd.notna(row.nearest_producing_well_distance_km)
                    else None,
                },
            }
        )
    out_geojson = derived_dir / "gas_prospect_cells_wv_v0.geojson"
    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")

    top_cols = [
        "cell_id",
        "score_rank",
        "score",
        "score_percentile",
        "prospect_tier",
        "county_code",
        "formation_name",
        "marcellus_flag",
        "mature_f12_neighbor_median_gas_5km",
        "producing_well_count_5km",
        "permit_count_5km",
        "center_longitude",
        "center_latitude",
    ]
    top_path = derived_dir / "top_prospects_wv_v0.csv"
    write_csv(score_frame[top_cols].head(100), top_path)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prospect_cells_wv_v0",
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
    meta_path = derived_dir / "gas_prospect_cells_wv_v0_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
