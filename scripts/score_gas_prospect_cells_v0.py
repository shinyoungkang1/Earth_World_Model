#!/usr/bin/env python3
"""Score a coarse Pennsylvania gas prospect grid using the baseline model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_geology_lookup(geology_path: Path):
    payload = load_json(geology_path)
    geoms = []
    props = []
    for feature in payload.get("features", []):
        geom = shape(feature["geometry"])
        geoms.append(geom)
        props.append(feature.get("properties", {}))
    tree = STRtree(geoms)
    return geoms, props, tree


def assign_geology(center_lon: float, center_lat: float, geoms, props, tree) -> dict:
    point = Point(center_lon, center_lat)
    for idx in tree.query(point):
        if geoms[int(idx)].covers(point):
            match = props[int(idx)]
            return {
                "geology_map_symbol": match.get("map_symbol"),
                "geology_name": match.get("name"),
                "geology_age": match.get("age"),
                "geology_lith1": match.get("lith1"),
            }
    return {
        "geology_map_symbol": None,
        "geology_name": None,
        "geology_age": None,
        "geology_lith1": None,
    }


def polygon_from_bbox(minx: float, miny: float, maxx: float, maxy: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def prospect_tier(percentile: float) -> str:
    if percentile >= 0.9:
        return "very_high"
    if percentile >= 0.7:
        return "high"
    if percentile >= 0.4:
        return "medium"
    return "background"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument(
        "--model-path",
        default="models/pa_mvp/gas_baseline_v0_label_obs12_ge_250000.joblib",
    )
    parser.add_argument("--grid-step-deg", type=float, default=0.05)
    parser.add_argument("--max-distance-km", type=float, default=25.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    canonical_dir = repo_root / "data" / "canonical" / "pa_mvp"
    features_dir = repo_root / "data" / "features" / "pa_mvp"
    derived_dir = repo_root / "data" / "derived" / "prospect_layers"
    derived_dir.mkdir(parents=True, exist_ok=True)

    model_path = repo_root / args.model_path
    pipeline = joblib.load(model_path)
    metrics_path = model_path.with_name(model_path.stem + "_metrics.json")
    metrics = load_json(metrics_path) if metrics_path.exists() else {}

    training = pd.read_csv(features_dir / "gas_training_table_v0.csv")
    wells = pd.read_csv(canonical_dir / "wells.csv")
    wells = wells.dropna(subset=["longitude_decimal", "latitude_decimal"]).copy()
    wells["county_name"] = wells["county_name"].fillna("UNKNOWN_COUNTY")
    wells["municipality_name"] = wells["municipality_name"].fillna("UNKNOWN_MUNICIPALITY")

    reference_year = int(training.loc[training["trainable_label_v0"] == True, "spud_year"].dropna().median())
    reference_month = int(training.loc[training["trainable_label_v0"] == True, "spud_month"].dropna().median())

    min_lon = float(wells["longitude_decimal"].min())
    max_lon = float(wells["longitude_decimal"].max())
    min_lat = float(wells["latitude_decimal"].min())
    max_lat = float(wells["latitude_decimal"].max())
    step = float(args.grid_step_deg)

    west_start = math.floor(min_lon / step) * step
    south_start = math.floor(min_lat / step) * step
    east_end = math.ceil(max_lon / step) * step
    north_end = math.ceil(max_lat / step) * step

    xs = np.arange(west_start, east_end, step)
    ys = np.arange(south_start, north_end, step)

    well_coords_rad = np.deg2rad(wells[["latitude_decimal", "longitude_decimal"]].to_numpy())
    nearest_tree = BallTree(well_coords_rad, metric="haversine")
    well_rows = wells.reset_index(drop=True)

    geoms, props, geology_tree = build_geology_lookup(canonical_dir / "bedrock_geology.geojson")

    rows = []
    for row_idx, west in enumerate(xs):
        for col_idx, south in enumerate(ys):
            east = west + step
            north = south + step
            center_lon = west + (step / 2.0)
            center_lat = south + (step / 2.0)
            center_coords_rad = np.deg2rad([[center_lat, center_lon]])
            dist_rad, ind = nearest_tree.query(center_coords_rad, k=1)
            nearest_idx = int(ind[0][0])
            nearest_distance_km = float(dist_rad[0][0] * EARTH_RADIUS_KM)
            if nearest_distance_km > args.max_distance_km:
                continue

            geology = assign_geology(center_lon, center_lat, geoms, props, geology_tree)
            nearest_well = well_rows.iloc[nearest_idx]
            rows.append(
                {
                    "cell_id": f"gas_v0_{row_idx}_{col_idx}",
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "bbox_north": north,
                    "center_longitude": center_lon,
                    "center_latitude": center_lat,
                    "county_name": nearest_well["county_name"],
                    "municipality_name": nearest_well["municipality_name"],
                    "nearest_well_api": nearest_well["well_api"],
                    "nearest_well_distance_km": nearest_distance_km,
                    "spud_year": reference_year,
                    "spud_month": reference_month,
                    **geology,
                }
            )

    score_frame = pd.DataFrame(rows)
    feature_frame = score_frame[
        [
            "center_latitude",
            "center_longitude",
            "spud_year",
            "spud_month",
            "county_name",
            "municipality_name",
            "geology_map_symbol",
            "geology_age",
            "geology_lith1",
        ]
    ].rename(
        columns={
            "center_latitude": "latitude_decimal",
            "center_longitude": "longitude_decimal",
        }
    )

    scores = pipeline.predict_proba(feature_frame)[:, 1]
    score_frame["score"] = scores
    score_frame["score_percentile"] = score_frame["score"].rank(method="average", pct=True)
    score_frame["prospect_tier"] = score_frame["score_percentile"].map(prospect_tier)
    score_frame["model_label_column"] = metrics.get("label_column", "label_obs12_ge_250000")
    score_frame["model_type"] = metrics.get("model_type", "logistic_regression")
    score_frame["model_path"] = str(model_path)

    score_frame = score_frame.sort_values(["score", "nearest_well_distance_km"], ascending=[False, True]).reset_index(
        drop=True
    )
    score_frame["score_rank"] = np.arange(1, len(score_frame) + 1)

    out_csv = derived_dir / "gas_prospect_cells_v0.csv"
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
                    "municipality_name": row.municipality_name,
                    "geology_map_symbol": row.geology_map_symbol,
                    "geology_name": row.geology_name,
                    "geology_age": row.geology_age,
                    "geology_lith1": row.geology_lith1,
                    "nearest_well_api": row.nearest_well_api,
                    "nearest_well_distance_km": round(float(row.nearest_well_distance_km), 3),
                    "center_longitude": round(float(row.center_longitude), 6),
                    "center_latitude": round(float(row.center_latitude), 6),
                },
            }
        )
    geojson_payload = {"type": "FeatureCollection", "features": features}
    out_geojson = derived_dir / "gas_prospect_cells_v0.geojson"
    out_geojson.write_text(json.dumps(geojson_payload), encoding="utf-8")

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prospect_cells_v0",
        "row_count": int(len(score_frame)),
        "grid_step_deg": step,
        "max_distance_km": args.max_distance_km,
        "reference_spud_year": reference_year,
        "reference_spud_month": reference_month,
        "score_min": float(score_frame["score"].min()),
        "score_p50": float(score_frame["score"].median()),
        "score_p90": float(score_frame["score"].quantile(0.9)),
        "score_max": float(score_frame["score"].max()),
        "tier_counts": {
            tier: int(count)
            for tier, count in score_frame["prospect_tier"].value_counts().sort_index().items()
        },
        "model_path": str(model_path),
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "output_csv_path": str(out_csv),
        "output_geojson_path": str(out_geojson),
    }
    out_meta = derived_dir / "gas_prospect_cells_v0_metadata.json"
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
