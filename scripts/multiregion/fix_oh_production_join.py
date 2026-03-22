#!/usr/bin/env python3
"""Fix OH production well_api join by spatially matching FracTracker wells to ODNR wells.

The FracTracker production GeoJSON uses county-level API codes (e.g. 34067200000000)
combined with well names as composite keys. ODNR wells use proper 14-digit APIs.
This script matches FracTracker wells to ODNR wells by spatial proximity and
rewrites production.csv with correct ODNR well_api values.

Usage:
    python scripts/multiregion/fix_oh_production_join.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
MAX_MATCH_DISTANCE_KM = 2.0


def main() -> int:
    repo_root = Path("/home/shin/Mineral_Gas_Locator")
    raw_dir = repo_root / "data" / "raw"
    canonical_dir = repo_root / "data" / "canonical" / "oh_mvp"

    # ---- Load ODNR wells (target for matching) ----
    wells = pd.read_csv(canonical_dir / "wells.csv", low_memory=False)
    wells["latitude"] = pd.to_numeric(wells["latitude"], errors="coerce")
    wells["longitude"] = pd.to_numeric(wells["longitude"], errors="coerce")
    wells = wells.dropna(subset=["latitude", "longitude", "well_api"]).copy()
    wells["well_api"] = wells["well_api"].astype(str)
    print(f"ODNR wells: {len(wells)}")

    # Build BallTree from ODNR wells
    coords_rad = np.deg2rad(wells[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")

    # ---- Load FracTracker production GeoJSON ----
    frac_path = raw_dir / "oh_odnr" / "oh_utica_marcellus_production_q2_2021.geojson"
    with open(frac_path, encoding="utf-8") as f:
        frac_data = json.load(f)
    print(f"FracTracker features: {len(frac_data['features'])}")

    # ---- Extract unique FracTracker wells with coordinates ----
    frac_wells: dict[str, dict] = {}
    for feat in frac_data["features"]:
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        api = str(props.get("API", "")).strip()
        name = str(props.get("WellName_N", "")).strip()
        composite_key = f"{api}_{name.replace(' ', '_')}" if name else api
        if composite_key not in frac_wells and coords[0] is not None:
            frac_wells[composite_key] = {
                "composite_key": composite_key,
                "latitude": float(props.get("SurfLat", coords[1])),
                "longitude": float(props.get("SurfLon", coords[0])),
                "well_name": name,
                "county": props.get("County", ""),
            }

    print(f"Unique FracTracker wells: {len(frac_wells)}")

    # ---- Spatial matching ----
    matched = 0
    unmatched = 0
    key_map: dict[str, str] = {}  # composite_key → ODNR well_api

    for composite_key, frac in frac_wells.items():
        lat, lon = frac["latitude"], frac["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            unmatched += 1
            continue

        query = np.deg2rad([[lat, lon]])
        dist, idx = tree.query(query, k=1, return_distance=True)
        dist_km = float(dist[0][0]) * EARTH_RADIUS_KM
        nearest_idx = int(idx[0][0])

        if dist_km <= MAX_MATCH_DISTANCE_KM:
            odnr_api = wells.iloc[nearest_idx]["well_api"]
            key_map[composite_key] = str(odnr_api)
            matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")
    print(f"Match rate: {matched / (matched + unmatched) * 100:.1f}%")

    # ---- Rebuild production.csv with ODNR well_api ----
    production = pd.read_csv(canonical_dir / "production.csv", low_memory=False)
    original_count = len(production)
    production["well_api_original"] = production["well_api"]
    production["well_api"] = production["well_api"].map(key_map)

    # Drop rows that couldn't be matched
    matched_prod = production.dropna(subset=["well_api"]).copy()
    dropped = original_count - len(matched_prod)
    print(f"\nProduction rows: {original_count}")
    print(f"Matched to ODNR: {len(matched_prod)}")
    print(f"Dropped (no match): {dropped}")
    print(f"Unique ODNR wells with production: {matched_prod['well_api'].nunique()}")

    # Write fixed production
    output_cols = [c for c in matched_prod.columns if c != "well_api_original"]
    matched_prod[output_cols].to_csv(canonical_dir / "production.csv", index=False)
    print(f"\nWrote: {canonical_dir / 'production.csv'}")

    # Write the mapping for reference
    mapping_df = pd.DataFrame([
        {"composite_key": k, "odnr_well_api": v}
        for k, v in key_map.items()
    ])
    mapping_path = canonical_dir / "fractracker_to_odnr_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Wrote mapping: {mapping_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
