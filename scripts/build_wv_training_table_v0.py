#!/usr/bin/env python3
"""Build the first WV horizontal gas training table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wv_gas_common import (
    build_balltree,
    build_first_life_summary,
    clean_formation,
    clean_flag,
    county_code_string,
    count_neighbors_before_date,
    deduplicate_wells,
    load_basin_config,
    nearest_neighbor_before_date_km,
    neighbor_value_stats_before_date,
    parse_date,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="wv_horizontal_statewide")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    canonical_dir = repo_root / "data" / "canonical" / "wv_mvp"
    features_dir = repo_root / "data" / "features" / args.basin_id
    features_dir.mkdir(parents=True, exist_ok=True)

    wells_raw = pd.read_csv(canonical_dir / "wells.csv")
    permits_raw = pd.read_csv(canonical_dir / "permits.csv")
    production = pd.read_csv(canonical_dir / "production.csv")

    wells = deduplicate_wells(wells_raw)
    wells["formation_name"] = clean_formation(wells["formation_name"])
    wells["marcellus_flag"] = clean_flag(wells["marcellus_flag"])
    wells["permit_year"] = parse_date(wells["permit_issued_date"]).dt.year
    wells["completion_year"] = parse_date(wells["completion_date"]).dt.year
    wells["record_year"] = parse_date(wells["record_date"]).dt.year

    permits = permits_raw.copy()
    permits["county_code"] = county_code_string(permits["county"])
    permits["permit_issued_date"] = parse_date(permits["permit_issued_date"])
    permits["latitude"] = pd.to_numeric(permits["latitude"], errors="coerce")
    permits["longitude"] = pd.to_numeric(permits["longitude"], errors="coerce")
    permits = permits.dropna(subset=["permit_issued_date", "latitude", "longitude"]).copy()
    permits = permits.drop_duplicates(subset=["permit_id"]).copy()

    first_life = build_first_life_summary(production)
    table = wells.merge(first_life, on="well_api", how="left")
    table["context_date"] = parse_date(table["first_prod_date"])
    table["trainable_label_wv_v0"] = table["label_f12_available"].eq(True)

    for threshold in [500000, 1000000, 2000000, 5000000]:
        table[f"label_f12_ge_{threshold}"] = (table["f12_gas"] >= threshold).where(
            table["label_f12_available"].eq(True),
            pd.NA,
        )
    for threshold in [1000000, 2000000, 5000000]:
        table[f"label_f24_ge_{threshold}"] = (table["f24_gas"] >= threshold).where(
            table["label_f24_available"].eq(True),
            pd.NA,
        )

    producing_context = table.dropna(subset=["context_date", "latitude", "longitude"]).copy()
    producing_context, producing_tree = build_balltree(producing_context, "latitude", "longitude")
    permit_context, permit_tree = build_balltree(permits, "latitude", "longitude")
    mature_context = table[
        table["label_f12_available"].eq(True) & table["context_date"].notna() & table["latitude"].notna() & table["longitude"].notna()
    ].copy()
    mature_context, mature_tree = build_balltree(mature_context, "latitude", "longitude")

    feature_rows = []
    for row in table[
        ["well_api", "latitude", "longitude", "context_date"]
    ].itertuples(index=False):
        target_date = pd.Timestamp(row.context_date) if pd.notna(row.context_date) else pd.NaT
        mature_stats_5 = neighbor_value_stats_before_date(
            row.latitude,
            row.longitude,
            target_date,
            mature_context,
            mature_tree,
            "context_date",
            "latitude",
            "longitude",
            "f12_gas",
            5.0,
        )
        mature_stats_10 = neighbor_value_stats_before_date(
            row.latitude,
            row.longitude,
            target_date,
            mature_context,
            mature_tree,
            "context_date",
            "latitude",
            "longitude",
            "f12_gas",
            10.0,
        )
        feature_rows.append(
            {
                "well_api": row.well_api,
                "producing_well_count_2km": count_neighbors_before_date(
                    row.latitude,
                    row.longitude,
                    target_date,
                    producing_context,
                    producing_tree,
                    "context_date",
                    "latitude",
                    "longitude",
                    2.0,
                ),
                "producing_well_count_5km": count_neighbors_before_date(
                    row.latitude,
                    row.longitude,
                    target_date,
                    producing_context,
                    producing_tree,
                    "context_date",
                    "latitude",
                    "longitude",
                    5.0,
                ),
                "nearest_producing_well_distance_km": nearest_neighbor_before_date_km(
                    row.latitude,
                    row.longitude,
                    target_date,
                    producing_context,
                    producing_tree,
                    "context_date",
                    "latitude",
                    "longitude",
                ),
                "permit_count_2km": count_neighbors_before_date(
                    row.latitude,
                    row.longitude,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                    2.0,
                ),
                "permit_count_5km": count_neighbors_before_date(
                    row.latitude,
                    row.longitude,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                    5.0,
                ),
                "nearest_permit_distance_km": nearest_neighbor_before_date_km(
                    row.latitude,
                    row.longitude,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                ),
                "mature_f12_neighbor_count_5km": mature_stats_5["count"],
                "mature_f12_neighbor_median_gas_5km": mature_stats_5["median"],
                "mature_f12_neighbor_p90_gas_5km": mature_stats_5["p90"],
                "mature_f12_neighbor_count_10km": mature_stats_10["count"],
                "mature_f12_neighbor_median_gas_10km": mature_stats_10["median"],
                "mature_f12_neighbor_p90_gas_10km": mature_stats_10["p90"],
            }
        )

    table = table.merge(pd.DataFrame(feature_rows), on="well_api", how="left")
    table["county_code"] = county_code_string(table["county_code"])
    table["formation_name"] = clean_formation(table["formation_name"])
    table["marcellus_flag"] = clean_flag(table["marcellus_flag"])

    feature_columns_numeric = [
        "latitude",
        "longitude",
        "producing_well_count_2km",
        "producing_well_count_5km",
        "nearest_producing_well_distance_km",
        "permit_count_2km",
        "permit_count_5km",
        "nearest_permit_distance_km",
        "mature_f12_neighbor_count_5km",
        "mature_f12_neighbor_median_gas_5km",
        "mature_f12_neighbor_p90_gas_5km",
        "mature_f12_neighbor_count_10km",
        "mature_f12_neighbor_median_gas_10km",
        "mature_f12_neighbor_p90_gas_10km",
    ]
    feature_columns_categorical = [
        "county_code",
        "formation_name",
        "marcellus_flag",
    ]

    output_path = features_dir / "gas_training_table_wv_v0.csv"
    write_csv(table, output_path)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_wv_v0",
        "basin_id": args.basin_id,
        "basin_name": basin["display_name"],
        "row_count_total": int(len(table)),
        "row_count_trainable_f12": int(table["label_f12_available"].eq(True).sum()),
        "row_count_trainable_f24": int(table["label_f24_available"].eq(True).sum()),
        "row_count_with_first_prod": int(table["first_prod_date"].notna().sum()),
        "first_prod_start": str(pd.to_datetime(table["first_prod_date"], errors="coerce").min().date()),
        "first_prod_end": str(pd.to_datetime(table["first_prod_date"], errors="coerce").max().date()),
        "label_definition": "True first-life gas labels anchored on first non-zero gas month from WVDEP H6A production history.",
        "feature_columns_numeric": feature_columns_numeric,
        "feature_columns_categorical": feature_columns_categorical,
        "label_positive_counts": {
            key: int(table[key].eq(True).sum())
            for key in [
                "label_f12_ge_500000",
                "label_f12_ge_1000000",
                "label_f12_ge_2000000",
                "label_f12_ge_5000000",
                "label_f24_ge_1000000",
                "label_f24_ge_2000000",
                "label_f24_ge_5000000",
            ]
        },
        "output_path": str(output_path),
    }
    meta_path = features_dir / "gas_training_table_wv_v0_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
