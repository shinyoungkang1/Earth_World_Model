#!/usr/bin/env python3
"""Build the basin-scoped gas v1 training table for the PA MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gas_v1_common import (
    assign_geology,
    build_balltree,
    build_geology_lookup,
    close_dem_datasets,
    count_neighbors_before_date,
    load_basin_config,
    load_dem_datasets,
    load_fault_index,
    nearest_fault_distance_km,
    nearest_neighbor_before_date_km,
    parse_date,
    sample_dem_features,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    counties = set(basin["counties"])

    canonical_dir = repo_root / "data" / "canonical" / "pa_mvp"
    raw_dir = repo_root / "data" / "raw"
    features_dir = repo_root / "data" / "features" / args.basin_id
    features_dir.mkdir(parents=True, exist_ok=True)

    wells = pd.read_csv(canonical_dir / "wells.csv")
    permits = pd.read_csv(canonical_dir / "permits.csv")
    production = pd.read_csv(canonical_dir / "production.csv")

    wells = wells[(wells["unconventional"] == True) & (wells["county_name"].isin(counties))].copy()
    wells = wells.dropna(subset=["latitude_decimal", "longitude_decimal"]).copy()
    wells["spud_date"] = parse_date(wells["spud_date"])
    wells.loc[wells["spud_date"] < pd.Timestamp("1950-01-01"), "spud_date"] = pd.NaT

    permits = permits[permits["county"].isin(counties)].copy()
    permits["permit_issued_date"] = parse_date(permits["permit_issued_date"])
    permits["spud_date"] = parse_date(permits["spud_date"])

    permit_agg = (
        permits.groupby("well_api", dropna=True)
        .agg(
            permit_count=("authorization_id", "size"),
            first_permit_issued_date=("permit_issued_date", "min"),
            latest_permit_issued_date=("permit_issued_date", "max"),
            first_permit_spud_date=("spud_date", "min"),
        )
        .reset_index()
    )
    permit_agg["has_permit_record"] = True

    prod = production[
        (production["production_scope"] == "unconventional") & (production["permit_num"].isin(wells["well_api"]))
    ].copy()
    prod["production_period_start_date"] = parse_date(prod["production_period_start_date"])
    prod["production_period_end_date"] = parse_date(prod["production_period_end_date"])
    prod["spud_date"] = parse_date(prod["spud_date"])
    prod.loc[prod["spud_date"] < pd.Timestamp("1950-01-01"), "spud_date"] = pd.NaT

    prod_window_start = prod["production_period_start_date"].min()
    prod_window_end = prod["production_period_end_date"].max()
    prod_agg = (
        prod.groupby("permit_num", dropna=True)
        .agg(
            observed_recent12_rows=("permit_num", "size"),
            observed_recent12_gas=("gas_quantity", "sum"),
            observed_recent12_gas_avg_monthly=("gas_quantity", "mean"),
            observed_recent12_gas_max_monthly=("gas_quantity", "max"),
            observed_first_prod_date=("production_period_start_date", "min"),
            observed_last_prod_date=("production_period_end_date", "max"),
            prod_spud_date=("spud_date", "min"),
        )
        .reset_index()
        .rename(columns={"permit_num": "well_api"})
    )
    prod_agg["has_unconv_prod_record"] = True

    table = wells.merge(permit_agg, on="well_api", how="left")
    table = table.merge(prod_agg, on="well_api", how="left")

    table["has_permit_record"] = table["has_permit_record"].eq(True)
    table["has_unconv_prod_record"] = table["has_unconv_prod_record"].eq(True)
    table["observed_recent12_gas"] = table["observed_recent12_gas"].fillna(0.0)
    table["observed_recent12_gas_avg_monthly"] = table["observed_recent12_gas_avg_monthly"].fillna(0.0)
    table["observed_recent12_gas_max_monthly"] = table["observed_recent12_gas_max_monthly"].fillna(0.0)
    table["observed_recent12_rows"] = table["observed_recent12_rows"].fillna(0).astype(int)
    table["context_date"] = table["spud_date"].combine_first(table["first_permit_issued_date"])

    developed_statuses = {
        "Active",
        "Plugged OG Well",
        "Regulatory Inactive Status",
        "Abandoned",
    }
    excluded_statuses = {
        "Operator Reported Not Drilled",
        "Proposed But Never Materialized",
        "Duplicate",
    }
    table["is_developed_status"] = table["well_status"].isin(developed_statuses)
    table["is_excluded_status"] = table["well_status"].isin(excluded_statuses)
    table["trainable_label_v1"] = (
        (table["is_developed_status"] | table["has_unconv_prod_record"]) & ~table["is_excluded_status"]
    )
    table["label_recent12_gt_0"] = (table["observed_recent12_gas"] > 0).where(table["trainable_label_v1"], pd.NA)
    table["label_recent12_ge_100000"] = (table["observed_recent12_gas"] >= 100000).where(
        table["trainable_label_v1"], pd.NA
    )
    table["label_recent12_ge_250000"] = (table["observed_recent12_gas"] >= 250000).where(
        table["trainable_label_v1"], pd.NA
    )
    table["label_recent12_ge_500000"] = (table["observed_recent12_gas"] >= 500000).where(
        table["trainable_label_v1"], pd.NA
    )
    table["label_recent12_ge_1000000"] = (table["observed_recent12_gas"] >= 1000000).where(
        table["trainable_label_v1"], pd.NA
    )

    table["label_f12_like_available"] = (
        table["has_unconv_prod_record"]
        & table["spud_date"].notna()
        & table["observed_first_prod_date"].notna()
        & ((table["observed_first_prod_date"] - table["spud_date"]).dt.days.between(0, 180, inclusive="both"))
        & (table["observed_recent12_rows"] >= 10)
    )
    table["label_f24_available"] = False

    geology_geoms, geology_props, geology_tree = build_geology_lookup(canonical_dir / "bedrock_geology.geojson")
    geology_rows = []
    for row in table[["well_api", "longitude_decimal", "latitude_decimal"]].itertuples(index=False):
        geology_rows.append({"well_api": row.well_api, **assign_geology(row.longitude_decimal, row.latitude_decimal, geology_geoms, geology_props, geology_tree)})
    table = table.merge(pd.DataFrame(geology_rows), on="well_api", how="left")

    dem_datasets = load_dem_datasets(raw_dir / "usgs" / "elevation_1")
    fault_geometries, fault_tree, fault_transformer = load_fault_index(raw_dir / "usgs" / "PAfaults_lcc.zip")
    terrain_rows = []
    for row in table[["well_api", "longitude_decimal", "latitude_decimal"]].itertuples(index=False):
        terrain = sample_dem_features(row.longitude_decimal, row.latitude_decimal, dem_datasets)
        terrain["fault_distance_km"] = nearest_fault_distance_km(
            row.longitude_decimal,
            row.latitude_decimal,
            fault_geometries,
            fault_tree,
            fault_transformer,
        )
        terrain_rows.append({"well_api": row.well_api, **terrain})
    close_dem_datasets(dem_datasets)
    table = table.merge(pd.DataFrame(terrain_rows), on="well_api", how="left")

    well_context = table.dropna(subset=["context_date"]).copy()
    well_context["context_date"] = parse_date(well_context["context_date"])
    well_context, well_tree = build_balltree(well_context, "latitude_decimal", "longitude_decimal")

    permit_context = permits.dropna(subset=["permit_issued_date", "latitude", "longitude"]).copy()
    permit_context["permit_issued_date"] = parse_date(permit_context["permit_issued_date"])
    permit_context, permit_tree = build_balltree(permit_context, "latitude", "longitude")

    context_rows = []
    for row in table[
        ["well_api", "latitude_decimal", "longitude_decimal", "context_date"]
    ].itertuples(index=False):
        target_date = pd.Timestamp(row.context_date) if pd.notna(row.context_date) else pd.NaT
        context_rows.append(
            {
                "well_api": row.well_api,
                "well_count_2km": count_neighbors_before_date(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    well_context,
                    well_tree,
                    "context_date",
                    "latitude_decimal",
                    "longitude_decimal",
                    2.0,
                ),
                "well_count_5km": count_neighbors_before_date(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    well_context,
                    well_tree,
                    "context_date",
                    "latitude_decimal",
                    "longitude_decimal",
                    5.0,
                ),
                "nearest_well_distance_km": nearest_neighbor_before_date_km(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    well_context,
                    well_tree,
                    "context_date",
                    "latitude_decimal",
                    "longitude_decimal",
                ),
                "permit_count_2km": count_neighbors_before_date(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                    2.0,
                ),
                "permit_count_5km": count_neighbors_before_date(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                    5.0,
                ),
                "nearest_permit_distance_km": nearest_neighbor_before_date_km(
                    row.latitude_decimal,
                    row.longitude_decimal,
                    target_date,
                    permit_context,
                    permit_tree,
                    "permit_issued_date",
                    "latitude",
                    "longitude",
                ),
            }
        )
    table = table.merge(pd.DataFrame(context_rows), on="well_api", how="left")

    keep_cols = [
        "well_api",
        "operator_name",
        "well_status",
        "well_type",
        "well_configuration",
        "county_name",
        "municipality_name",
        "latitude_decimal",
        "longitude_decimal",
        "spud_date",
        "context_date",
        "permit_count",
        "has_permit_record",
        "first_permit_issued_date",
        "latest_permit_issued_date",
        "first_permit_spud_date",
        "geology_map_symbol",
        "geology_name",
        "geology_age",
        "geology_lith1",
        "elevation_m",
        "slope_deg",
        "relief_3px_m",
        "fault_distance_km",
        "well_count_2km",
        "well_count_5km",
        "nearest_well_distance_km",
        "permit_count_2km",
        "permit_count_5km",
        "nearest_permit_distance_km",
        "has_unconv_prod_record",
        "observed_recent12_rows",
        "observed_recent12_gas",
        "observed_recent12_gas_avg_monthly",
        "observed_recent12_gas_max_monthly",
        "observed_first_prod_date",
        "observed_last_prod_date",
        "is_developed_status",
        "is_excluded_status",
        "trainable_label_v1",
        "label_recent12_gt_0",
        "label_recent12_ge_100000",
        "label_recent12_ge_250000",
        "label_recent12_ge_500000",
        "label_recent12_ge_1000000",
        "label_f12_like_available",
        "label_f24_available",
    ]
    table = table[keep_cols].sort_values(["county_name", "well_api"]).reset_index(drop=True)

    out_csv = features_dir / "gas_training_table_v1.csv"
    write_csv(table, out_csv)

    trainable = table[table["trainable_label_v1"] == True].copy()
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_v1",
        "basin_id": args.basin_id,
        "basin_counties": basin["counties"],
        "row_count_total": int(len(table)),
        "row_count_trainable_v1": int(len(trainable)),
        "row_count_with_recent_unconv_prod": int(table["has_unconv_prod_record"].sum()),
        "row_count_f12_like_available": int(table["label_f12_like_available"].sum()),
        "row_count_f24_available": int(table["label_f24_available"].sum()),
        "production_window_start": prod_window_start.strftime("%Y-%m-%d") if pd.notna(prod_window_start) else None,
        "production_window_end": prod_window_end.strftime("%Y-%m-%d") if pd.notna(prod_window_end) else None,
        "label_definition": "Recent 12-month unconventional gas sum over the currently ingested PA DEP monthly window; not a true first-life F12 label.",
        "label_positive_counts": {
            col: int(trainable[col].astype("boolean").fillna(False).sum())
            for col in [
                "label_recent12_gt_0",
                "label_recent12_ge_100000",
                "label_recent12_ge_250000",
                "label_recent12_ge_500000",
                "label_recent12_ge_1000000",
            ]
        },
        "feature_columns_numeric": [
            "latitude_decimal",
            "longitude_decimal",
            "elevation_m",
            "slope_deg",
            "relief_3px_m",
            "fault_distance_km",
            "well_count_2km",
            "well_count_5km",
            "nearest_well_distance_km",
            "permit_count_2km",
            "permit_count_5km",
            "nearest_permit_distance_km",
        ],
        "feature_columns_categorical": [
            "geology_map_symbol",
            "geology_age",
            "geology_lith1",
        ],
        "output_path": str(out_csv),
    }
    meta_path = features_dir / "gas_training_table_v1_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
