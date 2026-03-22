#!/usr/bin/env python3
"""Build the first gas training table for the PA MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def date_floor(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


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


def assign_geology(
    wells: pd.DataFrame,
    geology_path: Path,
) -> pd.DataFrame:
    geoms, props, tree = build_geology_lookup(geology_path)
    rows = []
    for row in wells[["well_api", "longitude_decimal", "latitude_decimal"]].itertuples(index=False):
        if pd.isna(row.longitude_decimal) or pd.isna(row.latitude_decimal):
            rows.append(
                {
                    "well_api": row.well_api,
                    "geology_map_symbol": pd.NA,
                    "geology_name": pd.NA,
                    "geology_age": pd.NA,
                    "geology_lith1": pd.NA,
                }
            )
            continue
        point = Point(float(row.longitude_decimal), float(row.latitude_decimal))
        match = None
        for idx in tree.query(point):
            if geoms[int(idx)].covers(point):
                match = props[int(idx)]
                break
        rows.append(
            {
                "well_api": row.well_api,
                "geology_map_symbol": (match or {}).get("map_symbol"),
                "geology_name": (match or {}).get("name"),
                "geology_age": (match or {}).get("age"),
                "geology_lith1": (match or {}).get("lith1"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    canonical_dir = repo_root / "data" / "canonical" / "pa_mvp"
    features_dir = repo_root / "data" / "features" / "pa_mvp"
    features_dir.mkdir(parents=True, exist_ok=True)

    wells = pd.read_csv(canonical_dir / "wells.csv")
    permits = pd.read_csv(canonical_dir / "permits.csv")
    production = pd.read_csv(canonical_dir / "production.csv")

    wells = wells[wells["unconventional"] == True].copy()
    wells["spud_date"] = date_floor(wells["spud_date"])
    wells["spud_year"] = pd.to_datetime(wells["spud_date"], errors="coerce").dt.year
    wells["spud_month"] = pd.to_datetime(wells["spud_date"], errors="coerce").dt.month
    wells["has_spud_date"] = wells["spud_date"].notna()

    permits["permit_issued_date"] = date_floor(permits["permit_issued_date"])
    permits["spud_date"] = date_floor(permits["spud_date"])
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

    unconv_prod = production[production["production_scope"] == "unconventional"].copy()
    unconv_prod["production_period_start_date"] = date_floor(unconv_prod["production_period_start_date"])
    unconv_prod["production_period_end_date"] = date_floor(unconv_prod["production_period_end_date"])
    prod_window_start = unconv_prod["production_period_start_date"].min()
    prod_window_end = unconv_prod["production_period_end_date"].max()
    prod_agg = (
        unconv_prod.groupby("permit_num", dropna=True)
        .agg(
            observed_monthly_rows=("permit_num", "size"),
            observed_gas_12mo=("gas_quantity", "sum"),
            observed_gas_avg_monthly=("gas_quantity", "mean"),
            observed_gas_max_monthly=("gas_quantity", "max"),
            observed_first_prod_date=("production_period_start_date", "min"),
            observed_last_prod_date=("production_period_end_date", "max"),
        )
        .reset_index()
        .rename(columns={"permit_num": "well_api"})
    )
    prod_agg["has_unconv_prod_record"] = True

    geology = assign_geology(wells, canonical_dir / "bedrock_geology.geojson")

    table = wells.merge(permit_agg, on="well_api", how="left")
    table = table.merge(prod_agg, on="well_api", how="left")
    table = table.merge(geology, on="well_api", how="left")

    table["has_permit_record"] = table["has_permit_record"].eq(True)
    table["has_unconv_prod_record"] = table["has_unconv_prod_record"].eq(True)
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
    table["trainable_label_v0"] = table["is_developed_status"] | table["has_unconv_prod_record"]
    table["observed_gas_12mo"] = table["observed_gas_12mo"].fillna(0.0)
    table["observed_gas_avg_monthly"] = table["observed_gas_avg_monthly"].fillna(0.0)
    table["observed_gas_max_monthly"] = table["observed_gas_max_monthly"].fillna(0.0)
    table["observed_monthly_rows"] = table["observed_monthly_rows"].fillna(0).astype(int)

    label_specs = [
        ("label_obs12_gt_0", table["observed_gas_12mo"] > 0),
        ("label_obs12_ge_1000", table["observed_gas_12mo"] >= 1000),
        ("label_obs12_ge_10000", table["observed_gas_12mo"] >= 10000),
        ("label_obs12_ge_50000", table["observed_gas_12mo"] >= 50000),
        ("label_obs12_ge_100000", table["observed_gas_12mo"] >= 100000),
        ("label_obs12_ge_250000", table["observed_gas_12mo"] >= 250000),
        ("label_obs12_ge_500000", table["observed_gas_12mo"] >= 500000),
        ("label_obs12_ge_1000000", table["observed_gas_12mo"] >= 1000000),
    ]
    for col, values in label_specs:
        table[col] = values.where(table["trainable_label_v0"], pd.NA)

    if prod_window_start:
        window_start_ts = pd.Timestamp(prod_window_start)
        spud_ts = pd.to_datetime(table["spud_date"], errors="coerce")
        table["mature_by_obs_window_start"] = (spud_ts <= window_start_ts).where(spud_ts.notna(), pd.NA)
    else:
        table["mature_by_obs_window_start"] = pd.NA

    table["feature_group_static"] = True
    table["feature_group_operational"] = True
    table["feature_group_label"] = True

    keep_cols = [
        "well_api",
        "ogo_id",
        "operator_name",
        "well_status",
        "well_type",
        "well_configuration",
        "county_name",
        "municipality_name",
        "latitude_decimal",
        "longitude_decimal",
        "spud_date",
        "spud_year",
        "spud_month",
        "has_spud_date",
        "permit_count",
        "has_permit_record",
        "first_permit_issued_date",
        "latest_permit_issued_date",
        "first_permit_spud_date",
        "geology_map_symbol",
        "geology_name",
        "geology_age",
        "geology_lith1",
        "has_unconv_prod_record",
        "observed_monthly_rows",
        "observed_gas_12mo",
        "observed_gas_avg_monthly",
        "observed_gas_max_monthly",
        "observed_first_prod_date",
        "observed_last_prod_date",
        "is_developed_status",
        "is_excluded_status",
        "trainable_label_v0",
        "mature_by_obs_window_start",
        "label_obs12_gt_0",
        "label_obs12_ge_1000",
        "label_obs12_ge_10000",
        "label_obs12_ge_50000",
        "label_obs12_ge_100000",
        "label_obs12_ge_250000",
        "label_obs12_ge_500000",
        "label_obs12_ge_1000000",
    ]
    label_cols = [spec[0] for spec in label_specs]

    table = table[keep_cols].sort_values("well_api").reset_index(drop=True)

    out_csv = features_dir / "gas_training_table_v0.csv"
    write_csv(table, out_csv)

    trainable = table[table["trainable_label_v0"] == True].copy()
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_v0",
        "row_count_total": int(len(table)),
        "row_count_trainable_v0": int(len(trainable)),
        "row_count_with_recent_unconv_prod": int(table["has_unconv_prod_record"].sum()),
        "production_window_start": prod_window_start,
        "production_window_end": prod_window_end,
        "label_positive_counts": {
            col: int(trainable[col].astype("boolean").fillna(False).sum())
            for col in label_cols
        },
        "output_path": str(out_csv),
    }
    meta_path = features_dir / "gas_training_table_v0_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
