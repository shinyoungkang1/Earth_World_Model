#!/usr/bin/env python3
"""Build the Ohio Utica gas training table from canonical OH data.

Reads canonical wells, permits, and production tables produced by
normalize_ohio.py, computes spatial neighbor and mature neighbor features,
and writes a training table compatible with the multiregion unified schema.

Output: data/features/oh_utica_statewide/gas_training_table_oh_v0.csv

Usage:
    python scripts/multiregion/build_oh_training_table_v0.py
    python scripts/multiregion/build_oh_training_table_v0.py --repo-root /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gas_v1_common import (
    build_balltree,
    count_neighbors_before_date,
    nearest_neighbor_before_date_km,
    parse_date,
    write_csv,
)
from multiregion.labels import (
    F12_THRESHOLDS,
    F24_THRESHOLDS,
    add_threshold_labels,
    compute_first_life_labels,
)
from wv_gas_common import neighbor_value_stats_before_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Ohio Utica gas training table.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="oh_utica_statewide")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    canonical_dir = repo_root / "data" / "canonical" / "oh_mvp"
    features_dir = repo_root / "data" / "features" / args.basin_id
    features_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load canonical tables ----
    wells = pd.read_csv(canonical_dir / "wells.csv", low_memory=False)
    permits = pd.read_csv(canonical_dir / "permits.csv", low_memory=False)

    production_path = canonical_dir / "production.csv"
    if production_path.exists():
        production = pd.read_csv(production_path, low_memory=False)
    else:
        print("WARNING: production.csv not found — labels will be empty.", file=sys.stderr)
        production = pd.DataFrame(columns=["well_api", "period_start_date", "gas_quantity"])

    # ---- Clean wells ----
    for col in ["latitude", "longitude"]:
        if col in wells.columns:
            wells[col] = pd.to_numeric(wells[col], errors="coerce")
    wells = wells.dropna(subset=["latitude", "longitude", "well_api"]).copy()
    wells = wells.drop_duplicates(subset=["well_api"]).copy()

    # Normalize dates
    for col in ["spud_date", "completion_date", "permit_issued_date"]:
        if col in wells.columns:
            wells[col] = parse_date(wells[col])

    # ---- Clean permits ----
    for col in ["latitude", "longitude"]:
        if col in permits.columns:
            permits[col] = pd.to_numeric(permits[col], errors="coerce")
    # Find a usable date column for permits
    permit_date_col = None
    for candidate in ["permit_issued_date", "spud_date", "completion_date"]:
        if candidate in permits.columns:
            permits[candidate] = parse_date(permits[candidate])
            if permits[candidate].notna().any():
                permit_date_col = candidate
                break
    permits = permits.dropna(subset=["latitude", "longitude"]).copy()

    # ---- Detect quarterly vs monthly production data ----
    # FracTracker OH data is quarterly (4 records/year vs 12 for monthly).
    # The month-based label method counts unique months, so quarterly data
    # can only reach ~4 covered months per year. We lower the threshold
    # accordingly: 4 quarters ≈ 12 months of coverage.
    months_per_well = production.groupby("well_api")["period_start_date"].nunique()
    if months_per_well.max() <= 40:
        # Quarterly data: use 4 covered periods as the f12 threshold
        f12_threshold = 4
        f24_threshold = 8
        print(f"Detected quarterly production data (max {months_per_well.max()} periods/well)")
    else:
        f12_threshold = 12
        f24_threshold = 24

    # ---- Compute first-life labels (month-based) ----
    labels = compute_first_life_labels(
        well_apis=wells["well_api"],
        production=production,
        method="month_based",
        well_api_col="well_api",
        period_col="period_start_date",
        gas_col="gas_quantity",
    )
    # Override availability flags for quarterly data
    if f12_threshold != 12:
        covered_months_f12 = labels["covered_days_f12"].fillna(0) / 30.44
        covered_months_f24 = labels["covered_days_f24"].fillna(0) / 30.44
        labels["label_f12_available"] = covered_months_f12.ge(f12_threshold)
        labels["label_f24_available"] = covered_months_f24.ge(f24_threshold)
    labels = add_threshold_labels(labels, f12_thresholds=F12_THRESHOLDS, f24_thresholds=F24_THRESHOLDS)
    table = wells.merge(labels, on="well_api", how="left")

    # ---- Context date (first_prod_date, fallback to spud_date or permit date) ----
    table["context_date"] = parse_date(table["first_prod_date"])
    for fallback_col in ["spud_date", "completion_date", "permit_issued_date"]:
        if fallback_col in table.columns:
            table["context_date"] = table["context_date"].combine_first(parse_date(table[fallback_col]))

    # ---- Spatial neighbor features ----
    well_context = table.dropna(subset=["context_date", "latitude", "longitude"]).copy()
    well_context["context_date"] = parse_date(well_context["context_date"])
    if well_context.empty:
        print("WARNING: No wells with context_date — spatial features will be zero/NaN.", file=sys.stderr)
        well_context, well_tree = well_context, None
    else:
        well_context, well_tree = build_balltree(well_context, "latitude", "longitude")

    if permit_date_col and permit_date_col in permits.columns:
        permit_context = permits.dropna(subset=[permit_date_col, "latitude", "longitude"]).copy()
    else:
        # No date column: use all permits as "current" context
        permit_context = permits.copy()
        permit_date_col = None
    if not permit_context.empty:
        permit_context, permit_tree = build_balltree(permit_context, "latitude", "longitude")
    else:
        permit_tree = None

    print(f"Wells with context_date: {table['context_date'].notna().sum()} / {len(table)}")

    context_rows = []
    for row in table[["well_api", "latitude", "longitude", "context_date"]].itertuples(index=False):
        target_date = pd.Timestamp(row.context_date) if pd.notna(row.context_date) else pd.NaT
        entry = {"well_api": row.well_api}

        if well_tree is not None:
            entry["well_count_2km"] = count_neighbors_before_date(
                row.latitude, row.longitude, target_date,
                well_context, well_tree, "context_date", "latitude", "longitude", 2.0,
            )
            entry["well_count_5km"] = count_neighbors_before_date(
                row.latitude, row.longitude, target_date,
                well_context, well_tree, "context_date", "latitude", "longitude", 5.0,
            )
            entry["nearest_well_km"] = nearest_neighbor_before_date_km(
                row.latitude, row.longitude, target_date,
                well_context, well_tree, "context_date", "latitude", "longitude",
            )
        else:
            entry.update({"well_count_2km": 0, "well_count_5km": 0, "nearest_well_km": float("nan")})

        if permit_tree is not None and permit_date_col is not None:
            entry["permit_count_2km"] = count_neighbors_before_date(
                row.latitude, row.longitude, target_date,
                permit_context, permit_tree, permit_date_col, "latitude", "longitude", 2.0,
            )
            entry["permit_count_5km"] = count_neighbors_before_date(
                row.latitude, row.longitude, target_date,
                permit_context, permit_tree, permit_date_col, "latitude", "longitude", 5.0,
            )
            entry["nearest_permit_km"] = nearest_neighbor_before_date_km(
                row.latitude, row.longitude, target_date,
                permit_context, permit_tree, permit_date_col, "latitude", "longitude",
            )
        else:
            entry.update({"permit_count_2km": 0, "permit_count_5km": 0, "nearest_permit_km": float("nan")})

        context_rows.append(entry)
    table = table.merge(pd.DataFrame(context_rows), on="well_api", how="left")

    # ---- Mature neighbor features ----
    mature_context = table[
        table["label_f12_available"].eq(True)
        & table["context_date"].notna()
        & table["latitude"].notna()
        & table["longitude"].notna()
    ].copy()
    if mature_context.empty:
        print("WARNING: No mature wells — mature neighbor features will be zero/NaN.", file=sys.stderr)
        mature_tree = None
    else:
        mature_context["context_date"] = parse_date(mature_context["context_date"])
        mature_context, mature_tree = build_balltree(mature_context, "latitude", "longitude")

    mature_rows = []
    for row in table[["well_api", "latitude", "longitude", "context_date"]].itertuples(index=False):
        target_date = pd.Timestamp(row.context_date) if pd.notna(row.context_date) else pd.NaT
        stats_5 = neighbor_value_stats_before_date(
            row.latitude, row.longitude, target_date,
            mature_context, mature_tree, "context_date", "latitude", "longitude",
            "f12_gas", 5.0,
        )
        stats_10 = neighbor_value_stats_before_date(
            row.latitude, row.longitude, target_date,
            mature_context, mature_tree, "context_date", "latitude", "longitude",
            "f12_gas", 10.0,
        )
        mature_rows.append(
            {
                "well_api": row.well_api,
                "mature_f12_count_5km": stats_5["count"],
                "mature_f12_median_gas_5km": stats_5["median"],
                "mature_f12_p90_gas_5km": stats_5["p90"],
                "mature_f12_count_10km": stats_10["count"],
                "mature_f12_median_gas_10km": stats_10["median"],
                "mature_f12_p90_gas_10km": stats_10["p90"],
            }
        )
    table = table.merge(pd.DataFrame(mature_rows), on="well_api", how="left")

    # ---- Select output columns ----
    # Use canonical names matching the multiregion unified schema.
    # DEM/geology features are not available for OH yet; merge.py fills them as NaN.
    keep_cols = [
        # Metadata
        "well_api",
        "county_name",
        "operator_name",
        "well_status",
        "formation_name",
        "latitude",
        "longitude",
        "spud_date",
        "first_prod_date",
        "context_date",
        # Spatial neighbors
        "well_count_2km",
        "well_count_5km",
        "nearest_well_km",
        "permit_count_2km",
        "permit_count_5km",
        "nearest_permit_km",
        # Mature neighbors
        "mature_f12_count_5km",
        "mature_f12_median_gas_5km",
        "mature_f12_p90_gas_5km",
        "mature_f12_count_10km",
        "mature_f12_median_gas_10km",
        "mature_f12_p90_gas_10km",
        # Labels
        "f12_gas",
        "f24_gas",
        "covered_days_f12",
        "covered_days_f24",
        "label_f12_available",
        "label_f24_available",
    ]
    for t in F12_THRESHOLDS:
        keep_cols.append(f"label_f12_ge_{t}")
    for t in F24_THRESHOLDS:
        keep_cols.append(f"label_f24_ge_{t}")

    table = table[[c for c in keep_cols if c in table.columns]].copy()
    table = table.sort_values(["county_name", "well_api"]).reset_index(drop=True)

    out_csv = features_dir / "gas_training_table_oh_v0.csv"
    write_csv(table, out_csv)

    # ---- Write metadata ----
    trainable = table[table["label_f12_available"].eq(True)].copy()
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_oh_v0",
        "basin_id": args.basin_id,
        "state": "OH",
        "row_count_total": int(len(table)),
        "row_count_f12_available": int(table["label_f12_available"].eq(True).sum()),
        "row_count_f24_available": int(table["label_f24_available"].eq(True).sum()),
        "label_definition": "True first-life F12/F24 labels using month-based production windows from ODNR data.",
        "label_positive_counts_f12": {
            f"label_f12_ge_{t}": int(trainable[f"label_f12_ge_{t}"].astype("boolean").fillna(False).sum())
            for t in F12_THRESHOLDS
            if f"label_f12_ge_{t}" in trainable.columns
        },
        "label_positive_counts_f24": {
            f"label_f24_ge_{t}": int(trainable[f"label_f24_ge_{t}"].astype("boolean").fillna(False).sum())
            for t in F24_THRESHOLDS
            if f"label_f24_ge_{t}" in trainable.columns
        },
        "feature_columns_numeric": [
            "latitude",
            "longitude",
            "well_count_2km",
            "well_count_5km",
            "nearest_well_km",
            "permit_count_2km",
            "permit_count_5km",
            "nearest_permit_km",
            "mature_f12_count_5km",
            "mature_f12_median_gas_5km",
            "mature_f12_p90_gas_5km",
            "mature_f12_count_10km",
            "mature_f12_median_gas_10km",
            "mature_f12_p90_gas_10km",
        ],
        "feature_columns_categorical": [
            "county_name",
            "formation_name",
        ],
        "note": "DEM/geology features not yet available for OH; will be NaN in unified table.",
        "output_path": str(out_csv),
    }
    meta_path = features_dir / "gas_training_table_oh_v0_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
