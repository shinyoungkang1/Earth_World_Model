#!/usr/bin/env python3
"""Build a Texas basin-scoped gas training table from canonical TX data.

Reads canonical wells, permits, and production tables produced by
normalize_texas.py, filters to a specific basin's counties, computes
spatial neighbor and mature neighbor features, and writes a training
table compatible with the multiregion unified schema.

One TX canonical directory (tx_mvp) serves all three basins (Permian,
Eagle Ford, Haynesville). Each basin gets its own output table filtered
by county.

Output: data/features/{basin_id}/gas_training_table_tx_v0.csv

Usage:
    python scripts/multiregion/build_tx_training_table_v0.py --basin-id tx_permian_delaware_midland
    python scripts/multiregion/build_tx_training_table_v0.py --basin-id tx_eagle_ford
    python scripts/multiregion/build_tx_training_table_v0.py --basin-id tx_haynesville
    python scripts/multiregion/build_tx_training_table_v0.py --all-basins
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
    load_basin_config,
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

TX_BASINS = [
    "tx_permian_delaware_midland",
    "tx_eagle_ford",
    "tx_haynesville",
]


def build_training_table(
    repo_root: Path,
    basin_id: str,
    wells_all: pd.DataFrame,
    permits_all: pd.DataFrame,
    production_all: pd.DataFrame,
) -> int:
    """Build and write the training table for a single TX basin."""
    basin = load_basin_config(repo_root, basin_id)
    counties = set(basin.get("counties", []))
    if not counties:
        print(f"ERROR: basin config {basin_id} has no counties list.", file=sys.stderr)
        return 1

    features_dir = repo_root / "data" / "features" / basin_id
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Basin: {basin.get('display_name', basin_id)}")
    print(f"Counties: {sorted(counties)}")
    print(f"{'='*60}")

    # ---- Filter wells to basin counties ----
    # TX well data may have county info in various columns; try common ones
    county_col = None
    for candidate in ["county_name", "county", "gis_county"]:
        if candidate in wells_all.columns:
            county_col = candidate
            break

    if county_col is not None:
        # Normalize county names for matching (strip, title case)
        wells_all["_county_match"] = wells_all[county_col].astype(str).str.strip().str.title()
        counties_normalized = {c.strip().title() for c in counties}
        wells = wells_all[wells_all["_county_match"].isin(counties_normalized)].copy()
        wells = wells.drop(columns=["_county_match"])
        wells_all = wells_all.drop(columns=["_county_match"])

        if county_col != "county_name":
            wells = wells.rename(columns={county_col: "county_name"})
    else:
        # No county column — use all wells (spatial bbox would be an alternative)
        print(f"WARNING: No county column found; using all {len(wells_all)} wells.", file=sys.stderr)
        wells = wells_all.copy()

    print(f"Wells in basin: {len(wells)}")
    if wells.empty:
        print(f"WARNING: No wells found for basin {basin_id}. Skipping.", file=sys.stderr)
        return 0

    # ---- Filter production to basin wells ----
    basin_apis = set(wells["well_api"].dropna())
    production = production_all[production_all["well_api"].isin(basin_apis)].copy()
    print(f"Production records in basin: {len(production)}")

    # ---- Filter permits to basin counties ----
    if county_col is not None and county_col in permits_all.columns:
        permits_all["_county_match"] = permits_all[county_col].astype(str).str.strip().str.title()
        permits = permits_all[permits_all["_county_match"].isin(counties_normalized)].copy()
        permits = permits.drop(columns=["_county_match"])
        permits_all = permits_all.drop(columns=["_county_match"])
    else:
        permits = permits_all[permits_all["well_api"].isin(basin_apis)].copy()

    # ---- Compute first-life labels (month-based) ----
    labels = compute_first_life_labels(
        well_apis=wells["well_api"],
        production=production,
        method="month_based",
        well_api_col="well_api",
        period_col="period_start_date",
        gas_col="gas_quantity",
    )
    labels = add_threshold_labels(labels, f12_thresholds=F12_THRESHOLDS, f24_thresholds=F24_THRESHOLDS)
    table = wells.merge(labels, on="well_api", how="left")

    # ---- Context date ----
    table["context_date"] = parse_date(table["first_prod_date"])
    if "spud_date" in table.columns:
        table["context_date"] = table["context_date"].combine_first(parse_date(table["spud_date"]))

    # ---- Spatial neighbor features ----
    well_context = table.dropna(subset=["context_date", "latitude", "longitude"]).copy()
    well_context["context_date"] = parse_date(well_context["context_date"])
    well_context, well_tree = build_balltree(well_context, "latitude", "longitude")

    permit_date_col = "permit_issued_date" if "permit_issued_date" in permits.columns else "spud_date"
    if permit_date_col in permits.columns:
        permit_context = permits.dropna(subset=[permit_date_col, "latitude", "longitude"]).copy()
        permit_context[permit_date_col] = parse_date(permit_context[permit_date_col])
        permit_context, permit_tree = build_balltree(permit_context, "latitude", "longitude")
    else:
        permit_context = pd.DataFrame(columns=["latitude", "longitude"])
        permit_context, permit_tree = build_balltree(permit_context, "latitude", "longitude")

    print(f"Computing spatial features for {len(table)} wells...")
    context_rows = []
    for row in table[["well_api", "latitude", "longitude", "context_date"]].itertuples(index=False):
        target_date = pd.Timestamp(row.context_date) if pd.notna(row.context_date) else pd.NaT
        context_rows.append(
            {
                "well_api": row.well_api,
                "well_count_2km": count_neighbors_before_date(
                    row.latitude, row.longitude, target_date,
                    well_context, well_tree, "context_date", "latitude", "longitude", 2.0,
                ),
                "well_count_5km": count_neighbors_before_date(
                    row.latitude, row.longitude, target_date,
                    well_context, well_tree, "context_date", "latitude", "longitude", 5.0,
                ),
                "nearest_well_km": nearest_neighbor_before_date_km(
                    row.latitude, row.longitude, target_date,
                    well_context, well_tree, "context_date", "latitude", "longitude",
                ),
                "permit_count_2km": count_neighbors_before_date(
                    row.latitude, row.longitude, target_date,
                    permit_context, permit_tree, permit_date_col, "latitude", "longitude", 2.0,
                ),
                "permit_count_5km": count_neighbors_before_date(
                    row.latitude, row.longitude, target_date,
                    permit_context, permit_tree, permit_date_col, "latitude", "longitude", 5.0,
                ),
                "nearest_permit_km": nearest_neighbor_before_date_km(
                    row.latitude, row.longitude, target_date,
                    permit_context, permit_tree, permit_date_col, "latitude", "longitude",
                ),
            }
        )
    table = table.merge(pd.DataFrame(context_rows), on="well_api", how="left")

    # ---- Mature neighbor features ----
    mature_context = table[
        table["label_f12_available"].eq(True)
        & table["context_date"].notna()
        & table["latitude"].notna()
        & table["longitude"].notna()
    ].copy()
    mature_context["context_date"] = parse_date(mature_context["context_date"])
    mature_context, mature_tree = build_balltree(mature_context, "latitude", "longitude")

    print(f"Computing mature neighbor features ({len(mature_context)} mature wells)...")
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
    keep_cols = [
        # Metadata
        "well_api",
        "county_name",
        "operator_name",
        "well_status",
        "well_type",
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

    out_csv = features_dir / "gas_training_table_tx_v0.csv"
    write_csv(table, out_csv)

    # ---- Write metadata ----
    trainable = table[table["label_f12_available"].eq(True)].copy()
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": f"gas_training_table_tx_v0",
        "basin_id": basin_id,
        "basin_display_name": basin.get("display_name", basin_id),
        "state": "TX",
        "counties": sorted(counties),
        "row_count_total": int(len(table)),
        "row_count_f12_available": int(table["label_f12_available"].eq(True).sum()),
        "row_count_f24_available": int(table["label_f24_available"].eq(True).sum()),
        "label_definition": "True first-life F12/F24 labels using month-based production windows from TX RRC data.",
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
            "well_type",
        ],
        "note": "DEM/geology features not yet available for TX; will be NaN in unified table.",
        "output_path": str(out_csv),
    }
    meta_path = features_dir / "gas_training_table_tx_v0_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Texas gas training table for a specific basin.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument(
        "--basin-id",
        default="tx_permian_delaware_midland",
        help="Basin config ID (e.g. tx_permian_delaware_midland, tx_eagle_ford, tx_haynesville).",
    )
    parser.add_argument(
        "--all-basins",
        action="store_true",
        help="Build training tables for all three TX basins.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    canonical_dir = repo_root / "data" / "canonical" / "tx_mvp"

    # ---- Load canonical tables once (shared across basins) ----
    wells_all = pd.read_csv(canonical_dir / "wells.csv", low_memory=False)
    permits_all = pd.read_csv(canonical_dir / "permits.csv", low_memory=False)

    production_path = canonical_dir / "production.csv"
    if production_path.exists():
        production_all = pd.read_csv(production_path, low_memory=False)
    else:
        print("WARNING: production.csv not found — labels will be empty.", file=sys.stderr)
        production_all = pd.DataFrame(columns=["well_api", "period_start_date", "gas_quantity"])

    # Clean shared data
    for col in ["latitude", "longitude"]:
        if col in wells_all.columns:
            wells_all[col] = pd.to_numeric(wells_all[col], errors="coerce")
    wells_all = wells_all.dropna(subset=["latitude", "longitude", "well_api"]).copy()
    wells_all = wells_all.drop_duplicates(subset=["well_api"]).copy()

    for col in ["spud_date", "completion_date"]:
        if col in wells_all.columns:
            wells_all[col] = parse_date(wells_all[col])

    for col in ["latitude", "longitude"]:
        if col in permits_all.columns:
            permits_all[col] = pd.to_numeric(permits_all[col], errors="coerce")
    permits_all = permits_all.dropna(subset=["latitude", "longitude"]).copy()

    basins = TX_BASINS if args.all_basins else [args.basin_id]
    rc = 0
    for basin_id in basins:
        result = build_training_table(
            repo_root, basin_id,
            wells_all.copy(), permits_all.copy(), production_all.copy(),
        )
        if result != 0:
            rc = result

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
