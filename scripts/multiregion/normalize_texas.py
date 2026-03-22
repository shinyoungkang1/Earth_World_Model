#!/usr/bin/env python3
"""Normalize Texas RRC raw data into canonical tables.

One normalizer serves all three TX basins (Permian, Eagle Ford, Haynesville).
Reads GeoJSON files produced by download_texas.py and outputs canonical CSVs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Utility helpers (mirror normalize_wv_mvp.py patterns)
# ---------------------------------------------------------------------------

def snake_case(name: str) -> str:
    name = name.strip().replace("/", " ")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

WELL_COLUMN_MAP = {
    "api": "well_api",
    "gis_lat83": "latitude",
    "gis_long83": "longitude",
    "gis_symbol_description": "well_type",
    "gis_location_source": "location_source",
    "gis_well_number": "well_number",
    "gis_api5": "api5",
}

PRODUCTION_COLUMN_MAP = {
    "api_no": "well_api",
    "cycle_year": "production_year",
    "cycle_month": "production_month",
    "mon_gas_prod": "gas_quantity",
    "mon_oil_prod": "oil_quantity",
    "mon_wtr_prod": "water_quantity",
}


# ---------------------------------------------------------------------------
# Wells normalization
# ---------------------------------------------------------------------------

def normalize_wells(raw_dir: Path, out_dir: Path) -> list[dict]:
    """Load well GeoJSON, flatten, rename to canonical columns, write wells.csv and permits.csv."""
    wells_path = raw_dir / "tx_rrc" / "tx_rrc_horizontal_wells.geojson"
    outputs: list[dict] = []

    wells_payload = load_json(wells_path)
    rows: list[dict] = []
    permit_features: list[dict] = []

    for feature in wells_payload.get("features", []):
        props = {snake_case(k): v for k, v in (feature.get("properties") or {}).items()}
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        # Use geometry coordinates as primary lat/lon source
        if coords[0] is not None:
            props["gis_long83"] = coords[0]
        if coords[1] is not None:
            props["gis_lat83"] = coords[1]
        rows.append(props)

        # Collect permit-status features (TX GIS layer has no status field, skip permit extraction)
        status = props.get("gis_symbol_description", "")
        if status and "permit" in str(status).lower():
            permit_features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": props,
                }
            )

    wells_df = pd.DataFrame(rows).rename(columns=WELL_COLUMN_MAP)

    # Coerce numeric columns
    for col in ["longitude", "latitude"]:
        if col in wells_df.columns:
            wells_df[col] = pd.to_numeric(wells_df[col], errors="coerce")

    # Normalize date columns
    for col in ["spud_date", "completion_date"]:
        if col in wells_df.columns:
            wells_df[col] = normalize_date_series(wells_df[col])

    wells_output = out_dir / "wells.csv"
    write_csv(wells_df, wells_output)
    outputs.append(
        {"dataset": "wells", "row_count": int(len(wells_df)), "output_path": str(wells_output)}
    )

    # Permits subset (TX GIS layer may not have permit status; extract if present)
    if "well_type" in wells_df.columns:
        permits_df = wells_df[
            wells_df["well_type"].astype(str).str.lower().str.contains("permit", na=False)
        ].copy()
    else:
        permits_df = wells_df.head(0).copy()
    permits_output = out_dir / "permits.csv"
    permits_geojson = out_dir / "permits.geojson"
    write_csv(permits_df, permits_output)
    permits_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": permit_features}),
        encoding="utf-8",
    )
    outputs.append(
        {
            "dataset": "permits",
            "row_count": int(len(permits_df)),
            "output_csv": str(permits_output),
            "output_geojson": str(permits_geojson),
        }
    )

    return outputs


# ---------------------------------------------------------------------------
# Production normalization
# ---------------------------------------------------------------------------

def normalize_production(raw_dir: Path, out_dir: Path) -> list[dict]:
    """Load production GeoJSON files, normalize to monthly rows, write production.csv."""
    manifest_path = raw_dir / "tx_rrc" / "download_manifest.json"
    manifest = load_json(manifest_path)
    production_items = [
        item
        for item in manifest.get("downloads", [])
        if item.get("dataset") == "tx_rrc_production"
    ]
    if not production_items:
        return []

    rows: list[dict] = []
    asset_rows: list[dict] = []

    for item in sorted(production_items, key=lambda entry: entry.get("year", 0)):
        path = Path(item["output_path"])
        if not path.exists():
            continue
        payload = load_json(path)

        for feature in payload.get("features", []):
            props = {snake_case(k): v for k, v in (feature.get("properties") or {}).items()}

            year_raw = props.get("cycle_year")
            month_raw = props.get("cycle_month")
            year = pd.to_numeric(pd.Series([year_raw]), errors="coerce").iloc[0]
            month = pd.to_numeric(pd.Series([month_raw]), errors="coerce").iloc[0]
            if pd.isna(year) or pd.isna(month):
                continue
            year = int(year)
            month = int(month)

            gas = pd.to_numeric(pd.Series([props.get("mon_gas_prod")]), errors="coerce").iloc[0]
            oil = pd.to_numeric(pd.Series([props.get("mon_oil_prod")]), errors="coerce").iloc[0]
            water = pd.to_numeric(pd.Series([props.get("mon_wtr_prod")]), errors="coerce").iloc[0]

            api = props.get("api_no")
            rows.append(
                {
                    "well_api": str(api).strip() if pd.notna(api) else pd.NA,
                    "production_year": year,
                    "production_month": month,
                    "period_start_date": f"{year:04d}-{month:02d}-01",
                    "gas_quantity": gas,
                    "oil_quantity": oil,
                    "water_quantity": water,
                    "source_file": path.name,
                }
            )

        asset_rows.append(
            {
                "dataset": "tx_rrc_production",
                "year": item.get("year"),
                "source_url": item.get("source_url"),
                "source_file": path.name,
                "output_path": str(path),
                "feature_count": item.get("feature_count"),
            }
        )

    production_df = pd.DataFrame(rows)
    for col in [
        "production_year",
        "production_month",
        "gas_quantity",
        "oil_quantity",
        "water_quantity",
    ]:
        if col in production_df.columns:
            production_df[col] = pd.to_numeric(production_df[col], errors="coerce")

    production_output = out_dir / "production.csv"
    write_csv(production_df, production_output)

    asset_df = pd.DataFrame(asset_rows)
    asset_output = out_dir / "production_assets.csv"
    write_csv(asset_df, asset_output)

    return [
        {
            "dataset": "production",
            "row_count": int(len(production_df)),
            "output_path": str(production_output),
        },
        {
            "dataset": "production_assets",
            "row_count": int(len(asset_df)),
            "output_path": str(asset_output),
        },
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Texas RRC raw data into canonical tables.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "canonical" / "tx_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    outputs.extend(normalize_wells(raw_dir, out_dir))
    outputs.extend(normalize_production(raw_dir, out_dir))

    registry = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SubTerra TX MVP",
        "basins": ["Permian", "Eagle Ford", "Haynesville"],
        "outputs": outputs,
    }
    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
