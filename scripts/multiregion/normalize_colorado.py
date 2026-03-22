#!/usr/bin/env python3
"""Normalize Colorado COGCC raw data into canonical tables."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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


def normalize_wells(raw_dir: Path, out_dir: Path) -> list[dict]:
    """Load COGCC well GeoJSON, flatten properties, rename to canonical columns."""
    wells_path = raw_dir / "co_cogcc" / "co_cogcc_wells.geojson"
    outputs: list[dict] = []

    wells_payload = load_json(wells_path)
    rows = []
    permit_features = []
    for feature in wells_payload.get("features", []):
        props = {snake_case(k): v for k, v in (feature.get("properties") or {}).items()}
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        props["longitude"] = coords[0]
        props["latitude"] = coords[1]
        rows.append(props)

        status = props.get("facil_stat", "")
        if status and "permit" in str(status).lower():
            permit_features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": props,
                }
            )

    wells_df = pd.DataFrame(rows).rename(
        columns={
            "api_label": "well_api",
            "county_name": "county_name",
            "facil_stat": "well_status",
            "operator_name": "operator_name",
            "dir_drill": "drill_type",
            "spud_date": "spud_date",
            "first_prod_date": "first_prod_date",
            "formation_name": "formation_name",
        }
    )
    for col in ["longitude", "latitude"]:
        if col in wells_df.columns:
            wells_df[col] = pd.to_numeric(wells_df[col], errors="coerce")
    for col in ["spud_date", "first_prod_date"]:
        if col in wells_df.columns:
            wells_df[col] = normalize_date_series(wells_df[col])

    wells_output = out_dir / "wells.csv"
    write_csv(wells_df, wells_output)
    outputs.append(
        {"dataset": "wells", "row_count": int(len(wells_df)), "output_path": str(wells_output)}
    )

    # Extract permits
    permits_df = wells_df[
        wells_df["well_status"].str.contains("permit", case=False, na=False)
    ].copy()
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


def normalize_production(raw_dir: Path, out_dir: Path) -> list[dict]:
    """Load COGCC production JSON chunks and normalize to monthly rows."""
    manifest = load_json(raw_dir / "co_cogcc" / "download_manifest.json")
    production_items = [
        item
        for item in manifest.get("downloads", [])
        if item.get("dataset") == "co_cogcc_production"
    ]
    if not production_items:
        return []

    rows = []
    for item in sorted(production_items, key=lambda entry: entry.get("chunk_offset", 0)):
        path = Path(item["output_path"])
        payload = load_json(path)
        for feature in payload.get("features", []):
            attrs = feature.get("attributes", {})
            props = {snake_case(k): v for k, v in attrs.items()}

            year = pd.to_numeric(pd.Series([props.get("prod_year")]), errors="coerce").iloc[0]
            month = pd.to_numeric(pd.Series([props.get("prod_month")]), errors="coerce").iloc[0]
            if pd.isna(year) or pd.isna(month):
                continue

            year = int(year)
            month = int(month)

            gas = pd.to_numeric(pd.Series([props.get("gas_produced")]), errors="coerce").iloc[0]
            oil = pd.to_numeric(pd.Series([props.get("oil_produced")]), errors="coerce").iloc[0]
            water = pd.to_numeric(
                pd.Series([props.get("water_produced")]), errors="coerce"
            ).iloc[0]

            rows.append(
                {
                    "well_api": props.get("api_label"),
                    "production_year": year,
                    "production_month": month,
                    "period_start_date": f"{year:04d}-{month:02d}-01",
                    "gas_quantity": gas,
                    "oil_quantity": oil,
                    "water_quantity": water,
                    "source_file": path.name,
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
    return [
        {
            "dataset": "production",
            "row_count": int(len(production_df)),
            "output_path": str(production_output),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Colorado COGCC data into canonical tables.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "canonical" / "co_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    outputs.extend(normalize_wells(raw_dir, out_dir))
    outputs.extend(normalize_production(raw_dir, out_dir))

    registry = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SubTerra CO MVP",
        "outputs": outputs,
    }
    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
