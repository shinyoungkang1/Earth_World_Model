#!/usr/bin/env python3
"""Normalize Ohio ODNR raw data into canonical tables."""

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
    """Normalize ODNR horizontal wells GeoJSON into canonical wells.csv and permits.csv."""
    wells_path = raw_dir / "oh_odnr" / "oh_odnr_horizontal_wells.geojson"
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

        status = str(props.get("well_status") or "").upper()
        if "PERMIT" in status:
            permit_features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": props,
                }
            )

    wells_df = pd.DataFrame(rows).rename(
        columns={
            "api_wellno": "well_api",
            "wl_cnty": "county_name",
            "wl_twp": "township_name",
            "slant": "well_type",
            "wl_status_desc": "well_status",
            "mapsymbol_desc": "map_symbol",
            "co_name": "operator_name",
            "proposedformation": "proposed_formation",
            "producingformation1": "formation_name",
            "producingformation2": "formation_name_2",
            "wh_lat": "wh_latitude",
            "wh_long": "wh_longitude",
            "well_nm": "well_name",
            "well_no": "well_number",
            "utica_shale": "utica_flag",
            "marcellus_shale": "marcellus_flag",
            "status_reas": "status_reason",
        }
    )

    for col in ["longitude", "latitude"]:
        if col in wells_df.columns:
            wells_df[col] = pd.to_numeric(wells_df[col], errors="coerce")

    for col in ["permit_issued_date", "spud_date", "completion_date"]:
        if col in wells_df.columns:
            wells_df[col] = normalize_date_series(wells_df[col])

    wells_output = out_dir / "wells.csv"
    write_csv(wells_df, wells_output)
    outputs.append(
        {"dataset": "wells", "row_count": int(len(wells_df)), "output_path": str(wells_output)}
    )

    # Extract permits from wells where status indicates permit
    permits_df = wells_df[
        wells_df["well_status"].astype(str).str.upper().str.contains("PERMIT", na=False)
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
    """Normalize Ohio production data into canonical production.csv.

    Handles two formats:
    1. Wide-format FracTracker data (Year1/Quarter1/Gas1... columns) — unpivots to long
    2. Standard long-format GeoJSON (if available in the future)
    """
    # Try the FracTracker wide-format production file first
    wide_path = raw_dir / "oh_odnr" / "oh_utica_marcellus_production_q2_2021.geojson"
    long_path = raw_dir / "oh_odnr" / "oh_odnr_production.geojson"

    if wide_path.exists():
        return _normalize_wide_production(wide_path, out_dir)
    if long_path.exists():
        return _normalize_long_production(long_path, out_dir)
    return []


def _normalize_wide_production(path: Path, out_dir: Path) -> list[dict]:
    """Unpivot FracTracker wide-format quarterly production to long monthly rows."""
    payload = load_json(path)
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        raw_api = props.get("API", "")
        well_name = props.get("WellName_N", "")
        surf_lat = props.get("SurfLat")
        surf_lon = props.get("SurfLon")
        # FracTracker API codes are county-level, not well-level.
        # Create a unique well identifier from county API + well name.
        api = f"{str(raw_api).strip()}_{well_name.strip().replace(' ', '_')}" if well_name else str(raw_api).strip()
        if not api:
            continue

        # The first columns (Year, Oil, Gas, Brine, Days) are quarter 0
        # Then Year1/Quarter1/Oil1/Gas1... through Year23/Quarter23/...
        # Extract all quarterly slots
        for suffix in [""] + [str(i) for i in range(1, 50)]:
            year_key = f"Year{suffix}"
            gas_key = f"Gas{suffix}"
            oil_key = f"Oil{suffix}"
            brine_key = f"Brine{suffix}"
            days_key = f"Days{suffix}"

            year_val = props.get(year_key)
            if year_val is None:
                break
            year_val = pd.to_numeric(pd.Series([year_val]), errors="coerce").iloc[0]
            if pd.isna(year_val) or year_val == 0:
                continue

            # Determine quarter (Quarter suffix gives Q number; first slot has no quarter)
            quarter_key = f"Quarter{suffix}" if suffix else None
            if quarter_key and quarter_key in props:
                quarter = pd.to_numeric(pd.Series([props[quarter_key]]), errors="coerce").iloc[0]
            else:
                quarter = 1  # First slot defaults to Q1

            if pd.isna(quarter):
                continue

            # Map quarter to month (use middle month: Q1=2, Q2=5, Q3=8, Q4=11)
            quarter = int(quarter)
            month = {1: 2, 2: 5, 3: 8, 4: 11}.get(quarter, 2)
            year = int(year_val)

            gas = pd.to_numeric(pd.Series([props.get(gas_key)]), errors="coerce").iloc[0]
            oil = pd.to_numeric(pd.Series([props.get(oil_key)]), errors="coerce").iloc[0]
            water = pd.to_numeric(pd.Series([props.get(brine_key)]), errors="coerce").iloc[0]
            days = pd.to_numeric(pd.Series([props.get(days_key)]), errors="coerce").iloc[0]

            # Skip quarters with no production data
            if pd.isna(gas) and pd.isna(oil):
                continue

            rows.append({
                "well_api": str(api).strip(),
                "production_year": year,
                "production_quarter": quarter,
                "production_month": month,
                "period_start_date": f"{year:04d}-{month:02d}-01",
                "gas_quantity": gas if pd.notna(gas) else 0.0,
                "oil_quantity": oil if pd.notna(oil) else 0.0,
                "water_quantity": water if pd.notna(water) else 0.0,
                "production_days": days if pd.notna(days) else 0,
                "production_scope": "horizontal_utica_marcellus",
            })

    production_df = pd.DataFrame(rows)
    for col in ["gas_quantity", "oil_quantity", "water_quantity", "production_days"]:
        if col in production_df.columns:
            production_df[col] = pd.to_numeric(production_df[col], errors="coerce")

    production_output = out_dir / "production.csv"
    write_csv(production_df, production_output)
    return [
        {
            "dataset": "production",
            "row_count": int(len(production_df)),
            "output_path": str(production_output),
            "source": "fractracker_utica_marcellus_q2_2021",
            "format": "quarterly_unpivoted",
        }
    ]


def _normalize_long_production(path: Path, out_dir: Path) -> list[dict]:
    """Normalize standard long-format GeoJSON production data."""
    payload = load_json(path)
    rows = []
    for feature in payload.get("features", []):
        props = {snake_case(k): v for k, v in (feature.get("properties") or {}).items()}
        rows.append(props)

    if not rows:
        return []

    production_df = pd.DataFrame(rows).rename(
        columns={
            "api_number": "well_api",
            "prod_year": "production_year",
            "prod_month": "production_month",
            "gas_prod": "gas_quantity",
            "oil_prod": "oil_quantity",
            "brine_prod": "water_quantity",
        }
    )

    for col in ["production_year", "production_month", "gas_quantity", "oil_quantity", "water_quantity"]:
        if col in production_df.columns:
            production_df[col] = pd.to_numeric(production_df[col], errors="coerce")

    year_col = production_df.get("production_year")
    month_col = production_df.get("production_month")
    if year_col is not None and month_col is not None:
        valid_mask = year_col.notna() & month_col.notna()
        production_df["period_start_date"] = pd.NA
        production_df.loc[valid_mask, "period_start_date"] = (
            year_col[valid_mask].astype(int).astype(str)
            + "-"
            + month_col[valid_mask].astype(int).apply(lambda m: f"{m:02d}")
            + "-01"
        )

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
        description="Normalize Ohio ODNR raw data into canonical tables.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "canonical" / "oh_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    outputs.extend(normalize_wells(raw_dir, out_dir))
    outputs.extend(normalize_production(raw_dir, out_dir))

    registry = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SubTerra OH MVP",
        "outputs": outputs,
    }
    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
