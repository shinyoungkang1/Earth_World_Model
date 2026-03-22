#!/usr/bin/env python3
"""Normalize West Virginia gas MVP raw data into canonical tables."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MONTHS = [
    ("jan", 1),
    ("feb", 2),
    ("mar", 3),
    ("apr", 4),
    ("may", 5),
    ("jun", 6),
    ("jul", 7),
    ("aug", 8),
    ("sep", 9),
    ("oct", 10),
    ("nov", 11),
    ("dec", 12),
]


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


def read_excel_or_csv_bytes(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True)
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), dtype=str)
    if name.lower().endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(data), dtype=str)
        except ImportError as exc:
            raise RuntimeError(f"Cannot read legacy XLS asset {name}: {exc}") from exc
    raise RuntimeError(f"Unsupported production asset type: {name}")


def read_tabular_asset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [
                member
                for member in zf.namelist()
                if member.lower().endswith((".xlsx", ".xls", ".csv"))
            ]
            if not members:
                raise RuntimeError(f"No tabular member found in {path}")
            with zf.open(members[0]) as fh:
                return read_excel_or_csv_bytes(fh.read(), members[0])
    if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
        return read_excel_or_csv_bytes(path.read_bytes(), path.name)
    raise RuntimeError(f"Unsupported production asset path: {path}")


def normalize_wells(raw_dir: Path, out_dir: Path) -> list[dict]:
    wells_path = raw_dir / "wv_dep" / "wv_dep_horizontal_wells.geojson"
    laterals_path = raw_dir / "wv_dep" / "wv_dep_horizontal_laterals.geojson"
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

        if props.get("wellstatus") in {"Permit Issued", "Permit Application"}:
            permit_features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": props,
                }
            )

    wells_df = pd.DataFrame(rows).rename(
        columns={
            "api": "well_api",
            "permitid": "permit_id",
            "respparty": "operator_name",
            "welltype": "well_type",
            "welluse": "well_use",
            "welldepth": "well_depth_class",
            "wellrig": "well_rig_class",
            "permittype": "permit_type",
            "issuedate": "permit_issued_date",
            "compdate": "completion_date",
            "wellstatus": "well_status",
            "farmname": "farm_name",
            "wellnumber": "well_number",
            "recdate": "record_date",
            "formation": "formation_name",
        }
    )
    for col in ["longitude", "latitude", "wellx", "welly", "objectid", "pkey"]:
        if col in wells_df.columns:
            wells_df[col] = pd.to_numeric(wells_df[col], errors="coerce")
    for col in ["permit_issued_date", "completion_date", "record_date"]:
        if col in wells_df.columns:
            wells_df[col] = normalize_date_series(wells_df[col])
    wells_output = out_dir / "wells.csv"
    write_csv(wells_df, wells_output)
    outputs.append({"dataset": "wells", "row_count": int(len(wells_df)), "output_path": str(wells_output)})

    permits_df = wells_df[wells_df["well_status"].isin(["Permit Issued", "Permit Application"])].copy()
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

    laterals_payload = load_json(laterals_path)
    lateral_rows = []
    for feature in laterals_payload.get("features", []):
        props = {snake_case(k): v for k, v in (feature.get("properties") or {}).items()}
        lateral_rows.append(props)
    laterals_df = pd.DataFrame(lateral_rows).rename(
        columns={
            "api": "well_api",
            "permitid": "permit_id",
            "respparty": "operator_name",
            "welltype": "well_type",
            "welluse": "well_use",
            "welldepth": "well_depth_class",
            "wellrig": "well_rig_class",
            "permittype": "permit_type",
            "issuedate": "permit_issued_date",
            "compdate": "completion_date",
            "wellstatus": "well_status",
            "farmname": "farm_name",
            "wellnumber": "well_number",
            "recdate": "record_date",
            "formation": "formation_name",
            "st_length_shape": "shape_length_m",
        }
    )
    for col in ["surface_x", "surface_y", "landing_x", "landing_y", "bottom_x", "bottom_y", "shape_length_m"]:
        if col in laterals_df.columns:
            laterals_df[col] = pd.to_numeric(laterals_df[col], errors="coerce")
    laterals_csv = out_dir / "laterals.csv"
    laterals_geojson = out_dir / "laterals.geojson"
    write_csv(laterals_df, laterals_csv)
    laterals_geojson.write_text(json.dumps(laterals_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "laterals",
            "row_count": int(len(laterals_df)),
            "output_csv": str(laterals_csv),
            "output_geojson": str(laterals_geojson),
        }
    )

    return outputs


def reported_through_month(item: dict) -> int:
    source_url = item.get("source_url", "")
    quarter_match = re.search(r"Q([1-4])", source_url, re.IGNORECASE)
    if quarter_match:
        return int(quarter_match.group(1)) * 3
    return 12


def normalize_production(raw_dir: Path, out_dir: Path) -> list[dict]:
    manifest = load_json(raw_dir / "wv_dep" / "download_manifest.json")
    production_items = [
        item for item in manifest.get("downloads", []) if item.get("dataset") == "wv_dep_h6a_production"
    ]
    if not production_items:
        return []

    rows = []
    asset_rows = []
    for item in sorted(production_items, key=lambda entry: entry.get("year", 0)):
        path = Path(item["output_path"])
        df = read_tabular_asset(path)
        df.columns = [snake_case(col) for col in df.columns]
        df["source_year"] = item.get("year")
        df["source_url"] = item.get("source_url")
        df["source_file"] = path.name
        df["reported_through_month"] = reported_through_month(item)
        df = df.rename(
            columns={
                "api": "well_api",
                "county": "county_name",
                "reporting_rp": "reporting_operator_name",
                "operator": "operator_name",
                "well_type": "well_type",
                "year": "production_year",
            }
        )

        for _, base_row in df.iterrows():
            year = pd.to_numeric(pd.Series([base_row.get("production_year")]), errors="coerce").iloc[0]
            if pd.isna(year):
                continue
            year = int(year)
            through_month = int(base_row.get("reported_through_month") or 12)
            for month_name, month_num in MONTHS:
                gas = pd.to_numeric(pd.Series([base_row.get(f"{month_name}_gas")]), errors="coerce").iloc[0]
                oil = pd.to_numeric(pd.Series([base_row.get(f"{month_name}_oil")]), errors="coerce").iloc[0]
                water = pd.to_numeric(pd.Series([base_row.get(f"{month_name}_water")]), errors="coerce").iloc[0]
                ngl = pd.to_numeric(pd.Series([base_row.get(f"{month_name}_ngl")]), errors="coerce").iloc[0]
                if month_num > through_month:
                    gas = oil = water = ngl = pd.NA

                rows.append(
                    {
                        "well_api": str(base_row.get("well_api")).strip() if pd.notna(base_row.get("well_api")) else pd.NA,
                        "county_name": base_row.get("county_name"),
                        "reporting_operator_name": base_row.get("reporting_operator_name"),
                        "operator_name": base_row.get("operator_name"),
                        "well_type": base_row.get("well_type"),
                        "production_scope": "horizontal_h6a",
                        "production_year": year,
                        "production_month": month_num,
                        "period_start_date": f"{year:04d}-{month_num:02d}-01",
                        "gas_quantity": gas,
                        "oil_quantity": oil,
                        "water_quantity": water,
                        "ngl_quantity": ngl,
                        "reported_through_month": through_month,
                        "source_file": base_row.get("source_file"),
                        "source_url": base_row.get("source_url"),
                    }
                )

        asset_rows.append(
            {
                "dataset": "wv_dep_h6a_production",
                "year": item.get("year"),
                "source_url": item.get("source_url"),
                "source_file": path.name,
                "output_path": str(path),
                "reported_through_month": reported_through_month(item),
            }
        )

    production_df = pd.DataFrame(rows)
    for col in [
        "production_year",
        "production_month",
        "gas_quantity",
        "oil_quantity",
        "water_quantity",
        "ngl_quantity",
        "reported_through_month",
    ]:
        if col in production_df.columns:
            production_df[col] = pd.to_numeric(production_df[col], errors="coerce")

    production_output = out_dir / "production.csv"
    write_csv(production_df, production_output)
    asset_df = pd.DataFrame(asset_rows)
    asset_output = out_dir / "production_assets.csv"
    write_csv(asset_df, asset_output)
    return [
        {"dataset": "production", "row_count": int(len(production_df)), "output_path": str(production_output)},
        {"dataset": "production_assets", "row_count": int(len(asset_df)), "output_path": str(asset_output)},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "canonical" / "wv_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.extend(normalize_wells(raw_dir, out_dir))
    outputs.extend(normalize_production(raw_dir, out_dir))

    registry = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SubTerra WV MVP",
        "outputs": outputs,
    }
    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
