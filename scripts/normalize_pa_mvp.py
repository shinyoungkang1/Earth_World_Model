#!/usr/bin/env python3
"""Normalize the current Pennsylvania gas MVP raw data into canonical tables."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


def snake_case(name: str) -> str:
    name = name.strip().replace("/", " ")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def normalize_bool(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return pd.NA


def normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


def normalize_epoch_ms_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    parsed = parsed.dt.tz_convert(None)
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe_dict(mapping: dict) -> dict:
    safe = {}
    for key, value in mapping.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=True, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "unable to decode CSV bytes")


def geometry_bbox(geometry: dict | None) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None
    coords = geometry.get("coordinates")
    if coords is None:
        return None
    xs: list[float] = []
    ys: list[float] = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                xs.append(float(node[0]))
                ys.append(float(node[1]))
            else:
                for child in node:
                    walk(child)

    walk(coords)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def normalize_wells(raw_dir: Path, out_dir: Path) -> dict:
    path = raw_dir / "pa_dep" / "pa_dep_unconventional_well_inventory.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.columns = [snake_case(c) for c in df.columns]
    df = df.rename(
        columns={
            "ogo": "ogo_id",
            "operator": "operator_name",
            "client_status": "client_status",
            "api": "well_api",
            "farm": "farm_name",
            "well_status": "well_status",
            "well_type": "well_type",
            "configuration": "well_configuration",
            "region": "region_name",
            "county": "county_name",
            "municipality": "municipality_name",
            "zip": "operator_zip",
            "site_id": "site_id",
            "primary_facility_id": "primary_facility_id",
        }
    )
    for col in ["latitude_decimal", "longitude_decimal", "site_id", "primary_facility_id", "client_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["spud_date"]:
        if col in df.columns:
            df[col] = normalize_date_series(df[col])
    for col in ["unconventional", "conservation"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_bool)
    output_path = out_dir / "wells.csv"
    write_csv(df, output_path)
    return {"dataset": "wells", "row_count": int(len(df)), "output_path": str(output_path)}


def normalize_production(raw_dir: Path, out_dir: Path) -> dict:
    manifest = load_json(raw_dir / "pa_dep" / "download_manifest.json")
    production_items = [item for item in manifest["downloads"] if item["dataset"] == "pa_dep_production"]
    frames = []
    for item in production_items:
        csv_path = Path(item["output_path"])
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
        df.columns = [snake_case(c) for c in df.columns]
        df["source_period_id"] = item.get("period_id")
        df["source_period_label"] = item.get("period_label")
        label = item.get("period_label", "")
        df["production_scope"] = "unconventional" if "Unconventional" in label else "conventional"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in [
        "gas_quantity",
        "gas_operating_days",
        "oil_quantity",
        "oil_operating_days",
        "condensate_quantity",
        "condensate_operating_days",
        "latitude_decimal",
        "longitude_decimal",
        "client_id",
        "prod_grp_no",
        "og_production_periods_id",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["production_period_start_date", "production_period_end_date", "spud_date", "final_date"]:
        if col in df.columns:
            df[col] = normalize_date_series(df[col])
    if "unconventional_ind" in df.columns:
        df["unconventional_ind"] = df["unconventional_ind"].map(normalize_bool)
    output_path = out_dir / "production.csv"
    write_csv(df, output_path)
    return {"dataset": "production", "row_count": int(len(df)), "output_path": str(output_path)}


def normalize_permits(raw_dir: Path, out_dir: Path) -> dict | None:
    path = raw_dir / "pa_dep" / "pa_dep_unconventional_permits.geojson"
    if not path.exists():
        return None

    payload = load_json(path)
    rows = []
    features = []
    for feature in payload.get("features", []):
        props = {snake_case(k): v for k, v in feature.get("properties", {}).items()}
        props["permit_issued_date"] = normalize_epoch_ms_series(pd.Series([props.get("permit_issued_date")])).iloc[0]
        props["spud_date"] = normalize_date_series(pd.Series([props.get("spud_date")])).iloc[0]
        props["unconventional"] = normalize_bool(props.get("unconventional"))
        props["longitude"] = None
        props["latitude"] = None
        geometry = feature.get("geometry")
        if geometry and geometry.get("type") == "Point":
            coords = geometry.get("coordinates", [])
            if len(coords) >= 2:
                props["longitude"] = coords[0]
                props["latitude"] = coords[1]
        rows.append(props)
        features.append({"type": "Feature", "geometry": geometry, "properties": json_safe_dict(props)})

    df = pd.DataFrame(rows)
    for col in [
        "latitude_decimal",
        "longitude_decimal",
        "zip_code",
        "authorization_id",
        "client_id",
        "prmry_fac_id",
        "longitude",
        "latitude",
        "objectid",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    csv_path = out_dir / "permits.csv"
    geojson_path = out_dir / "permits.geojson"
    write_csv(df, csv_path)
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    return {
        "dataset": "permits",
        "row_count": int(len(df)),
        "output_csv": str(csv_path),
        "output_geojson": str(geojson_path),
    }


def normalize_bedrock_geology(raw_dir: Path, out_dir: Path) -> dict | None:
    path = raw_dir / "dcnr" / "pa_dcnr_bedrock_geology.geojson"
    payloads = []
    if path.exists():
        payloads.append(load_json(path))
    else:
        chunk_dir = raw_dir / "dcnr" / "pa_dcnr_bedrock_geology_chunks"
        chunk_paths = sorted(chunk_dir.glob("chunk_*.geojson"))
        if not chunk_paths:
            return None
        payloads.extend(load_json(chunk_path) for chunk_path in chunk_paths)

    rows = []
    features = []
    for payload in payloads:
        for feature in payload.get("features", []):
            props = {snake_case(k): v for k, v in feature.get("properties", {}).items()}
            rows.append(props)
            features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": json_safe_dict(props),
                }
            )

    df = pd.DataFrame(rows)
    units = (
        df[
            [c for c in ["map_symbol", "name", "age", "lith1", "lith2", "lith3", "l_desc"] if c in df.columns]
        ]
        .drop_duplicates()
        .sort_values(["map_symbol", "name"], na_position="last")
    )

    geojson_path = out_dir / "bedrock_geology.geojson"
    units_path = out_dir / "bedrock_geology_units.csv"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    write_csv(units, units_path)
    return {
        "dataset": "bedrock_geology",
        "feature_count": int(len(df)),
        "unit_count": int(len(units)),
        "output_geojson": str(geojson_path),
        "output_units_csv": str(units_path),
    }


def normalize_usgs_csv_references(raw_dir: Path, out_dir: Path) -> list[dict]:
    zip_path = raw_dir / "usgs" / "PAcsv.zip"
    outputs = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            with zf.open(member) as fh:
                df = read_csv_bytes(fh.read())
            df.columns = [snake_case(c) for c in df.columns]
            output_path = out_dir / f"{snake_case(Path(member).stem)}.csv"
            write_csv(df, output_path)
            outputs.append(
                {
                    "dataset": snake_case(Path(member).stem),
                    "row_count": int(len(df)),
                    "output_path": str(output_path),
                }
            )
    return outputs


def archive_inventory(raw_dir: Path, out_dir: Path) -> dict:
    archive_specs = [
        ("usgs", raw_dir / "usgs" / "PAfaults_lcc.zip"),
        ("usgs", raw_dir / "usgs" / "PAcsv.zip"),
        ("usgs", raw_dir / "usgs" / "qfault_gis.zip"),
        ("usgs", raw_dir / "usgs" / "qfault.kmz"),
        ("usgs", raw_dir / "usgs" / "ds9_geophysics.zip"),
    ]
    rows = []
    for source_system, path in archive_specs:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                rows.append(
                    {
                        "source_system": source_system,
                        "archive_path": str(path),
                        "archive_name": path.name,
                        "member_name": info.filename,
                        "member_size_bytes": info.file_size,
                    }
                )
    df = pd.DataFrame(rows)
    output_path = out_dir / "raw_archive_members.csv"
    write_csv(df, output_path)
    return {"dataset": "raw_archive_members", "row_count": int(len(df)), "output_path": str(output_path)}


def normalize_raster_catalogs(raw_dir: Path, out_dir: Path) -> list[dict]:
    source_specs = {
        "satellite_sentinel2_catalog": ("raster_scenes_sentinel2.csv", "sentinel-2-l2a"),
        "satellite_landsat_catalog": ("raster_scenes_landsat.csv", "landsat-c2-l1-oli-tirs"),
    }
    outputs = []
    catalog_rows = []
    for source_id, (filename, default_collection) in source_specs.items():
        manifest_path = raw_dir / "satellite_catalog" / source_id / "download_manifest.json"
        items_path = raw_dir / "satellite_catalog" / source_id / "catalog_items.json"
        if not manifest_path.exists() or not items_path.exists():
            continue

        manifest = load_json(manifest_path)
        payload = load_json(items_path)
        rows = []
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            assets = feature.get("assets", {})
            bbox = feature.get("bbox") or [None, None, None, None]
            rows.append(
                {
                    "source_id": source_id,
                    "collection": feature.get("collection", default_collection),
                    "scene_id": feature.get("id"),
                    "datetime": props.get("datetime"),
                    "start_datetime": props.get("start_datetime"),
                    "end_datetime": props.get("end_datetime"),
                    "cloud_cover": props.get("eo:cloud_cover"),
                    "platform": props.get("platform"),
                    "instruments": ";".join(props.get("instruments", []) or []),
                    "sat_orbit_state": props.get("sat:orbit_state"),
                    "bbox_west": bbox[0],
                    "bbox_south": bbox[1],
                    "bbox_east": bbox[2],
                    "bbox_north": bbox[3],
                    "thumbnail_href": (assets.get("thumbnail") or {}).get("href"),
                    "product_metadata_href": (assets.get("product_metadata") or {}).get("href"),
                    "catalog_checked_at_utc": manifest.get("checked_at_utc"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            for col in ["cloud_cover", "bbox_west", "bbox_south", "bbox_east", "bbox_north"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["datetime", "start_datetime", "end_datetime"]:
                df[col] = normalize_date_series(df[col])
        output_path = out_dir / filename
        write_csv(df, output_path)
        outputs.append(
            {
                "dataset": output_path.stem,
                "row_count": int(len(df)),
                "output_path": str(output_path),
            }
        )
        catalog_rows.append(
            {
                "source_id": source_id,
                "collection": manifest.get("collection"),
                "checked_at_utc": manifest.get("checked_at_utc"),
                "last_seen_datetime": manifest.get("last_seen_datetime"),
                "new_item_count": manifest.get("new_item_count"),
                "total_catalog_count": manifest.get("total_catalog_count"),
            }
        )

    if catalog_rows:
        catalog_df = pd.DataFrame(catalog_rows)
        catalog_path = out_dir / "raster_catalogs.csv"
        write_csv(catalog_df, catalog_path)
        outputs.append(
            {
                "dataset": "raster_catalogs",
                "row_count": int(len(catalog_df)),
                "output_path": str(catalog_path),
            }
        )
    return outputs


def normalize_raster_assets(raw_dir: Path, out_dir: Path) -> list[dict]:
    manifest_path = raw_dir / "satellite_assets" / "satellite_sentinel2_quicklooks" / "download_manifest.json"
    if not manifest_path.exists():
        return []

    manifest = load_json(manifest_path)
    asset_rows = []
    for item in manifest.get("downloads", []):
        asset_rows.append(
            {
                "source_id": manifest.get("source_id"),
                "collection": item.get("collection"),
                "scene_id": item.get("scene_id"),
                "grid_code": item.get("grid_code"),
                "datetime": item.get("datetime"),
                "cloud_cover": item.get("cloud_cover"),
                "platform": item.get("platform"),
                "asset_kind": item.get("asset_kind"),
                "asset_role": item.get("asset_role"),
                "status": item.get("status"),
                "href": item.get("href"),
                "local_path": item.get("local_path"),
                "file_size_bytes": item.get("file_size_bytes"),
                "image_width": item.get("image_width"),
                "image_height": item.get("image_height"),
                "bbox_west": item.get("bbox_west"),
                "bbox_south": item.get("bbox_south"),
                "bbox_east": item.get("bbox_east"),
                "bbox_north": item.get("bbox_north"),
                "checked_at_utc": manifest.get("checked_at_utc"),
            }
        )

    outputs = []
    if asset_rows:
        asset_df = pd.DataFrame(asset_rows)
        for col in [
            "cloud_cover",
            "file_size_bytes",
            "image_width",
            "image_height",
            "bbox_west",
            "bbox_south",
            "bbox_east",
            "bbox_north",
        ]:
            asset_df[col] = pd.to_numeric(asset_df[col], errors="coerce")
        asset_path = out_dir / "raster_assets.csv"
        write_csv(asset_df, asset_path)
        outputs.append(
            {
                "dataset": "raster_assets",
                "row_count": int(len(asset_df)),
                "output_path": str(asset_path),
            }
        )

    composite_rows = []
    for item in manifest.get("composites", []):
        composite_rows.append(
            {
                "composite_id": item.get("composite_id"),
                "source_id": item.get("source_id"),
                "collection": item.get("collection"),
                "composite_kind": item.get("composite_kind"),
                "rendering_method": item.get("rendering_method"),
                "scene_count": item.get("scene_count"),
                "scene_ids": ";".join(item.get("scene_ids", [])),
                "image_path": item.get("image_path"),
                "image_width": item.get("image_width"),
                "image_height": item.get("image_height"),
                "bbox_west": item.get("bbox_west"),
                "bbox_south": item.get("bbox_south"),
                "bbox_east": item.get("bbox_east"),
                "bbox_north": item.get("bbox_north"),
                "generated_at_utc": item.get("generated_at_utc"),
            }
        )
    if composite_rows:
        composite_df = pd.DataFrame(composite_rows)
        for col in [
            "scene_count",
            "image_width",
            "image_height",
            "bbox_west",
            "bbox_south",
            "bbox_east",
            "bbox_north",
        ]:
            composite_df[col] = pd.to_numeric(composite_df[col], errors="coerce")
        composite_path = out_dir / "raster_composites.csv"
        write_csv(composite_df, composite_path)
        outputs.append(
            {
                "dataset": "raster_composites",
                "row_count": int(len(composite_df)),
                "output_path": str(composite_path),
            }
        )
    return outputs


def normalize_planet_catalog(raw_dir: Path, out_dir: Path) -> list[dict]:
    manifest_path = raw_dir / "planet_catalog" / "planet_psscene_catalog" / "download_manifest.json"
    items_path = raw_dir / "planet_catalog" / "planet_psscene_catalog" / "catalog_items.json"
    if not manifest_path.exists() or not items_path.exists():
        return []

    manifest = load_json(manifest_path)
    payload = load_json(items_path)
    rows = []
    geojson_features = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        geometry = feature.get("geometry")
        bbox = geometry_bbox(geometry) or (None, None, None, None)
        row = {
            "source_id": "planet_psscene_catalog",
            "item_type": props.get("item_type"),
            "scene_id": feature.get("id"),
            "acquired": props.get("acquired"),
            "published": props.get("published"),
            "updated": props.get("updated"),
            "cloud_cover": props.get("cloud_cover"),
            "cloud_percent": props.get("cloud_percent"),
            "clear_percent": props.get("clear_percent"),
            "clear_confidence_percent": props.get("clear_confidence_percent"),
            "visible_percent": props.get("visible_percent"),
            "visible_confidence_percent": props.get("visible_confidence_percent"),
            "gsd": props.get("gsd"),
            "ground_control": props.get("ground_control"),
            "quality_category": props.get("quality_category"),
            "instrument": props.get("instrument"),
            "satellite_id": props.get("satellite_id"),
            "provider": props.get("provider"),
            "strip_id": props.get("strip_id"),
            "view_angle": props.get("view_angle"),
            "thumbnail_href": (feature.get("_links") or {}).get("thumbnail"),
            "assets_href": (feature.get("_links") or {}).get("assets"),
            "self_href": (feature.get("_links") or {}).get("_self"),
            "bbox_west": bbox[0],
            "bbox_south": bbox[1],
            "bbox_east": bbox[2],
            "bbox_north": bbox[3],
            "catalog_checked_at_utc": manifest.get("checked_at_utc"),
        }
        rows.append(row)
        geojson_features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": json_safe_dict(row),
            }
        )

    outputs = []
    if rows:
        df = pd.DataFrame(rows)
        for col in [
            "cloud_cover",
            "cloud_percent",
            "clear_percent",
            "clear_confidence_percent",
            "visible_percent",
            "visible_confidence_percent",
            "gsd",
            "view_angle",
            "bbox_west",
            "bbox_south",
            "bbox_east",
            "bbox_north",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["acquired", "published", "updated"]:
            df[col] = normalize_date_series(df[col])
        csv_path = out_dir / "planet_scenes_psscene.csv"
        geojson_path = out_dir / "planet_scenes_psscene.geojson"
        write_csv(df, csv_path)
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": geojson_features}),
            encoding="utf-8",
        )
        outputs.append(
            {
                "dataset": "planet_scenes_psscene",
                "row_count": int(len(df)),
                "output_csv": str(csv_path),
                "output_geojson": str(geojson_path),
            }
        )

    draft_rows = []
    for draft_path_str in manifest.get("order_draft_paths", []):
        draft_path = Path(draft_path_str)
        if not draft_path.exists():
            continue
        draft = load_json(draft_path)
        product = (draft.get("products") or [{}])[0]
        draft_rows.append(
            {
                "draft_name": draft.get("name"),
                "draft_path": str(draft_path),
                "source_type": draft.get("source_type"),
                "order_type": draft.get("order_type"),
                "item_type": product.get("item_type"),
                "product_bundle": product.get("product_bundle"),
                "item_count": len(product.get("item_ids", [])),
                "generated_at_utc": manifest.get("checked_at_utc"),
            }
        )
    if draft_rows:
        draft_df = pd.DataFrame(draft_rows)
        draft_path = out_dir / "planet_order_drafts.csv"
        write_csv(draft_df, draft_path)
        outputs.append(
            {
                "dataset": "planet_order_drafts",
                "row_count": int(len(draft_df)),
                "output_path": str(draft_path),
            }
        )
    return outputs


def build_asset_status(raw_dir: Path, out_dir: Path) -> dict:
    rows = []
    for manifest_path in [
        raw_dir / "pa_dep" / "download_manifest.json",
        raw_dir / "usgs" / "download_manifest_arc1.json",
        raw_dir / "usgs" / "download_manifest_geophysics.json",
        raw_dir / "arcgis_download_manifest.json",
        raw_dir / "dcnr" / "download_manifest_bedrock_geology_chunks.json",
        raw_dir / "satellite_catalog" / "satellite_sentinel2_catalog" / "download_manifest.json",
        raw_dir / "satellite_catalog" / "satellite_landsat_catalog" / "download_manifest.json",
        raw_dir / "satellite_assets" / "satellite_sentinel2_quicklooks" / "download_manifest.json",
        raw_dir / "planet_catalog" / "planet_psscene_catalog" / "download_manifest.json",
    ]:
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        if "catalog_items_path" in manifest:
            rows.append(
                {
                    "manifest_path": str(manifest_path),
                    "dataset": manifest.get("source_id"),
                    "status": "downloaded",
                    "raw_path": manifest["catalog_items_path"],
                }
            )
            continue
        for item in manifest.get("downloads", []):
            output_path = item.get("output_path") or item.get("tif_path") or item.get("local_path")
            if output_path:
                rows.append(
                    {
                        "manifest_path": str(manifest_path),
                        "dataset": item.get("dataset"),
                        "status": "downloaded",
                        "raw_path": output_path,
                    }
                )
            if item.get("xml_path"):
                rows.append(
                    {
                        "manifest_path": str(manifest_path),
                        "dataset": item.get("dataset"),
                        "status": "downloaded",
                        "raw_path": item["xml_path"],
                    }
                )
        for draft_path in manifest.get("order_draft_paths", []):
            rows.append(
                {
                    "manifest_path": str(manifest_path),
                    "dataset": "planet_order_draft",
                    "status": "generated",
                    "raw_path": draft_path,
                }
            )
        for item in manifest.get("composites", []):
            if item.get("image_path"):
                rows.append(
                    {
                        "manifest_path": str(manifest_path),
                        "dataset": item.get("composite_id"),
                        "status": "derived",
                        "raw_path": item["image_path"],
                    }
                )

    for bad_name in ["PAgeol_lcc.zip", "PAgeol_lcc.zip.pyfetch"]:
        path = raw_dir / "usgs" / bad_name
        if path.exists():
            rows.append(
                {
                    "manifest_path": pd.NA,
                    "dataset": "pa_geology_ofr_2005_1325",
                    "status": "invalid_zip",
                    "raw_path": str(path),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates()
    output_path = out_dir / "raw_asset_status.csv"
    write_csv(df, output_path)
    return {"dataset": "raw_asset_status", "row_count": int(len(df)), "output_path": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve data paths.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "canonical" / "pa_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    outputs.append(normalize_wells(raw_dir, out_dir))
    outputs.append(normalize_production(raw_dir, out_dir))
    permit_result = normalize_permits(raw_dir, out_dir)
    if permit_result:
        outputs.append(permit_result)
    geology_result = normalize_bedrock_geology(raw_dir, out_dir)
    if geology_result:
        outputs.append(geology_result)
    outputs.extend(normalize_usgs_csv_references(raw_dir, out_dir))
    outputs.extend(normalize_raster_catalogs(raw_dir, out_dir))
    outputs.extend(normalize_raster_assets(raw_dir, out_dir))
    outputs.extend(normalize_planet_catalog(raw_dir, out_dir))
    outputs.append(archive_inventory(raw_dir, out_dir))
    outputs.append(build_asset_status(raw_dir, out_dir))

    registry = {
        "normalized_at_utc": pd.Timestamp.utcnow().isoformat(),
        "outputs": outputs,
    }
    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
