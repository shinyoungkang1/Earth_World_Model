#!/usr/bin/env python3
"""Build the manifest and request tables for regional multiscale `#8` collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EARTH_RADIUS_M = 6_378_137.0


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pandas is required for regional multiscale manifest building.") from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a regional multiscale manifest plus grouped collection request tables.",
    )
    parser.add_argument("--input-path", required=True, help="CSV or parquet with region/sample center rows.")
    parser.add_argument("--region-id-column", default="region_id")
    parser.add_argument("--center-sample-id-column", default="sample_id")
    parser.add_argument("--center-latitude-column", default="center_latitude")
    parser.add_argument("--center-longitude-column", default="center_longitude")
    parser.add_argument("--fallback-latitude-column", default="latitude")
    parser.add_argument("--fallback-longitude-column", default="longitude")
    parser.add_argument("--tile-size-meters", type=float, default=2560.0)
    parser.add_argument("--local-grid-radius", type=int, default=1, help="1 means full 3x3.")
    parser.add_argument("--local-resolution-meters", type=float, default=10.0)
    parser.add_argument("--local-chip-size", type=int, default=256)
    parser.add_argument("--regional-side-meters", type=float, default=12800.0)
    parser.add_argument("--regional-resolution-meters", type=float, default=80.0)
    parser.add_argument("--regional-chip-size", type=int, default=160)
    parser.add_argument("--split-seed", default="regional_multiscale_v1")
    parser.add_argument("--split-bucket-column", default="continent_bucket")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--output-manifest-parquet", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--output-local-requests-csv", required=True)
    parser.add_argument("--output-regional-requests-csv", required=True)
    parser.add_argument("--output-center-targets-csv", required=True)
    return parser.parse_args()


def _read_table(path: Path):
    pd = require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported input format for {path}. Expected .csv or .parquet")


def _resolve_center_columns(frame, *, region_id_column: str, center_sample_id_column: str, center_latitude_column: str, center_longitude_column: str, fallback_latitude_column: str, fallback_longitude_column: str):
    pd = require_pandas()
    lat_column = center_latitude_column if center_latitude_column in frame.columns else fallback_latitude_column
    lon_column = center_longitude_column if center_longitude_column in frame.columns else fallback_longitude_column
    if lat_column not in frame.columns or lon_column not in frame.columns:
        raise SystemExit(
            "Could not resolve center coordinates; expected either "
            f"{center_latitude_column}/{center_longitude_column} or {fallback_latitude_column}/{fallback_longitude_column}"
        )
    out = frame.copy()
    if center_sample_id_column not in out.columns:
        out[center_sample_id_column] = [f"region_{idx:08d}" for idx in range(len(out))]
    if region_id_column not in out.columns:
        out[region_id_column] = out[center_sample_id_column]
    out = out.rename(
        columns={
            region_id_column: "region_id",
            center_sample_id_column: "center_sample_id",
            lat_column: "center_latitude",
            lon_column: "center_longitude",
        }
    )
    out["center_sample_id"] = out["center_sample_id"].astype(str)
    out["region_id"] = out["region_id"].astype(str)
    out["center_latitude"] = pd.to_numeric(out["center_latitude"], errors="coerce")
    out["center_longitude"] = pd.to_numeric(out["center_longitude"], errors="coerce")
    out = out.dropna(subset=["region_id", "center_sample_id", "center_latitude", "center_longitude"]).copy()
    out = out.drop_duplicates(subset=["region_id"], keep="first").reset_index(drop=True)
    return out


def _stable_fraction(key: str, seed: str) -> float:
    digest = hashlib.md5(f"{seed}::{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) / float(16 ** len(digest))


def assign_split(frame, *, seed: str, train_fraction: float, val_fraction: float, split_bucket_column: str) -> Any:
    if not (0.0 < train_fraction < 1.0):
        raise SystemExit("train-fraction must be between 0 and 1.")
    if not (0.0 <= val_fraction < 1.0):
        raise SystemExit("val-fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise SystemExit("train-fraction + val-fraction must be < 1.")

    out = frame.copy()
    bucket_values = out[split_bucket_column].astype(str) if split_bucket_column in out.columns else None
    splits: list[str] = []
    for row in out.itertuples(index=False):
        bucket = getattr(row, split_bucket_column) if split_bucket_column in out.columns else "global"
        region_id = str(getattr(row, "region_id"))
        frac = _stable_fraction(f"{bucket}::{region_id}", seed)
        if frac < train_fraction:
            splits.append("train")
        elif frac < train_fraction + val_fraction:
            splits.append("val")
        else:
            splits.append("test")
    out["split"] = splits
    return out


def offset_lat_lon(lat_deg: float, lon_deg: float, *, dx_m: float, dy_m: float) -> tuple[float, float]:
    distance_m = math.hypot(dx_m, dy_m)
    if distance_m == 0.0:
        return float(lat_deg), float(lon_deg)
    bearing = math.atan2(dx_m, dy_m)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    angular = distance_m / EARTH_RADIUS_M
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular) +
        math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)


def role_for_offset(dx_idx: int, dy_idx: int) -> str:
    mapping = {
        (0, 0): "center",
        (0, 1): "north",
        (0, -1): "south",
        (1, 0): "east",
        (-1, 0): "west",
        (1, 1): "north_east",
        (-1, 1): "north_west",
        (1, -1): "south_east",
        (-1, -1): "south_west",
    }
    if (dx_idx, dy_idx) not in mapping:
        return f"offset_{dx_idx}_{dy_idx}"
    return mapping[(dx_idx, dy_idx)]


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pd = require_pandas()
    frame = _read_table(Path(args.input_path))
    centers = _resolve_center_columns(
        frame,
        region_id_column=args.region_id_column,
        center_sample_id_column=args.center_sample_id_column,
        center_latitude_column=args.center_latitude_column,
        center_longitude_column=args.center_longitude_column,
        fallback_latitude_column=args.fallback_latitude_column,
        fallback_longitude_column=args.fallback_longitude_column,
    )
    centers = assign_split(
        centers,
        seed=str(args.split_seed),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        split_bucket_column=str(args.split_bucket_column),
    )

    passthrough_columns = [
        column
        for column in centers.columns
        if column not in {"region_id", "center_sample_id", "center_latitude", "center_longitude", "split"}
    ]

    manifest_rows: list[dict[str, Any]] = []
    local_request_rows: list[dict[str, Any]] = []
    regional_request_rows: list[dict[str, Any]] = []
    center_target_rows: list[dict[str, Any]] = []

    radius = int(args.local_grid_radius)
    tile_size_m = float(args.tile_size_meters)
    local_side_m = tile_size_m * float(2 * radius + 1)

    for row in centers.itertuples(index=False):
        region_id = str(getattr(row, "region_id"))
        center_sample_id = str(getattr(row, "center_sample_id"))
        center_latitude = float(getattr(row, "center_latitude"))
        center_longitude = float(getattr(row, "center_longitude"))
        split = str(getattr(row, "split"))
        passthrough = {column: getattr(row, column) for column in passthrough_columns}

        local_request_id = f"{center_sample_id}__local3x3"
        regional_request_id = f"{center_sample_id}__regional_context"

        for dy_idx in range(radius, -radius - 1, -1):
            for dx_idx in range(-radius, radius + 1):
                dx_m = float(dx_idx) * tile_size_m
                dy_m = float(dy_idx) * tile_size_m
                tile_latitude, tile_longitude = offset_lat_lon(
                    center_latitude,
                    center_longitude,
                    dx_m=dx_m,
                    dy_m=dy_m,
                )
                tile_role = role_for_offset(dx_idx, dy_idx)
                tile_id = f"{center_sample_id}__{tile_role}"
                manifest_rows.append(
                    {
                        "region_id": region_id,
                        "center_sample_id": center_sample_id,
                        "tile_id": tile_id,
                        "tile_role": tile_role,
                        "collection_role": "local_3x3_highres",
                        "collection_request_id": local_request_id,
                        "latitude": tile_latitude,
                        "longitude": tile_longitude,
                        "grid_dx_m": dx_m,
                        "grid_dy_m": dy_m,
                        "tile_side_m": tile_size_m,
                        "resolution_m": float(args.local_resolution_meters),
                        "split": split,
                        "is_center_tile": tile_role == "center",
                        "is_target_tile": tile_role == "center",
                        "is_static_context_tile": True,
                        **passthrough,
                    }
                )
                local_request_rows.append(
                    {
                        "request_id": local_request_id,
                        "sample_id": tile_id,
                        "center_sample_id": center_sample_id,
                        "region_id": region_id,
                        "tile_role": tile_role,
                        "latitude": tile_latitude,
                        "longitude": tile_longitude,
                        "grid_dx_m": dx_m,
                        "grid_dy_m": dy_m,
                        "collection_role": "local_3x3_highres",
                        "region_side_meters": local_side_m,
                        "resolution_meters": float(args.local_resolution_meters),
                        "chip_size": int(args.local_chip_size),
                        "split": split,
                        **passthrough,
                    }
                )

        manifest_rows.append(
            {
                "region_id": region_id,
                "center_sample_id": center_sample_id,
                "tile_id": f"{center_sample_id}__regional_context",
                "tile_role": "regional_context",
                "collection_role": "regional_context_lowres",
                "collection_request_id": regional_request_id,
                "latitude": center_latitude,
                "longitude": center_longitude,
                "grid_dx_m": 0.0,
                "grid_dy_m": 0.0,
                "tile_side_m": float(args.regional_side_meters),
                "resolution_m": float(args.regional_resolution_meters),
                "split": split,
                "is_center_tile": False,
                "is_target_tile": False,
                "is_static_context_tile": True,
                **passthrough,
            }
        )
        regional_request_rows.append(
            {
                "request_id": regional_request_id,
                "sample_id": f"{center_sample_id}__regional_context",
                "center_sample_id": center_sample_id,
                "region_id": region_id,
                "tile_role": "regional_context",
                "latitude": center_latitude,
                "longitude": center_longitude,
                "grid_dx_m": 0.0,
                "grid_dy_m": 0.0,
                "collection_role": "regional_context_lowres",
                "region_side_meters": float(args.regional_side_meters),
                "resolution_meters": float(args.regional_resolution_meters),
                "chip_size": int(args.regional_chip_size),
                "split": split,
                **passthrough,
            }
        )
        center_target_rows.append(
            {
                "sample_id": center_sample_id,
                "region_id": region_id,
                "center_sample_id": center_sample_id,
                "latitude": center_latitude,
                "longitude": center_longitude,
                "split": split,
                **passthrough,
            }
        )

    metadata = {
        "dataset": "regional_multiscale_manifest_v1",
        "input_path": str(Path(args.input_path)),
        "region_count": int(len(centers)),
        "tile_size_meters": tile_size_m,
        "local_grid_radius": radius,
        "local_tile_count": int((2 * radius + 1) ** 2),
        "local_side_meters": local_side_m,
        "regional_side_meters": float(args.regional_side_meters),
        "local_resolution_meters": float(args.local_resolution_meters),
        "regional_resolution_meters": float(args.regional_resolution_meters),
        "split_seed": str(args.split_seed),
        "split_counts": {
            split_name: int((centers["split"] == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "ablation_ladder": [
            "center_only",
            "center_plus_cardinal_neighbors",
            "center_plus_full_3x3",
            "center_plus_3x3_plus_regional_context",
        ],
    }
    return manifest_rows, metadata, local_request_rows, regional_request_rows, center_target_rows


def main() -> None:
    args = parse_args()
    pd = require_pandas()
    manifest_rows, metadata, local_request_rows, regional_request_rows, center_target_rows = build_manifest(args)

    manifest_frame = pd.DataFrame(manifest_rows)
    local_requests_frame = pd.DataFrame(local_request_rows)
    regional_requests_frame = pd.DataFrame(regional_request_rows)
    center_targets_frame = pd.DataFrame(center_target_rows)

    output_manifest = Path(args.output_manifest_parquet)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_bytes(b"") if False else None
    manifest_frame.to_parquet(output_manifest, index=False)

    local_requests_path = Path(args.output_local_requests_csv)
    local_requests_path.parent.mkdir(parents=True, exist_ok=True)
    local_requests_frame.to_csv(local_requests_path, index=False)

    regional_requests_path = Path(args.output_regional_requests_csv)
    regional_requests_path.parent.mkdir(parents=True, exist_ok=True)
    regional_requests_frame.to_csv(regional_requests_path, index=False)

    center_targets_path = Path(args.output_center_targets_csv)
    center_targets_path.parent.mkdir(parents=True, exist_ok=True)
    center_targets_frame.to_csv(center_targets_path, index=False)

    output_metadata = Path(args.output_metadata_json)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "output_manifest_parquet": str(output_manifest),
            "output_local_requests_csv": str(local_requests_path),
            "output_regional_requests_csv": str(regional_requests_path),
            "output_center_targets_csv": str(center_targets_path),
        }
    )
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
