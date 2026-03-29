#!/usr/bin/env python3
"""Build the manifest and request tables for regional multiscale collection."""

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
    parser.add_argument(
        "--local-grid-radius",
        type=int,
        default=1,
        help="Radius of the collected local high-resolution grid. 1=3x3, 2=5x5.",
    )
    parser.add_argument(
        "--local-window-radius",
        type=int,
        default=1,
        help="Radius of derived valid local target windows. 1 means target-centered 3x3 windows.",
    )
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
    parser.add_argument(
        "--output-local-training-windows-parquet",
        default="",
        help="Optional parquet output for valid target-centered local windows. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--output-local-window-members-parquet",
        default="",
        help="Optional parquet output listing all local-grid member tiles for each derived local window.",
    )
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


def ring_index(dx_idx: int, dy_idx: int) -> int:
    return int(max(abs(int(dx_idx)), abs(int(dy_idx))))


def _default_training_window_path(output_manifest: str, filename: str) -> Path:
    return Path(output_manifest).resolve().parent / filename


def build_local_training_windows(
    *,
    region_id: str,
    center_sample_id: str,
    split: str,
    local_request_id: str,
    tile_specs: list[dict[str, Any]],
    passthrough: dict[str, Any],
    local_grid_radius: int,
    local_window_radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if local_window_radius < 0:
        raise SystemExit("local-window-radius must be >= 0.")
    valid_target_radius = int(local_grid_radius - local_window_radius)
    if valid_target_radius < 0:
        raise SystemExit(
            f"local-window-radius={local_window_radius} is too large for local-grid-radius={local_grid_radius}."
        )
    target_specs = [
        spec
        for spec in tile_specs
        if ring_index(int(spec["grid_dx_idx"]), int(spec["grid_dy_idx"])) <= valid_target_radius
    ]
    window_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    local_support_side = int(2 * local_window_radius + 1)
    local_tile_count = int((2 * local_grid_radius + 1) ** 2)
    for target_spec in sorted(target_specs, key=lambda item: (int(item["grid_dy_idx"]), int(item["grid_dx_idx"]))):
        target_dx = int(target_spec["grid_dx_idx"])
        target_dy = int(target_spec["grid_dy_idx"])
        target_tile_id = str(target_spec["tile_id"])
        target_tile_role = str(target_spec["tile_role"])
        window_id = f"{center_sample_id}__target_{target_tile_role}__ctx{local_support_side}x{local_support_side}"
        local_support_count = 0
        halo_count = 0
        for member_spec in tile_specs:
            member_dx = int(member_spec["grid_dx_idx"])
            member_dy = int(member_spec["grid_dy_idx"])
            relative_dx = int(member_dx - target_dx)
            relative_dy = int(member_dy - target_dy)
            in_local_support = abs(relative_dx) <= local_window_radius and abs(relative_dy) <= local_window_radius
            if in_local_support:
                local_support_count += 1
            else:
                halo_count += 1
            member_rows.append(
                {
                    "window_id": window_id,
                    "region_id": region_id,
                    "center_sample_id": center_sample_id,
                    "collection_request_id": local_request_id,
                    "target_tile_id": target_tile_id,
                    "target_tile_role": target_tile_role,
                    "target_grid_dx_idx": target_dx,
                    "target_grid_dy_idx": target_dy,
                    "target_ring_index": ring_index(target_dx, target_dy),
                    "member_tile_id": str(member_spec["tile_id"]),
                    "member_tile_role": str(member_spec["tile_role"]),
                    "member_grid_dx_idx": member_dx,
                    "member_grid_dy_idx": member_dy,
                    "member_ring_index": ring_index(member_dx, member_dy),
                    "member_relative_dx_idx": relative_dx,
                    "member_relative_dy_idx": relative_dy,
                    "member_relative_role": role_for_offset(relative_dx, relative_dy),
                    "member_is_target_center": bool(str(member_spec["tile_id"]) == target_tile_id),
                    "member_is_original_region_center": bool(str(member_spec["tile_role"]) == "center"),
                    "member_in_local_support": bool(in_local_support),
                    "member_in_halo_context": bool(not in_local_support),
                    "local_grid_radius": int(local_grid_radius),
                    "local_grid_side": int(2 * local_grid_radius + 1),
                    "local_window_radius": int(local_window_radius),
                    "local_window_side": int(local_support_side),
                    "split": split,
                    **passthrough,
                }
            )
        if local_support_count != local_support_side ** 2:
            raise SystemExit(
                f"Unexpected local support count={local_support_count} for {window_id}; "
                f"expected {local_support_side ** 2}."
            )
        window_rows.append(
            {
                "window_id": window_id,
                "region_id": region_id,
                "center_sample_id": center_sample_id,
                "collection_request_id": local_request_id,
                "target_tile_id": target_tile_id,
                "target_tile_role": target_tile_role,
                "target_grid_dx_idx": target_dx,
                "target_grid_dy_idx": target_dy,
                "target_ring_index": ring_index(target_dx, target_dy),
                "split": split,
                "local_grid_radius": int(local_grid_radius),
                "local_grid_side": int(2 * local_grid_radius + 1),
                "local_grid_tile_count": int(local_tile_count),
                "local_window_radius": int(local_window_radius),
                "local_window_side": int(local_support_side),
                "local_window_tile_count": int(local_support_count),
                "halo_tile_count": int(halo_count),
                "is_primary_center_window": bool(target_tile_role == "center"),
                **passthrough,
            }
        )
    return window_rows, member_rows


def build_manifest(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
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
    local_window_radius = int(args.local_window_radius)
    local_grid_side = int(2 * radius + 1)
    tile_size_m = float(args.tile_size_meters)
    local_side_m = tile_size_m * float(local_grid_side)
    valid_target_radius = int(radius - local_window_radius)
    if valid_target_radius < 0:
        raise SystemExit(
            f"local-window-radius={local_window_radius} is too large for local-grid-radius={radius}."
        )

    local_window_rows: list[dict[str, Any]] = []
    local_window_member_rows: list[dict[str, Any]] = []

    for row in centers.itertuples(index=False):
        region_id = str(getattr(row, "region_id"))
        center_sample_id = str(getattr(row, "center_sample_id"))
        center_latitude = float(getattr(row, "center_latitude"))
        center_longitude = float(getattr(row, "center_longitude"))
        split = str(getattr(row, "split"))
        passthrough = {column: getattr(row, column) for column in passthrough_columns}

        local_request_id = f"{center_sample_id}__local{local_grid_side}x{local_grid_side}"
        regional_request_id = f"{center_sample_id}__regional_context"
        region_tile_specs: list[dict[str, Any]] = []

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
                current_ring_index = ring_index(dx_idx, dy_idx)
                is_valid_local_target = bool(current_ring_index <= valid_target_radius)
                manifest_rows.append(
                    {
                        "region_id": region_id,
                        "center_sample_id": center_sample_id,
                        "tile_id": tile_id,
                        "tile_role": tile_role,
                        "collection_role": "local_grid_highres",
                        "collection_request_id": local_request_id,
                        "latitude": tile_latitude,
                        "longitude": tile_longitude,
                        "grid_dx_idx": int(dx_idx),
                        "grid_dy_idx": int(dy_idx),
                        "grid_ring_index": int(current_ring_index),
                        "grid_dx_m": dx_m,
                        "grid_dy_m": dy_m,
                        "tile_side_m": tile_size_m,
                        "resolution_m": float(args.local_resolution_meters),
                        "local_grid_radius": int(radius),
                        "local_grid_side": int(local_grid_side),
                        "local_window_radius": int(local_window_radius),
                        "local_window_side": int(2 * local_window_radius + 1),
                        "split": split,
                        "is_center_tile": tile_role == "center",
                        "is_target_tile": tile_role == "center",
                        "is_valid_local_target": is_valid_local_target,
                        "is_inner_local_tile": bool(current_ring_index <= max(0, radius - 1)),
                        "is_halo_tile": bool(current_ring_index > max(0, radius - 1)),
                        "is_static_context_tile": True,
                        **passthrough,
                    }
                )
                tile_spec = {
                    "request_id": local_request_id,
                    "sample_id": tile_id,
                    "center_sample_id": center_sample_id,
                    "region_id": region_id,
                    "tile_id": tile_id,
                    "tile_role": tile_role,
                    "latitude": tile_latitude,
                    "longitude": tile_longitude,
                    "grid_dx_idx": int(dx_idx),
                    "grid_dy_idx": int(dy_idx),
                    "grid_ring_index": int(current_ring_index),
                    "grid_dx_m": dx_m,
                    "grid_dy_m": dy_m,
                    "collection_role": "local_grid_highres",
                    "region_side_meters": local_side_m,
                    "resolution_meters": float(args.local_resolution_meters),
                    "chip_size": int(args.local_chip_size),
                    "local_grid_radius": int(radius),
                    "local_grid_side": int(local_grid_side),
                    "local_window_radius": int(local_window_radius),
                    "local_window_side": int(2 * local_window_radius + 1),
                    "is_valid_local_target": is_valid_local_target,
                    "split": split,
                    **passthrough,
                }
                local_request_rows.append(tile_spec)
                region_tile_specs.append(tile_spec)

        training_windows, training_window_members = build_local_training_windows(
            region_id=region_id,
            center_sample_id=center_sample_id,
            split=split,
            local_request_id=local_request_id,
            tile_specs=region_tile_specs,
            passthrough=passthrough,
            local_grid_radius=radius,
            local_window_radius=local_window_radius,
        )
        local_window_rows.extend(training_windows)
        local_window_member_rows.extend(training_window_members)

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
        "dataset": "regional_multiscale_manifest_v2",
        "input_path": str(Path(args.input_path)),
        "region_count": int(len(centers)),
        "tile_size_meters": tile_size_m,
        "local_grid_radius": radius,
        "local_grid_side": int(local_grid_side),
        "local_tile_count": int((2 * radius + 1) ** 2),
        "local_side_meters": local_side_m,
        "local_window_radius": int(local_window_radius),
        "local_window_side": int(2 * local_window_radius + 1),
        "valid_target_radius": int(valid_target_radius),
        "valid_target_tile_count_per_region": int((2 * valid_target_radius + 1) ** 2),
        "local_training_window_count": int(len(local_window_rows)),
        "local_training_window_member_count": int(len(local_window_member_rows)),
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
            "target_centered_inner_3x3_from_local_grid",
            "target_centered_inner_3x3_plus_regional_context",
        ],
    }
    return (
        manifest_rows,
        metadata,
        local_request_rows,
        regional_request_rows,
        center_target_rows,
        local_window_rows,
        local_window_member_rows,
    )


def main() -> None:
    args = parse_args()
    pd = require_pandas()
    (
        manifest_rows,
        metadata,
        local_request_rows,
        regional_request_rows,
        center_target_rows,
        local_window_rows,
        local_window_member_rows,
    ) = build_manifest(args)

    manifest_frame = pd.DataFrame(manifest_rows)
    local_requests_frame = pd.DataFrame(local_request_rows)
    regional_requests_frame = pd.DataFrame(regional_request_rows)
    center_targets_frame = pd.DataFrame(center_target_rows)
    local_training_windows_frame = pd.DataFrame(local_window_rows)
    local_window_members_frame = pd.DataFrame(local_window_member_rows)

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

    local_training_windows_path = (
        Path(args.output_local_training_windows_parquet)
        if args.output_local_training_windows_parquet
        else _default_training_window_path(args.output_manifest_parquet, "local_training_windows.parquet")
    )
    local_training_windows_path.parent.mkdir(parents=True, exist_ok=True)
    local_training_windows_frame.to_parquet(local_training_windows_path, index=False)

    local_window_members_path = (
        Path(args.output_local_window_members_parquet)
        if args.output_local_window_members_parquet
        else _default_training_window_path(args.output_manifest_parquet, "local_window_members.parquet")
    )
    local_window_members_path.parent.mkdir(parents=True, exist_ok=True)
    local_window_members_frame.to_parquet(local_window_members_path, index=False)

    output_metadata = Path(args.output_metadata_json)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "output_manifest_parquet": str(output_manifest),
            "output_local_requests_csv": str(local_requests_path),
            "output_regional_requests_csv": str(regional_requests_path),
            "output_center_targets_csv": str(center_targets_path),
            "output_local_training_windows_parquet": str(local_training_windows_path),
            "output_local_window_members_parquet": str(local_window_members_path),
        }
    )
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
