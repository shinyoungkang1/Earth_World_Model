#!/usr/bin/env python3
"""Convert grouped regional multiscale EE exports into local-grid and regional-context sequences."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

_NP = None
_PD = None
_RASTERIO = None

S2_BAND_COUNT = 10
SCL_BAND_COUNT = 1
S1_BAND_COUNT = 2
PRESENCE_BAND_COUNT = 2
EE_BANDS_PER_WEEK = S2_BAND_COUNT + SCL_BAND_COUNT + S1_BAND_COUNT + PRESENCE_BAND_COUNT
DEFAULT_S2_INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
WEEK_PATTERN = re.compile(r"w(?P<start>\d{2})(?:_(?P<end>\d{2}))?")
TILE_SUFFIX_PATTERN = re.compile(r"(?P<prefix>.+?)(?P<x>\d{10})-(?P<y>\d{10})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grouped local-grid plus regional-context Earth Engine exports into sequence artifacts.",
    )
    parser.add_argument("--input-dir", required=True, help="Root directory containing grouped request subdirectories.")
    parser.add_argument("--manifest-path", required=True, help="Regional multiscale manifest parquet or CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output-format", default="npz", choices=["npz", "zarr", "both"])
    parser.add_argument("--zarr-shard-size", type=int, default=32)
    parser.add_argument("--zarr-compressor-clevel", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def require_numpy():
    global _NP
    if _NP is None:
        import numpy as np  # type: ignore
        _NP = np
    return _NP


def require_pandas():
    global _PD
    if _PD is None:
        import pandas as pd  # type: ignore
        _PD = pd
    return _PD


def require_rasterio():
    global _RASTERIO
    if _RASTERIO is None:
        import rasterio  # type: ignore
        from rasterio.merge import merge as raster_merge  # type: ignore
        _RASTERIO = (rasterio, raster_merge)
    return _RASTERIO


def _read_table(path: Path) -> pd.DataFrame:
    pd = require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported manifest format for {path}. Expected .csv or .parquet")


def weekly_windows(year: int) -> list[tuple[dt.date, dt.date]]:
    start = dt.date(year, 1, 1)
    end = dt.date(year + 1, 1, 1)
    windows: list[tuple[dt.date, dt.date]] = []
    current = start
    while current < end:
        nxt = min(current + dt.timedelta(days=7), end)
        windows.append((current, nxt))
        current = nxt
    return windows


def parse_week_start_and_count(path: Path, band_count: int) -> tuple[int, int]:
    match = WEEK_PATTERN.search(path.stem)
    week_start = int(match.group("start")) if match else 0
    inferred_count = band_count // EE_BANDS_PER_WEEK
    if match and match.group("end") is not None:
        week_end = int(match.group("end"))
        explicit_count = max(0, week_end - week_start)
        if explicit_count > 0:
            return week_start, explicit_count
    return week_start, inferred_count


def center_crop_hw(array: np.ndarray, target_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    np = require_numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array [bands,height,width], got shape={array.shape}")
    _bands, height, width = array.shape
    original_height = int(height)
    original_width = int(width)
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if height < target_size or width < target_size:
        pad_y = max(0, target_size - height)
        pad_x = max(0, target_size - width)
        pad_top = int(pad_y // 2)
        pad_bottom = int(pad_y - pad_top)
        pad_left = int(pad_x // 2)
        pad_right = int(pad_x - pad_left)
        array = np.pad(
            array,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=np.nan,
        )
        _bands, height, width = array.shape
    crop_y0 = 0
    crop_x0 = 0
    if height == target_size and width == target_size:
        cropped = array
    else:
        crop_y0 = int(max(0, (height - target_size) // 2))
        crop_x0 = int(max(0, (width - target_size) // 2))
        cropped = array[:, crop_y0 : crop_y0 + target_size, crop_x0 : crop_x0 + target_size]
    info = {
        "original_height": original_height,
        "original_width": original_width,
        "post_pad_height": int(height),
        "post_pad_width": int(width),
        "final_height": int(cropped.shape[1]),
        "final_width": int(cropped.shape[2]),
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "crop_y0": crop_y0,
        "crop_x0": crop_x0,
        "was_padded": bool(pad_top or pad_bottom or pad_left or pad_right),
        "was_cropped": bool(crop_y0 or crop_x0 or height != target_size or width != target_size),
    }
    if info["was_padded"] and info["was_cropped"]:
        info["normalization"] = "padded_and_cropped"
    elif info["was_padded"]:
        info["normalization"] = "padded"
    elif info["was_cropped"]:
        info["normalization"] = "cropped"
    else:
        info["normalization"] = "none"
    return cropped, info


def tiled_export_group_key(path: Path) -> str:
    match = TILE_SUFFIX_PATTERN.match(path.stem)
    if match:
        return match.group("prefix")
    return path.stem


def read_tif_or_mosaic(paths: list[Path]) -> tuple[np.ndarray, Path]:
    rasterio, raster_merge = require_rasterio()
    ordered = sorted(paths)
    if len(ordered) == 1:
        with rasterio.open(ordered[0]) as ds:
            return ds.read().astype(require_numpy().float32), ordered[0]
    datasets = [rasterio.open(path) for path in ordered]
    try:
        merged, _ = raster_merge(datasets)
    finally:
        for ds in datasets:
            ds.close()
    return merged.astype(require_numpy().float32), ordered[0]


def frame_metadata_for_week(
    *,
    sample_id: str,
    tif_path: Path,
    week_index: int,
    year: int,
    has_s2: bool,
    has_s1: bool,
) -> dict[str, Any]:
    windows = weekly_windows(year)
    if 0 <= week_index < len(windows):
        start, end = windows[week_index]
        start_iso = start.isoformat()
        end_iso = end.isoformat()
    else:
        start_iso = None
        end_iso = None
    return {
        "source": "earth_engine_export",
        "sample_id": sample_id,
        "week_index": int(week_index),
        "source_tif": str(tif_path),
        "has_s2": bool(has_s2),
        "has_s1": bool(has_s1),
        "start": start_iso,
        "end": end_iso,
    }


def day_of_year_for_week(week_index: int, year: int, frame_meta: dict[str, Any]) -> int:
    pd = require_pandas()
    for key in ("start",):
        value = frame_meta.get(key)
        if value:
            return int(pd.Timestamp(value).dayofyear)
    windows = weekly_windows(year)
    if 0 <= week_index < len(windows):
        return int(pd.Timestamp(windows[week_index][0]).dayofyear)
    return -1


def discover_sample_tiffs(input_dir: Path, sample_limit: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for tif_path in sorted(input_dir.rglob("*.tif")):
        sample_id = tif_path.parent.name if tif_path.parent != input_dir else tif_path.stem
        groups.setdefault(str(sample_id), []).append(tif_path)
    if sample_limit > 0:
        return dict(list(sorted(groups.items()))[:sample_limit])
    return dict(sorted(groups.items()))


def sanitize_id(value: str) -> str:
    return str(value).replace("::", "__").replace("/", "_").replace(":", "_")


def _normalize_super_tile(array: np.ndarray, target_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    return center_crop_hw(array, int(target_size))


def _grid_lookup(local_manifest: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for request_id, request_frame in local_manifest.groupby("collection_request_id", sort=True):
        request_rows: list[dict[str, Any]] = []
        tile_side_m = float(request_frame["tile_side_m"].iloc[0])
        if tile_side_m <= 0:
            raise ValueError(f"Invalid tile_side_m for request {request_id}")
        for row in request_frame.itertuples(index=False):
            tile_chip_size = int(round(float(row.tile_side_m) / float(row.resolution_m)))
            dx_idx = int(round(float(row.grid_dx_m) / tile_side_m))
            dy_idx = int(round(float(row.grid_dy_m) / tile_side_m))
            request_rows.append(
                {
                    "region_id": str(row.region_id),
                    "center_sample_id": str(row.center_sample_id),
                    "tile_id": str(row.tile_id),
                    "tile_role": str(row.tile_role),
                    "split": str(row.split),
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                    "collection_role": str(row.collection_role),
                    "tile_chip_size": tile_chip_size,
                    "dx_idx": dx_idx,
                    "dy_idx": dy_idx,
                    "grid_ring_index": int(getattr(row, "grid_ring_index", max(abs(dx_idx), abs(dy_idx)))),
                    "local_grid_side": int(getattr(row, "local_grid_side", 0) or 0),
                    "is_valid_local_target": bool(getattr(row, "is_valid_local_target", dx_idx == 0 and dy_idx == 0)),
                    "is_inner_local_tile": bool(getattr(row, "is_inner_local_tile", max(abs(dx_idx), abs(dy_idx)) <= 1)),
                }
            )
        dy_values = sorted({entry["dy_idx"] for entry in request_rows}, reverse=True)
        dx_values = sorted({entry["dx_idx"] for entry in request_rows})
        dy_to_row = {dy: idx for idx, dy in enumerate(dy_values)}
        dx_to_col = {dx: idx for idx, dx in enumerate(dx_values)}
        for entry in request_rows:
            entry["grid_row"] = int(dy_to_row[entry["dy_idx"]])
            entry["grid_col"] = int(dx_to_col[entry["dx_idx"]])
            entry["grid_size"] = int(len(dx_values))
        lookup[str(request_id)] = sorted(request_rows, key=lambda item: (item["grid_row"], item["grid_col"]))
    return lookup


def _regional_lookup(regional_manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in regional_manifest.itertuples(index=False):
        lookup[str(row.collection_request_id)] = {
            "region_id": str(row.region_id),
            "center_sample_id": str(row.center_sample_id),
            "tile_id": str(row.tile_id),
            "tile_role": str(row.tile_role),
            "split": str(row.split),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "collection_role": str(row.collection_role),
            "chip_size": int(round(float(row.tile_side_m) / float(row.resolution_m))),
        }
    return lookup


def _decode_tile_week_records(
    *,
    sample_id: str,
    tif_path: Path,
    tile_array: np.ndarray,
    year: int,
    frame_metadata_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    np = require_numpy()
    band_count = int(tile_array.shape[0])
    if band_count % EE_BANDS_PER_WEEK != 0:
        raise ValueError(f"Unexpected EE band count={band_count} in {tif_path}")
    week_start, week_count = parse_week_start_and_count(tif_path, band_count)
    tile_records: list[dict[str, Any]] = []
    for offset in range(int(week_count)):
        base_idx = offset * EE_BANDS_PER_WEEK
        s2 = tile_array[base_idx : base_idx + S2_BAND_COUNT].astype(np.float32)
        scl = tile_array[base_idx + S2_BAND_COUNT].astype(np.float32)
        s1 = tile_array[
            base_idx + S2_BAND_COUNT + SCL_BAND_COUNT :
            base_idx + S2_BAND_COUNT + SCL_BAND_COUNT + S1_BAND_COUNT
        ].astype(np.float32)
        presence = tile_array[
            base_idx
            + S2_BAND_COUNT
            + SCL_BAND_COUNT
            + S1_BAND_COUNT :
            base_idx
            + S2_BAND_COUNT
            + SCL_BAND_COUNT
            + S1_BAND_COUNT
            + PRESENCE_BAND_COUNT
        ]
        s2_present = bool(np.nanmean(presence[0]) > 0.5)
        s1_present = bool(np.nanmean(presence[1]) > 0.5)

        scl_int = np.where(np.isfinite(scl), np.rint(scl), -9999).astype(np.int16)
        s2_valid_mask = s2_present & np.isfinite(scl) & ~np.isin(scl_int, list(DEFAULT_S2_INVALID_SCL))
        s2_valid_mask = s2_valid_mask & np.all(np.isfinite(s2), axis=0)
        s1_valid_mask = s1_present & np.all(np.isfinite(s1), axis=0)
        s2 = np.where(s2_valid_mask[None, :, :], s2, np.nan).astype(np.float32)
        s1 = np.where(s1_valid_mask[None, :, :], s1, np.nan).astype(np.float32)

        week_index = int(week_start + offset)
        frame_meta = frame_metadata_for_week(
            sample_id=sample_id,
            tif_path=tif_path,
            week_index=week_index,
            year=year,
            has_s2=bool(s2_valid_mask.any()),
            has_s1=bool(s1_valid_mask.any()),
        )
        if frame_metadata_overrides:
            frame_meta.update(frame_metadata_overrides)
        tile_records.append(
            {
                "week_index": week_index,
                "day_of_year": day_of_year_for_week(week_index, year, frame_meta),
                "s2": s2,
                "s1": s1,
                "s2_mask": s2_valid_mask.astype(bool),
                "s1_mask": s1_valid_mask.astype(bool),
                "s2_ok": bool(s2_valid_mask.any()),
                "s1_ok": bool(s1_valid_mask.any()),
                "frame_meta": frame_meta,
            }
        )
    return tile_records


def _finalize_record(
    *,
    sample_id: str,
    sequence_prefix: str,
    year: int,
    latitude: float,
    longitude: float,
    region_id: str,
    center_sample_id: str,
    tile_role: str,
    collection_role: str,
    split: str,
    week_records: list[dict[str, Any]],
    failed_chunks: list[dict[str, Any]],
    normalization_events: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    np = require_numpy()
    week_records.sort(key=lambda entry: int(entry["week_index"]))
    if len({int(entry["week_index"]) for entry in week_records}) != len(week_records):
        raise ValueError(f"Duplicate week indices detected for sample_id={sample_id}")
    if not week_records:
        raise ValueError(f"No readable weekly exports found for sample_id={sample_id}")
    frame_mask = np.asarray([bool(entry["s2_ok"] and entry["s1_ok"]) for entry in week_records], dtype=bool)
    s2_frame_mask = np.asarray([bool(entry["s2_ok"]) for entry in week_records], dtype=bool)
    s1_frame_mask = np.asarray([bool(entry["s1_ok"]) for entry in week_records], dtype=bool)
    record = {
        "sequence_id": f"{sequence_prefix}::{sample_id}::{year}",
        "sample_id": sample_id,
        "search_group_id": region_id,
        "anchor_year": int(year),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "frame_count": int(len(week_records)),
        "paired_frame_count": int(frame_mask.sum()),
        "s2": np.stack([entry["s2"] for entry in week_records], axis=0).astype(np.float32),
        "s1": np.stack([entry["s1"] for entry in week_records], axis=0).astype(np.float32),
        "frame_mask": frame_mask,
        "s2_frame_mask": s2_frame_mask,
        "s1_frame_mask": s1_frame_mask,
        "s2_valid_mask": np.stack([entry["s2_mask"] for entry in week_records], axis=0).astype(bool),
        "s1_valid_mask": np.stack([entry["s1_mask"] for entry in week_records], axis=0).astype(bool),
        "day_of_year": np.asarray([entry["day_of_year"] for entry in week_records], dtype=np.int16),
        "bin_index": np.asarray([entry["week_index"] for entry in week_records], dtype=np.int16),
        "frame_metadata": [entry["frame_meta"] for entry in week_records],
        "failed_chunks": failed_chunks,
        "normalization_events": normalization_events,
        "region_id": region_id,
        "center_sample_id": center_sample_id,
        "tile_role": tile_role,
        "collection_role": collection_role,
        "split": split,
    }
    if extra_metadata:
        record.update(extra_metadata)
    return record


def _local_records_for_group(
    *,
    request_id: str,
    tif_paths: list[Path],
    tile_specs: list[dict[str, Any]],
    year: int,
) -> list[dict[str, Any]]:
    tile_records: dict[str, list[dict[str, Any]]] = {spec["tile_id"]: [] for spec in tile_specs}
    normalization_events: list[dict[str, Any]] = []
    failed_chunks: list[dict[str, Any]] = []

    grid_size = int(tile_specs[0]["grid_size"])
    local_chip_size = int(tile_specs[0]["tile_chip_size"])
    for tif_path in tif_paths:
        grouped_paths = [tif_path]
        try:
            array, source_path = read_tif_or_mosaic(grouped_paths)
        except Exception as exc:
            failed_chunks.append(
                {
                    "sample_id": request_id,
                    "source_tif": str(tif_path),
                    "reason": str(exc),
                    "request_id": request_id,
                }
            )
            continue
        target_size = int(local_chip_size * grid_size)
        cropped, crop_info = _normalize_super_tile(array, target_size)
        if crop_info["normalization"] != "none":
            normalization_events.append(
                {
                    "sample_id": request_id,
                    "source_tif": str(source_path),
                    "target_size": int(target_size),
                    **crop_info,
                }
            )
        for spec in tile_specs:
            row_idx = int(spec["grid_row"])
            col_idx = int(spec["grid_col"])
            y0 = row_idx * local_chip_size
            y1 = y0 + local_chip_size
            x0 = col_idx * local_chip_size
            x1 = x0 + local_chip_size
            tile_array = cropped[:, y0:y1, x0:x1]
            tile_records[spec["tile_id"]].extend(
                _decode_tile_week_records(
                    sample_id=str(spec["tile_id"]),
                    tif_path=source_path,
                    tile_array=tile_array,
                    year=year,
                    frame_metadata_overrides={
                        "request_id": request_id,
                        "center_sample_id": spec["center_sample_id"],
                        "region_id": spec["region_id"],
                        "tile_role": spec["tile_role"],
                        "collection_role": spec["collection_role"],
                    },
                )
            )

    records: list[dict[str, Any]] = []
    for spec in tile_specs:
        records.append(
            _finalize_record(
                sample_id=str(spec["tile_id"]),
                sequence_prefix="ee_regional_local",
                year=year,
                latitude=float(spec["latitude"]),
                longitude=float(spec["longitude"]),
                region_id=str(spec["region_id"]),
                center_sample_id=str(spec["center_sample_id"]),
                tile_role=str(spec["tile_role"]),
                collection_role=str(spec["collection_role"]),
                split=str(spec["split"]),
                week_records=tile_records[str(spec["tile_id"])],
                failed_chunks=list(failed_chunks),
                normalization_events=list(normalization_events),
                extra_metadata={
                    "grid_dx_idx": int(spec["dx_idx"]),
                    "grid_dy_idx": int(spec["dy_idx"]),
                    "grid_ring_index": int(spec["grid_ring_index"]),
                    "local_grid_side": int(spec["local_grid_side"]),
                    "is_valid_local_target": bool(spec["is_valid_local_target"]),
                    "is_inner_local_tile": bool(spec["is_inner_local_tile"]),
                },
            )
        )
    return records


def _regional_record_for_group(
    *,
    request_id: str,
    tif_paths: list[Path],
    region_spec: dict[str, Any],
    year: int,
) -> dict[str, Any]:
    week_records: list[dict[str, Any]] = []
    normalization_events: list[dict[str, Any]] = []
    failed_chunks: list[dict[str, Any]] = []
    chip_size = int(region_spec["chip_size"])
    for tif_path in tif_paths:
        grouped_paths = [tif_path]
        try:
            array, source_path = read_tif_or_mosaic(grouped_paths)
        except Exception as exc:
            failed_chunks.append(
                {
                    "sample_id": request_id,
                    "source_tif": str(tif_path),
                    "reason": str(exc),
                    "request_id": request_id,
                }
            )
            continue
        cropped, crop_info = center_crop_hw(array, chip_size)
        if crop_info["normalization"] != "none":
            normalization_events.append(
                {
                    "sample_id": request_id,
                    "source_tif": str(source_path),
                    "target_size": int(chip_size),
                    **crop_info,
                }
            )
        week_records.extend(
            _decode_tile_week_records(
                sample_id=str(region_spec["tile_id"]),
                tif_path=source_path,
                tile_array=cropped,
                year=year,
                frame_metadata_overrides={
                    "request_id": request_id,
                    "center_sample_id": region_spec["center_sample_id"],
                    "region_id": region_spec["region_id"],
                    "tile_role": region_spec["tile_role"],
                    "collection_role": region_spec["collection_role"],
                },
            )
        )

    return _finalize_record(
        sample_id=str(region_spec["tile_id"]),
        sequence_prefix="ee_regional_context",
        year=year,
        latitude=float(region_spec["latitude"]),
        longitude=float(region_spec["longitude"]),
        region_id=str(region_spec["region_id"]),
        center_sample_id=str(region_spec["center_sample_id"]),
        tile_role=str(region_spec["tile_role"]),
        collection_role=str(region_spec["collection_role"]),
        split=str(region_spec["split"]),
        week_records=week_records,
        failed_chunks=failed_chunks,
        normalization_events=normalization_events,
    )


def _write_npz_records(output_dir: Path, records: list[dict[str, Any]], *, index_name: str) -> list[dict[str, Any]]:
    pd = require_pandas()
    np = require_numpy()
    sequences_dir = output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        sequence_path = sequences_dir / f"{sanitize_id(str(record['sequence_id']))}.npz"
        np.savez_compressed(
            sequence_path,
            s2=record["s2"],
            s1=record["s1"],
            frame_mask=record["frame_mask"],
            s2_frame_mask=record["s2_frame_mask"],
            s1_frame_mask=record["s1_frame_mask"],
            s2_valid_mask=record["s2_valid_mask"],
            s1_valid_mask=record["s1_valid_mask"],
            day_of_year=record["day_of_year"],
            bin_index=record["bin_index"],
        )
        rows.append(
            {
                "sequence_id": str(record["sequence_id"]),
                "sample_id": str(record["sample_id"]),
                "anchor_year": int(record["anchor_year"]),
                "search_group_id": str(record.get("search_group_id") or ""),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "sequence_path": str(sequence_path),
                "frame_count": int(record["frame_count"]),
                "paired_frame_count": int(record["paired_frame_count"]),
                "s2_frame_count": int(np.asarray(record["s2_frame_mask"]).sum()),
                "s1_frame_count": int(np.asarray(record["s1_frame_mask"]).sum()),
                "frame_mask_json": json.dumps(np.asarray(record["frame_mask"], dtype=int).tolist()),
                "s2_frame_mask_json": json.dumps(np.asarray(record["s2_frame_mask"], dtype=int).tolist()),
                "s1_frame_mask_json": json.dumps(np.asarray(record["s1_frame_mask"], dtype=int).tolist()),
                "day_of_year_json": json.dumps(np.asarray(record["day_of_year"], dtype=int).tolist()),
                "bin_indices_json": json.dumps(np.asarray(record["bin_index"], dtype=int).tolist()),
                "frame_metadata_json": json.dumps(record.get("frame_metadata", [])),
                "status": "written",
            }
        )
    metadata_lookup = {
        str(record["sequence_id"]): {
            "region_id": str(record["region_id"]),
            "center_sample_id": str(record["center_sample_id"]),
            "tile_role": str(record["tile_role"]),
            "collection_role": str(record["collection_role"]),
            "split": str(record["split"]),
            "grid_dx_idx": record.get("grid_dx_idx"),
            "grid_dy_idx": record.get("grid_dy_idx"),
            "grid_ring_index": record.get("grid_ring_index"),
            "local_grid_side": record.get("local_grid_side"),
            "is_valid_local_target": record.get("is_valid_local_target"),
            "is_inner_local_tile": record.get("is_inner_local_tile"),
        }
        for record in records
    }
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(metadata_lookup.get(str(row["sequence_id"]), {}))
        enriched_rows.append(enriched)
    pd.DataFrame(enriched_rows).to_parquet(output_dir / index_name, index=False)
    return enriched_rows


def _write_zarr_records(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    index_name: str,
    shard_prefix: str,
    zarr_shard_size: int,
    zarr_compressor_clevel: int,
) -> list[dict[str, Any]]:
    pd = require_pandas()
    zarr_dir = output_dir / "zarr_shards"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    from ewm.data.dense_temporal_writer import write_dense_temporal_zarr_shard  # noqa: WPS433
    rows: list[dict[str, Any]] = []
    for shard_idx in range(0, len(records), int(zarr_shard_size)):
        shard_records = records[shard_idx : shard_idx + int(zarr_shard_size)]
        shard_path = zarr_dir / f"{shard_prefix}_{shard_idx // int(zarr_shard_size):05d}.zarr"
        rows.extend(
            write_dense_temporal_zarr_shard(
                shard_path,
                shard_records,
                compressor_clevel=int(zarr_compressor_clevel),
            )
        )
    metadata_lookup = {
        str(record["sequence_id"]): {
            "region_id": str(record["region_id"]),
            "center_sample_id": str(record["center_sample_id"]),
            "tile_role": str(record["tile_role"]),
            "collection_role": str(record["collection_role"]),
            "split": str(record["split"]),
            "grid_dx_idx": record.get("grid_dx_idx"),
            "grid_dy_idx": record.get("grid_dy_idx"),
            "grid_ring_index": record.get("grid_ring_index"),
            "local_grid_side": record.get("local_grid_side"),
            "is_valid_local_target": record.get("is_valid_local_target"),
            "is_inner_local_tile": record.get("is_inner_local_tile"),
        }
        for record in records
    }
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(metadata_lookup.get(str(row["sequence_id"]), {}))
        enriched_rows.append(enriched)
    pd.DataFrame(enriched_rows).to_parquet(output_dir / index_name, index=False)
    return enriched_rows


def _write_failures(path: Path, records: list[dict[str, Any]]) -> None:
    pd = require_pandas()
    rows = [failure for record in records for failure in record.get("failed_chunks", [])]
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


def _write_normalization(path: Path, records: list[dict[str, Any]]) -> None:
    pd = require_pandas()
    rows = [event for record in records for event in record.get("normalization_events", [])]
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> int:
    args = parse_args()
    pd = require_pandas()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_table(Path(args.manifest_path))
    local_manifest = manifest[manifest["collection_role"].isin({"local_grid_highres", "local_3x3_highres"})].copy()
    regional_manifest = manifest[manifest["collection_role"] == "regional_context_lowres"].copy()
    local_lookup = _grid_lookup(local_manifest)
    regional_lookup = _regional_lookup(regional_manifest)

    grouped_tiffs = discover_sample_tiffs(input_dir, int(args.sample_limit))
    if not grouped_tiffs:
        raise SystemExit(f"No TIFF exports found under {input_dir}")

    local_records: list[dict[str, Any]] = []
    regional_records: list[dict[str, Any]] = []

    for request_id, tif_paths in grouped_tiffs.items():
        if request_id in local_lookup:
            local_records.extend(
                _local_records_for_group(
                    request_id=request_id,
                    tif_paths=tif_paths,
                    tile_specs=local_lookup[request_id],
                    year=int(args.year),
                )
            )
        elif request_id in regional_lookup:
            regional_records.append(
                _regional_record_for_group(
                    request_id=request_id,
                    tif_paths=tif_paths,
                    region_spec=regional_lookup[request_id],
                    year=int(args.year),
                )
            )

    local_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    local_zarr_rows: list[dict[str, Any]] = []
    regional_zarr_rows: list[dict[str, Any]] = []

    if args.output_format in {"npz", "both"}:
        if local_records:
            local_dir = output_dir / "local_grid"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_rows = _write_npz_records(local_dir, local_records, index_name="dense_temporal_index.parquet")
        if regional_records:
            regional_dir = output_dir / "regional_context"
            regional_dir.mkdir(parents=True, exist_ok=True)
            regional_rows = _write_npz_records(regional_dir, regional_records, index_name="dense_temporal_index.parquet")

    if args.output_format in {"zarr", "both"}:
        if local_records:
            local_dir = output_dir / "local_grid"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_zarr_rows = _write_zarr_records(
                local_dir,
                local_records,
                index_name="dense_temporal_zarr_index.parquet",
                shard_prefix="local_grid_shard",
                zarr_shard_size=int(args.zarr_shard_size),
                zarr_compressor_clevel=int(args.zarr_compressor_clevel),
            )
        if regional_records:
            regional_dir = output_dir / "regional_context"
            regional_dir.mkdir(parents=True, exist_ok=True)
            regional_zarr_rows = _write_zarr_records(
                regional_dir,
                regional_records,
                index_name="dense_temporal_zarr_index.parquet",
                shard_prefix="regional_context_shard",
                zarr_shard_size=int(args.zarr_shard_size),
                zarr_compressor_clevel=int(args.zarr_compressor_clevel),
            )

    if local_records:
        _write_failures(output_dir / "local_grid" / "failed_chunks.parquet", local_records)
        _write_normalization(output_dir / "local_grid" / "normalization_events.parquet", local_records)
    if regional_records:
        _write_failures(output_dir / "regional_context" / "failed_chunks.parquet", regional_records)
        _write_normalization(output_dir / "regional_context" / "normalization_events.parquet", regional_records)

    summary = {
        "input_dir": str(input_dir),
        "manifest_path": str(args.manifest_path),
        "output_dir": str(output_dir),
        "year": int(args.year),
        "local_sequence_count": int(len(local_records)),
        "regional_sequence_count": int(len(regional_records)),
        "local_region_count": int(len({str(record["region_id"]) for record in local_records})),
        "regional_region_count": int(len({str(record["region_id"]) for record in regional_records})),
        "output_format": str(args.output_format),
        "local_npz_rows": int(len(local_rows)),
        "regional_npz_rows": int(len(regional_rows)),
        "local_zarr_rows": int(len(local_zarr_rows)),
        "regional_zarr_rows": int(len(regional_zarr_rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
