#!/usr/bin/env python3
"""Convert exact-chip Earth Engine GeoTIFF exports into dense temporal training files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as raster_merge


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dense_temporal_materialization import DEFAULT_S2_INVALID_SCL, sanitize_id  # noqa: E402
from ewm.data.dense_temporal_writer import write_dense_temporal_zarr_shard  # noqa: E402


S2_BAND_COUNT = 10
SCL_BAND_COUNT = 1
S1_BAND_COUNT = 2
PRESENCE_BAND_COUNT = 2
EE_BANDS_PER_WEEK = S2_BAND_COUNT + SCL_BAND_COUNT + S1_BAND_COUNT + PRESENCE_BAND_COUNT
WEEK_PATTERN = re.compile(r"w(?P<start>\d{2})(?:_(?P<end>\d{2}))?")
TILE_SUFFIX_PATTERN = re.compile(r"(?P<prefix>.+?)(?P<x>\d{10})-(?P<y>\d{10})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert exact-chip Earth Engine weekly GeoTIFF exports into dense temporal npz/zarr shards."
    )
    parser.add_argument("--input-dir", required=True, help="Root directory containing sample subdirectories with TIFF chunks.")
    parser.add_argument(
        "--manifests-dir",
        default=None,
        help="Optional directory with per-export JSON manifests from run_earth_engine_shard_export.py.",
    )
    parser.add_argument(
        "--locations-path",
        default=None,
        help="Optional CSV with sample_id,latitude,longitude used to fill sample metadata when manifests are absent.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--output-format", default="npz", choices=["npz", "zarr", "both"])
    parser.add_argument("--zarr-shard-size", type=int, default=32)
    parser.add_argument("--zarr-compressor-clevel", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


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


def center_crop_hw(array: np.ndarray, target_size: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array [bands,height,width], got shape={array.shape}")
    _bands, height, width = array.shape
    if height < target_size or width < target_size:
        raise ValueError(f"Cannot center crop shape={array.shape} to target_size={target_size}")
    if height == target_size and width == target_size:
        return array
    y0 = max(0, (height - target_size) // 2)
    x0 = max(0, (width - target_size) // 2)
    return array[:, y0 : y0 + target_size, x0 : x0 + target_size]


def load_locations(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    mapping: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            mapping[sample_id] = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
    return mapping


def load_manifests(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    manifests: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        if json_path.name in {"summary.json", "launch_summary.json", "rerun_summary.json", "live_status.json"}:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        points = payload.get("points") or []
        sample_id = None
        if len(points) == 1:
            candidate = points[0].get("sample_id")
            if candidate and candidate != "center":
                sample_id = str(candidate)
        if sample_id is None:
            description = str(payload.get("description") or "")
            match = re.search(r"_([A-Za-z0-9_]+)_\d{4}_w\d{2}", description)
            if match:
                sample_id = match.group(1)
        if sample_id:
            manifests[str(sample_id)] = payload
    return manifests


def discover_sample_tiffs(input_dir: Path, sample_limit: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for tif_path in sorted(input_dir.rglob("*.tif")):
        if tif_path.parent == input_dir:
            sample_id = tif_path.stem
        else:
            sample_id = tif_path.parent.name
        groups.setdefault(str(sample_id), []).append(tif_path)
    if sample_limit > 0:
        limited = dict(list(sorted(groups.items()))[:sample_limit])
        return limited
    return dict(sorted(groups.items()))


def maybe_load_json_error(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if isinstance(payload, dict) and "error" in payload:
        return payload
    return None


def tiled_export_group_key(path: Path) -> str:
    match = TILE_SUFFIX_PATTERN.match(path.stem)
    if match:
        return match.group("prefix")
    return path.stem


def read_tif_or_mosaic(paths: list[Path]) -> tuple[np.ndarray, Path]:
    ordered = sorted(paths)
    if len(ordered) == 1:
        with rasterio.open(ordered[0]) as ds:
            return ds.read().astype(np.float32), ordered[0]

    datasets = [rasterio.open(path) for path in ordered]
    try:
        merged, _ = raster_merge(datasets)
    finally:
        for ds in datasets:
            ds.close()
    return merged.astype(np.float32), ordered[0]


def frame_metadata_for_week(
    *,
    sample_id: str,
    tif_path: Path,
    week_index: int,
    year: int,
    manifests: dict[str, dict[str, Any]],
    has_s2: bool,
    has_s1: bool,
) -> dict[str, Any]:
    payload = manifests.get(sample_id)
    week_meta: dict[str, Any] | None = None
    if payload is not None:
        for entry in payload.get("weekly_manifest") or []:
            if int(entry.get("week_index", -1)) == int(week_index):
                week_meta = dict(entry)
                break

    windows = weekly_windows(year)
    if 0 <= week_index < len(windows):
        start, end = windows[week_index]
        start_iso = start.isoformat()
        end_iso = end.isoformat()
    else:
        start_iso = None
        end_iso = None

    result = {
        "source": "earth_engine_export",
        "sample_id": sample_id,
        "week_index": int(week_index),
        "source_tif": str(tif_path),
        "has_s2": bool(has_s2),
        "has_s1": bool(has_s1),
        "start": start_iso,
        "end": end_iso,
    }
    if week_meta is not None:
        result.update(week_meta)
    return result


def day_of_year_for_week(week_index: int, year: int, frame_meta: dict[str, Any]) -> int:
    for key in ("start",):
        value = frame_meta.get(key)
        if value:
            return int(pd.Timestamp(value).dayofyear)
    windows = weekly_windows(year)
    if 0 <= week_index < len(windows):
        return int(pd.Timestamp(windows[week_index][0]).dayofyear)
    return -1


def read_ee_tif_records(
    *,
    sample_id: str,
    tif_paths: list[Path],
    manifests: dict[str, dict[str, Any]],
    locations: dict[str, dict[str, float]],
    year: int,
    chip_size: int,
) -> dict[str, Any]:
    week_records: list[dict[str, Any]] = []
    failed_chunks: list[dict[str, Any]] = []
    tif_groups: dict[str, list[Path]] = {}
    for tif_path in sorted(tif_paths):
        tif_groups.setdefault(tiled_export_group_key(tif_path), []).append(tif_path)

    for _group_key, grouped_paths in sorted(tif_groups.items()):
        tif_path = sorted(grouped_paths)[0]
        try:
            array, tif_path = read_tif_or_mosaic(grouped_paths)
        except Exception as exc:
            error_payload = maybe_load_json_error(tif_path) if len(grouped_paths) == 1 else None
            failed_chunks.append(
                {
                    "sample_id": sample_id,
                    "source_tif": json.dumps([str(path) for path in grouped_paths]),
                    "reason": str(exc),
                    "error_payload": error_payload,
                }
            )
            continue
        cropped = center_crop_hw(array, chip_size)
        band_count = cropped.shape[0]
        if band_count % EE_BANDS_PER_WEEK != 0:
            raise ValueError(f"Unexpected EE band count={band_count} in {tif_path}")
        week_start, week_count = parse_week_start_and_count(tif_path, band_count)

        for offset in range(int(week_count)):
            base = offset * EE_BANDS_PER_WEEK
            s2 = cropped[base : base + S2_BAND_COUNT].astype(np.float32)
            scl = cropped[base + S2_BAND_COUNT].astype(np.float32)
            s1 = cropped[base + S2_BAND_COUNT + SCL_BAND_COUNT : base + S2_BAND_COUNT + SCL_BAND_COUNT + S1_BAND_COUNT].astype(
                np.float32
            )
            presence = cropped[
                base
                + S2_BAND_COUNT
                + SCL_BAND_COUNT
                + S1_BAND_COUNT : base
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
                manifests=manifests,
                has_s2=bool(s2_valid_mask.any()),
                has_s1=bool(s1_valid_mask.any()),
            )
            week_records.append(
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

    week_records.sort(key=lambda entry: int(entry["week_index"]))
    if len({int(entry["week_index"]) for entry in week_records}) != len(week_records):
        raise ValueError(f"Duplicate week indices detected for sample_id={sample_id}")
    if not week_records:
        raise ValueError(f"No readable weekly exports found for sample_id={sample_id}")

    location = locations.get(sample_id, {})
    latitude = float(location.get("latitude", np.nan))
    longitude = float(location.get("longitude", np.nan))
    sequence_id = f"ee::{sample_id}::{year}"

    frame_mask = np.asarray([bool(entry["s2_ok"] and entry["s1_ok"]) for entry in week_records], dtype=bool)
    s2_frame_mask = np.asarray([bool(entry["s2_ok"]) for entry in week_records], dtype=bool)
    s1_frame_mask = np.asarray([bool(entry["s1_ok"]) for entry in week_records], dtype=bool)
    return {
        "sequence_id": sequence_id,
        "sample_id": sample_id,
        "search_group_id": "",
        "anchor_year": int(year),
        "latitude": latitude,
        "longitude": longitude,
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
    }


def write_npz_records(output_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    locations = load_locations(Path(args.locations_path) if args.locations_path else None)
    manifests = load_manifests(Path(args.manifests_dir) if args.manifests_dir else None)
    sample_groups = discover_sample_tiffs(input_dir, int(args.sample_limit))
    if not sample_groups:
        raise SystemExit(f"No TIFF exports found under {input_dir}")

    records: list[dict[str, Any]] = []
    for sample_id, tif_paths in sample_groups.items():
        record = read_ee_tif_records(
            sample_id=sample_id,
            tif_paths=tif_paths,
            manifests=manifests,
            locations=locations,
            year=int(args.year),
            chip_size=int(args.chip_size),
        )
        records.append(record)

    npz_rows: list[dict[str, Any]] = []
    zarr_rows: list[dict[str, Any]] = []
    if args.output_format in {"npz", "both"}:
        npz_rows = write_npz_records(output_dir, records)
        pd.DataFrame(npz_rows).to_parquet(output_dir / "dense_temporal_index.parquet", index=False)

    if args.output_format in {"zarr", "both"}:
        zarr_dir = output_dir / "zarr_shards"
        zarr_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(0, len(records), int(args.zarr_shard_size)):
            shard_records = records[shard_idx : shard_idx + int(args.zarr_shard_size)]
            shard_path = zarr_dir / f"dense_temporal_shard_{shard_idx // int(args.zarr_shard_size):05d}.zarr"
            zarr_rows.extend(
                write_dense_temporal_zarr_shard(
                    shard_path,
                    shard_records,
                    compressor_clevel=int(args.zarr_compressor_clevel),
                )
            )
        pd.DataFrame(zarr_rows).to_parquet(output_dir / "dense_temporal_zarr_index.parquet", index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sample_count": int(len(records)),
        "frame_count_total": int(sum(int(record["frame_count"]) for record in records)),
        "paired_frame_count_total": int(sum(int(record["paired_frame_count"]) for record in records)),
        "failed_chunk_count": int(sum(len(record.get("failed_chunks", [])) for record in records)),
        "output_format": str(args.output_format),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failure_rows = [
        failure
        for record in records
        for failure in record.get("failed_chunks", [])
    ]
    if failure_rows:
        pd.DataFrame(failure_rows).to_parquet(output_dir / "failed_chunks.parquet", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
