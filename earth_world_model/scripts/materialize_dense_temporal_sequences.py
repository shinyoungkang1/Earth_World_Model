#!/usr/bin/env python3
"""Materialize dense temporal Sentinel-2 + Sentinel-1 sequences from a planned manifest.

This consumes the output of `plan_dense_temporal_sequences.py`, fetches the
selected STAC items, crops aligned chips for each weekly / biweekly bin, and
writes per-sequence `.npz` tensors plus index metadata.

The first version is intentionally simple:

- sequential execution
- resumable via `--skip-existing`
- shardable via `--shard-index` / `--shard-count`
- generic STAC backend with optional signing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pystac_client import Client


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dense_temporal_materialization import (  # noqa: E402
    DEFAULT_CDSE_PROCESS_URL,
    DEFAULT_CDSE_REQUEST_TIMEOUT,
    DEFAULT_CDSE_RESOLUTION_M,
    DEFAULT_CDSE_TOKEN_URL,
    DEFAULT_S1_BAND_ASSETS,
    DEFAULT_S2_BAND_ASSETS,
    DEFAULT_S2_INVALID_SCL,
    DEFAULT_S2_SCL_ASSET,
    DEFAULT_STAC_API_URL,
    DEFAULT_PIXEL_ACCESS_MODE,
    assign_sequence_shard,
    empty_frame,
    fetch_item_by_id,
    load_table,
    parse_asset_list,
    parse_invalid_scl_values,
    parse_iso_day_of_year,
    read_s2s1_frame_process_fused,
    read_s1_frame_process,
    read_s1_frame,
    read_s2_frame_process,
    read_s2_frame,
    reference_window_and_transform,
    require_cdse_client_credentials,
    resolve_asset,
    sanitize_id,
    write_json,
)
from ewm.data.dense_temporal_writer import write_dense_temporal_zarr_shard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize dense temporal Sentinel-2 + Sentinel-1 sequences from sequence_plan.parquet."
    )
    parser.add_argument("--plan-path", required=True, help="Planner output parquet with per-bin item selections.")
    parser.add_argument("--output-dir", default=None, help="Directory for .npz outputs and metadata.")
    parser.add_argument("--stac-api-url", default=DEFAULT_STAC_API_URL)
    parser.add_argument("--signing-mode", default="none", choices=["none", "planetary_computer"])
    parser.add_argument(
        "--pixel-access-mode",
        default=DEFAULT_PIXEL_ACCESS_MODE,
        choices=["direct", "cdse_process"],
        help="Use raw asset reads or authenticated Copernicus Data Space process requests.",
    )
    parser.add_argument("--cdse-process-url", default=DEFAULT_CDSE_PROCESS_URL)
    parser.add_argument("--cdse-token-url", default=DEFAULT_CDSE_TOKEN_URL)
    parser.add_argument("--cdse-resolution-m", type=float, default=DEFAULT_CDSE_RESOLUTION_M)
    parser.add_argument("--cdse-request-timeout", type=int, default=DEFAULT_CDSE_REQUEST_TIMEOUT)
    parser.add_argument(
        "--fuse-s1-s2-per-bin",
        action="store_true",
        help="For cdse_process mode, request S2+S1 together in one fused Process API call when both are present.",
    )
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--sequence-limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-paired-bins", type=int, default=1)
    parser.add_argument("--min-materialized-paired-bins", type=int, default=1)
    parser.add_argument("--paired-bins-only", action="store_true")
    parser.add_argument("--max-bins-per-sequence", type=int, default=0)
    parser.add_argument("--output-format", default="npz", choices=["npz", "zarr", "both"])
    parser.add_argument("--zarr-shard-size", type=int, default=32)
    parser.add_argument("--zarr-compressor-clevel", type=int, default=5)
    parser.add_argument(
        "--s2-band-assets",
        default=",".join(DEFAULT_S2_BAND_ASSETS),
        help="Comma-separated Sentinel-2 band asset keys to materialize.",
    )
    parser.add_argument(
        "--s1-band-assets",
        default=",".join(DEFAULT_S1_BAND_ASSETS),
        help="Comma-separated Sentinel-1 band asset keys to materialize.",
    )
    parser.add_argument("--s2-scl-asset", default=DEFAULT_S2_SCL_ASSET)
    parser.add_argument(
        "--s2-invalid-scl",
        default=",".join(str(value) for value in DEFAULT_S2_INVALID_SCL),
        help="Comma-separated invalid Sentinel-2 SCL class values.",
    )
    return parser.parse_args()


def resolve_output_dir(plan_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
        return output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir)
    return plan_path.parent / f"{plan_path.stem}_materialized_npz"


def bool_from_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def build_sequence_frame(plan_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        plan_frame.groupby("sequence_id", dropna=False)
        .agg(
            sample_id=("sample_id", "first"),
            anchor_year=("anchor_year", "first"),
            search_group_id=("search_group_id", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            bin_count=("bin_index", "count"),
            paired_found_bins=("paired_found", lambda series: int(pd.Series(series).fillna(False).astype(bool).sum())),
            s2_found_bins=("s2_found", lambda series: int(pd.Series(series).fillna(False).astype(bool).sum())),
            s1_found_bins=("s1_found", lambda series: int(pd.Series(series).fillna(False).astype(bool).sum())),
        )
        .reset_index()
    )
    return grouped.sort_values("sequence_id").reset_index(drop=True)


def trim_sequence_frame(
    sequence_frame: pd.DataFrame,
    *,
    min_paired_bins: int,
    sequence_offset: int,
    sequence_limit: int,
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    result = sequence_frame[sequence_frame["paired_found_bins"] >= int(min_paired_bins)].copy()
    result = assign_sequence_shard(result, shard_index=shard_index, shard_count=shard_count)
    if sequence_offset > 0:
        result = result.iloc[sequence_offset:].copy()
    if sequence_limit > 0:
        result = result.head(sequence_limit).copy()
    return result.reset_index(drop=True)


def summarize_plan(sequence_frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "plan_path": str(Path(args.plan_path).resolve()),
        "output_dir": str(output_dir),
        "sequence_count_selected": int(len(sequence_frame)),
        "sequence_offset": int(args.sequence_offset),
        "sequence_limit": int(args.sequence_limit),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "min_paired_bins": int(args.min_paired_bins),
        "min_materialized_paired_bins": int(args.min_materialized_paired_bins),
        "paired_bins_only": bool(args.paired_bins_only),
        "max_bins_per_sequence": int(args.max_bins_per_sequence),
        "stac_api_url": args.stac_api_url,
        "signing_mode": args.signing_mode,
        "pixel_access_mode": args.pixel_access_mode,
        "cdse_process_url": args.cdse_process_url,
        "cdse_token_url": args.cdse_token_url,
        "cdse_resolution_m": float(args.cdse_resolution_m),
        "cdse_request_timeout": int(args.cdse_request_timeout),
        "fuse_s1_s2_per_bin": bool(args.fuse_s1_s2_per_bin),
        "chip_size": int(args.chip_size),
        "output_format": str(args.output_format),
        "zarr_shard_size": int(args.zarr_shard_size),
        "s2_band_assets": parse_asset_list(args.s2_band_assets),
        "s1_band_assets": parse_asset_list(args.s1_band_assets),
        "s2_scl_asset": args.s2_scl_asset,
        "s2_invalid_scl": parse_invalid_scl_values(args.s2_invalid_scl),
        "dry_run": bool(args.dry_run),
    }
    if not sequence_frame.empty:
        summary["selected_anchor_years"] = sorted(int(value) for value in sequence_frame["anchor_year"].dropna().unique())
    else:
        summary["selected_anchor_years"] = []
    return summary


def main() -> int:
    args = parse_args()

    plan_path = Path(args.plan_path)
    output_dir = resolve_output_dir(plan_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    zarr_shards_dir = output_dir / "zarr_shards"
    if args.output_format in {"zarr", "both"}:
        zarr_shards_dir.mkdir(parents=True, exist_ok=True)

    plan_frame = load_table(plan_path)
    required_columns = {
        "sequence_id",
        "sample_id",
        "anchor_year",
        "latitude",
        "longitude",
        "bin_index",
        "bin_start",
        "bin_end",
        "s2_found",
        "s1_found",
        "paired_found",
        "s2_item_id",
        "s2_collection",
        "s2_datetime",
        "s1_item_id",
        "s1_collection",
        "s1_datetime",
    }
    missing_columns = sorted(required_columns - set(plan_frame.columns))
    if missing_columns:
        raise ValueError(f"Plan is missing required columns: {missing_columns}")

    sequence_frame = build_sequence_frame(plan_frame)
    sequence_frame = trim_sequence_frame(
        sequence_frame,
        min_paired_bins=args.min_paired_bins,
        sequence_offset=args.sequence_offset,
        sequence_limit=args.sequence_limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    selected_sequences_path = output_dir / "selected_sequences.parquet"
    sequence_frame.to_parquet(selected_sequences_path, index=False)

    summary = summarize_plan(sequence_frame, output_dir, args)
    if args.dry_run or sequence_frame.empty:
        write_json(output_dir / "metadata.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    catalog = Client.open(args.stac_api_url)
    if str(args.pixel_access_mode) == "cdse_process":
        require_cdse_client_credentials()
    item_cache: dict[tuple[str, str], Any] = {}
    s2_band_assets = parse_asset_list(args.s2_band_assets)
    s1_band_assets = parse_asset_list(args.s1_band_assets)
    invalid_scl_values = parse_invalid_scl_values(args.s2_invalid_scl)

    index_rows: list[dict[str, Any]] = []
    zarr_index_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    written_sequences = 0
    reused_sequences = 0
    written_zarr_shards = 0
    zarr_buffer: list[dict[str, Any]] = []

    def flush_zarr_buffer() -> None:
        nonlocal written_zarr_shards, zarr_buffer, zarr_index_rows
        if not zarr_buffer:
            return
        shard_path = zarr_shards_dir / f"dense_temporal_shard_{written_zarr_shards:05d}.zarr"
        zarr_rows = write_dense_temporal_zarr_shard(
            shard_path,
            zarr_buffer,
            chunk_sequences=args.zarr_shard_size,
            compressor_clevel=args.zarr_compressor_clevel,
        )
        zarr_index_rows.extend(zarr_rows)
        written_zarr_shards += 1
        zarr_buffer = []

    for sequence_idx, sequence_row in enumerate(sequence_frame.itertuples(index=False), start=1):
        sequence_id = str(sequence_row.sequence_id)
        sample_id = str(sequence_row.sample_id)
        anchor_year = int(sequence_row.anchor_year)
        latitude = float(sequence_row.latitude)
        longitude = float(sequence_row.longitude)
        sequence_plan = (
            plan_frame[plan_frame["sequence_id"] == sequence_id]
            .sort_values("bin_index")
            .reset_index(drop=True)
            .copy()
        )
        if args.paired_bins_only:
            sequence_plan = sequence_plan[sequence_plan["paired_found"].fillna(False).astype(bool)].copy()
        if args.max_bins_per_sequence > 0:
            sequence_plan = sequence_plan.head(int(args.max_bins_per_sequence)).copy()
        sequence_plan = sequence_plan.reset_index(drop=True)

        sequence_path = sequences_dir / f"{sanitize_id(sequence_id)}.npz"
        if args.skip_existing and args.output_format == "npz" and sequence_path.exists():
            reused_sequences += 1
            index_rows.append(
                {
                    "sequence_id": sequence_id,
                    "sample_id": sample_id,
                    "anchor_year": anchor_year,
                    "search_group_id": getattr(sequence_row, "search_group_id", None),
                    "latitude": latitude,
                    "longitude": longitude,
                    "sequence_path": str(sequence_path),
                    "frame_count": int(len(sequence_plan)),
                    "paired_frame_count": int(sequence_plan["paired_found"].fillna(False).astype(bool).sum()),
                    "s2_frame_count": int(sequence_plan["s2_found"].fillna(False).astype(bool).sum()),
                    "s1_frame_count": int(sequence_plan["s1_found"].fillna(False).astype(bool).sum()),
                    "frame_mask_json": json.dumps(sequence_plan["paired_found"].fillna(False).astype(bool).tolist()),
                    "s2_frame_mask_json": json.dumps(sequence_plan["s2_found"].fillna(False).astype(bool).tolist()),
                    "s1_frame_mask_json": json.dumps(sequence_plan["s1_found"].fillna(False).astype(bool).tolist()),
                    "day_of_year_json": json.dumps(
                        [parse_iso_day_of_year(value) for value in sequence_plan["s2_datetime"].fillna(sequence_plan["s1_datetime"]).tolist()]
                    ),
                    "bin_indices_json": json.dumps(sequence_plan["bin_index"].astype(int).tolist()),
                    "frame_metadata_json": json.dumps([]),
                    "status": "reused_existing",
                }
            )
            if sequence_idx == 1 or sequence_idx % 10 == 0:
                print(
                    f"[dense-materialize] sequence={sequence_idx}/{len(sequence_frame)} "
                    f"written={written_sequences} reused={reused_sequences} failures={len(failure_rows)}",
                    flush=True,
                )
            continue

        s2_frames: list[np.ndarray] = []
        s1_frames: list[np.ndarray] = []
        s2_valid_masks: list[np.ndarray] = []
        s1_valid_masks: list[np.ndarray] = []
        frame_mask: list[bool] = []
        s2_frame_mask: list[bool] = []
        s1_frame_mask: list[bool] = []
        day_of_year: list[int] = []
        bin_indices: list[int] = []
        frame_metadata: list[dict[str, Any]] = []

        if sequence_plan.empty:
            failure_rows.append(
                {
                    "sequence_id": sequence_id,
                    "bin_index": -1,
                    "modality": "sequence",
                    "item_id": None,
                    "reason": "sequence_plan_empty_after_filters",
                }
            )
            continue

        for plan_row in sequence_plan.itertuples(index=False):
            bin_index = int(plan_row.bin_index)
            bin_indices.append(bin_index)
            day_of_year.append(parse_iso_day_of_year(plan_row.s2_datetime or plan_row.s1_datetime))

            s2_ok = False
            s1_ok = False
            s2_data, s2_mask = empty_frame(band_count=len(s2_band_assets), chip_size=args.chip_size)
            s1_data, s1_mask = empty_frame(band_count=len(s1_band_assets), chip_size=args.chip_size)
            dst_crs = None
            dst_transform = None
            frame_meta: dict[str, Any] = {
                "bin_index": bin_index,
                "bin_start": str(plan_row.bin_start),
                "bin_end": str(plan_row.bin_end),
                "s2_found": bool_from_value(plan_row.s2_found),
                "s1_found": bool_from_value(plan_row.s1_found),
                "paired_found": bool_from_value(plan_row.paired_found),
            }

            fused_done = False
            if (
                str(args.pixel_access_mode) == "cdse_process"
                and bool(args.fuse_s1_s2_per_bin)
                and bool_from_value(plan_row.s2_found)
                and bool_from_value(plan_row.s1_found)
                and plan_row.s2_item_id
                and plan_row.s1_item_id
            ):
                try:
                    s2_item = fetch_item_by_id(
                        catalog,
                        collection=str(plan_row.s2_collection),
                        item_id=str(plan_row.s2_item_id),
                        signing_mode=args.signing_mode,
                        cache=item_cache,
                    )
                    s1_item = fetch_item_by_id(
                        catalog,
                        collection=str(plan_row.s1_collection),
                        item_id=str(plan_row.s1_item_id),
                        signing_mode=args.signing_mode,
                        cache=item_cache,
                    )
                    (
                        s2_data,
                        s2_mask,
                        s2_meta,
                        s1_data,
                        s1_mask,
                        s1_meta,
                        dst_crs,
                        dst_transform,
                    ) = read_s2s1_frame_process_fused(
                        s2_item,
                        s1_item,
                        lon=longitude,
                        lat=latitude,
                        chip_size=args.chip_size,
                        s2_band_assets=s2_band_assets,
                        s2_scl_asset=(args.s2_scl_asset or None),
                        invalid_scl_values=invalid_scl_values,
                        s1_band_assets=s1_band_assets,
                        process_url=args.cdse_process_url,
                        token_url=args.cdse_token_url,
                        resolution_m=args.cdse_resolution_m,
                        request_timeout=args.cdse_request_timeout,
                    )
                    frame_meta["s2"] = s2_meta
                    frame_meta["s1"] = s1_meta
                    frame_meta["fused_request"] = True
                    s2_ok = bool(s2_mask.any())
                    s1_ok = bool(s1_mask.any())
                    fused_done = True
                except Exception as exc:
                    failure_rows.append(
                        {
                            "sequence_id": sequence_id,
                            "bin_index": bin_index,
                            "modality": "fused",
                            "item_id": f"{plan_row.s2_item_id}|{plan_row.s1_item_id}",
                            "reason": str(exc),
                        }
                    )
                    frame_meta["fused_error"] = str(exc)

            if (not fused_done) and bool_from_value(plan_row.s2_found) and plan_row.s2_item_id:
                try:
                    s2_item = fetch_item_by_id(
                        catalog,
                        collection=str(plan_row.s2_collection),
                        item_id=str(plan_row.s2_item_id),
                        signing_mode=args.signing_mode,
                        cache=item_cache,
                    )
                    if str(args.pixel_access_mode) == "cdse_process":
                        s2_data, s2_mask, s2_meta, dst_crs, dst_transform = read_s2_frame_process(
                            s2_item,
                            lon=longitude,
                            lat=latitude,
                            chip_size=args.chip_size,
                            band_assets=s2_band_assets,
                            scl_asset=(args.s2_scl_asset or None),
                            invalid_scl_values=invalid_scl_values,
                            process_url=args.cdse_process_url,
                            token_url=args.cdse_token_url,
                            resolution_m=args.cdse_resolution_m,
                            request_timeout=args.cdse_request_timeout,
                        )
                    else:
                        s2_data, s2_mask, s2_meta, dst_crs, dst_transform = read_s2_frame(
                            s2_item,
                            lon=longitude,
                            lat=latitude,
                            chip_size=args.chip_size,
                            band_assets=s2_band_assets,
                            scl_asset=(args.s2_scl_asset or None),
                            invalid_scl_values=invalid_scl_values,
                        )
                    frame_meta["s2"] = s2_meta
                    s2_ok = bool(s2_mask.any())
                except Exception as exc:
                    failure_rows.append(
                        {
                            "sequence_id": sequence_id,
                            "bin_index": bin_index,
                            "modality": "s2",
                            "item_id": str(plan_row.s2_item_id),
                            "reason": str(exc),
                        }
                    )
                    frame_meta["s2_error"] = str(exc)

            if (not fused_done) and bool_from_value(plan_row.s1_found) and plan_row.s1_item_id:
                try:
                    s1_item = fetch_item_by_id(
                        catalog,
                        collection=str(plan_row.s1_collection),
                        item_id=str(plan_row.s1_item_id),
                        signing_mode=args.signing_mode,
                        cache=item_cache,
                    )
                    if str(args.pixel_access_mode) == "cdse_process":
                        s1_data, s1_mask, s1_meta = read_s1_frame_process(
                            s1_item,
                            lon=longitude,
                            lat=latitude,
                            chip_size=args.chip_size,
                            band_assets=s1_band_assets,
                            process_url=args.cdse_process_url,
                            token_url=args.cdse_token_url,
                            resolution_m=args.cdse_resolution_m,
                            request_timeout=args.cdse_request_timeout,
                        )
                    else:
                        if dst_crs is None or dst_transform is None:
                            reference_asset = resolve_asset(s1_item, s1_band_assets[0])
                            dst_crs, _window, dst_transform = reference_window_and_transform(
                                reference_asset.href,
                                lon=longitude,
                                lat=latitude,
                                chip_size=args.chip_size,
                            )
                        s1_data, s1_mask, s1_meta = read_s1_frame(
                            s1_item,
                            dst_crs=dst_crs,
                            dst_transform=dst_transform,
                            chip_size=args.chip_size,
                            band_assets=s1_band_assets,
                        )
                    frame_meta["s1"] = s1_meta
                    s1_ok = bool(s1_mask.any())
                except Exception as exc:
                    failure_rows.append(
                        {
                            "sequence_id": sequence_id,
                            "bin_index": bin_index,
                            "modality": "s1",
                            "item_id": str(plan_row.s1_item_id),
                            "reason": str(exc),
                        }
                    )
                    frame_meta["s1_error"] = str(exc)

            s2_frames.append(s2_data)
            s1_frames.append(s1_data)
            s2_valid_masks.append(s2_mask)
            s1_valid_masks.append(s1_mask)
            s2_frame_mask.append(s2_ok)
            s1_frame_mask.append(s1_ok)
            frame_mask.append(s2_ok and s1_ok)
            frame_metadata.append(frame_meta)

        paired_frame_count = int(sum(frame_mask))
        if paired_frame_count < int(args.min_materialized_paired_bins):
            failure_rows.append(
                {
                    "sequence_id": sequence_id,
                    "bin_index": -1,
                    "modality": "sequence",
                    "item_id": None,
                    "reason": (
                        f"paired_frame_count={paired_frame_count} below min_materialized_paired_bins="
                        f"{args.min_materialized_paired_bins}"
                    ),
                }
            )
            if sequence_idx == 1 or sequence_idx % 10 == 0:
                print(
                    f"[dense-materialize] sequence={sequence_idx}/{len(sequence_frame)} "
                    f"written={written_sequences} reused={reused_sequences} failures={len(failure_rows)}",
                    flush=True,
                )
            continue

        s2_array = np.stack(s2_frames, axis=0).astype(np.float32)
        s1_array = np.stack(s1_frames, axis=0).astype(np.float32)
        frame_mask_array = np.asarray(frame_mask, dtype=bool)
        s2_frame_mask_array = np.asarray(s2_frame_mask, dtype=bool)
        s1_frame_mask_array = np.asarray(s1_frame_mask, dtype=bool)
        s2_valid_mask_array = np.stack(s2_valid_masks, axis=0).astype(bool)
        s1_valid_mask_array = np.stack(s1_valid_masks, axis=0).astype(bool)
        day_of_year_array = np.asarray(day_of_year, dtype=np.int16)
        bin_index_array = np.asarray(bin_indices, dtype=np.int16)

        if args.output_format in {"npz", "both"}:
            np.savez_compressed(
                sequence_path,
                s2=s2_array,
                s1=s1_array,
                frame_mask=frame_mask_array,
                s2_frame_mask=s2_frame_mask_array,
                s1_frame_mask=s1_frame_mask_array,
                s2_valid_mask=s2_valid_mask_array,
                s1_valid_mask=s1_valid_mask_array,
                day_of_year=day_of_year_array,
                bin_index=bin_index_array,
            )
        if args.output_format in {"zarr", "both"}:
            zarr_buffer.append(
                {
                    "sequence_id": sequence_id,
                    "sample_id": sample_id,
                    "search_group_id": getattr(sequence_row, "search_group_id", None),
                    "anchor_year": anchor_year,
                    "latitude": latitude,
                    "longitude": longitude,
                    "frame_count": int(len(frame_mask)),
                    "paired_frame_count": paired_frame_count,
                    "s2": s2_array,
                    "s1": s1_array,
                    "frame_mask": frame_mask_array,
                    "s2_frame_mask": s2_frame_mask_array,
                    "s1_frame_mask": s1_frame_mask_array,
                    "s2_valid_mask": s2_valid_mask_array,
                    "s1_valid_mask": s1_valid_mask_array,
                    "day_of_year": day_of_year_array,
                    "bin_index": bin_index_array,
                    "frame_metadata": frame_metadata,
                }
            )
            if len(zarr_buffer) >= int(args.zarr_shard_size):
                flush_zarr_buffer()
        written_sequences += 1

        index_rows.append(
            {
                "sequence_id": sequence_id,
                "sample_id": sample_id,
                "anchor_year": anchor_year,
                "search_group_id": getattr(sequence_row, "search_group_id", None),
                "latitude": latitude,
                "longitude": longitude,
                "sequence_path": str(sequence_path) if args.output_format in {"npz", "both"} else None,
                "frame_count": int(len(frame_mask)),
                "paired_frame_count": paired_frame_count,
                "s2_frame_count": int(sum(s2_frame_mask)),
                "s1_frame_count": int(sum(s1_frame_mask)),
                "frame_mask_json": json.dumps([int(value) for value in frame_mask]),
                "s2_frame_mask_json": json.dumps([int(value) for value in s2_frame_mask]),
                "s1_frame_mask_json": json.dumps([int(value) for value in s1_frame_mask]),
                "day_of_year_json": json.dumps(day_of_year),
                "bin_indices_json": json.dumps(bin_indices),
                "frame_metadata_json": json.dumps(frame_metadata),
                "status": "written",
            }
        )

        if sequence_idx == 1 or sequence_idx % 10 == 0:
            print(
                f"[dense-materialize] sequence={sequence_idx}/{len(sequence_frame)} "
                f"written={written_sequences} reused={reused_sequences} failures={len(failure_rows)}",
                flush=True,
            )

    index_frame = pd.DataFrame(index_rows)
    failures_frame = pd.DataFrame(failure_rows)
    index_path = output_dir / "dense_temporal_index.parquet"
    failures_path = output_dir / "dense_temporal_failures.parquet"
    if args.output_format in {"zarr", "both"}:
        flush_zarr_buffer()
    zarr_index_frame = pd.DataFrame(zarr_index_rows)
    zarr_index_path = output_dir / "dense_temporal_zarr_index.parquet"
    index_frame.to_parquet(index_path, index=False)
    failures_frame.to_parquet(failures_path, index=False)
    if args.output_format in {"zarr", "both"}:
        zarr_index_frame.to_parquet(zarr_index_path, index=False)

    summary.update(
        {
            "selected_sequences_path": str(selected_sequences_path),
            "index_path": str(index_path),
            "failures_path": str(failures_path),
            "written_sequences": int(written_sequences),
            "reused_sequences": int(reused_sequences),
            "failure_row_count": int(len(failure_rows)),
            "item_cache_size": int(len(item_cache)),
            "written_zarr_shards": int(written_zarr_shards),
        }
    )
    if args.output_format in {"zarr", "both"}:
        summary["zarr_index_path"] = str(zarr_index_path)
    write_json(output_dir / "metadata.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
