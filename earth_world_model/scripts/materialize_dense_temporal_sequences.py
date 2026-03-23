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
import time
from typing import Any

import numpy as np
import pandas as pd
from pystac_client import Client


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dense_temporal_materialization import (  # noqa: E402
    DEFAULT_CDSE_S3_ENDPOINT_URL,
    DEFAULT_CDSE_S3_REGION,
    DEFAULT_CDSE_PROCESS_URL,
    DEFAULT_CDSE_REQUEST_TIMEOUT,
    DEFAULT_CDSE_RESOLUTION_M,
    DEFAULT_CDSE_TOKEN_URL,
    DEFAULT_DIRECT_ASSET_SOURCE,
    DEFAULT_S1_BAND_ASSETS,
    DEFAULT_S2_BAND_ASSETS,
    DEFAULT_S2_INVALID_SCL,
    DEFAULT_S2_SCL_ASSET,
    DEFAULT_STAC_API_URL,
    DEFAULT_PIXEL_ACCESS_MODE,
    assign_sequence_shard,
    configure_direct_asset_access,
    empty_frame,
    evaluate_s2_chip_quality,
    fetch_item_by_id,
    fixed_chip_target_grid,
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
        "--direct-asset-source",
        default=DEFAULT_DIRECT_ASSET_SOURCE,
        choices=["remote", "cdse_s3_cache"],
        help="How direct exact-item raster reads should be sourced before rasterio opens them.",
    )
    parser.add_argument(
        "--local-asset-cache-dir",
        default=None,
        help="Local scratch directory for cached direct raster assets when using --direct-asset-source=cdse_s3_cache.",
    )
    parser.add_argument("--cdse-s3-endpoint-url", default=DEFAULT_CDSE_S3_ENDPOINT_URL)
    parser.add_argument("--cdse-s3-region", default=DEFAULT_CDSE_S3_REGION)
    parser.add_argument(
        "--pixel-access-mode",
        default=DEFAULT_PIXEL_ACCESS_MODE,
        choices=["direct", "cdse_process"],
        help="Use raw asset reads or authenticated Copernicus Data Space process requests.",
    )
    parser.add_argument(
        "--allow-noncanonical-pixel-access",
        action="store_true",
        help=(
            "Allow cdse_process for exploratory runs. Canonical dense corpora default to direct exact-item "
            "materialization so sensor preprocessing stays consistent across all sequences."
        ),
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
        "--materialization-order",
        default="scene",
        choices=["scene", "sequence"],
        help="Process frames in source-scene order within each search group to improve raster cache reuse.",
    )
    parser.add_argument(
        "--log-every-frame-requests",
        type=int,
        default=25,
        help="Emit a progress line every N frame requests within the active search group.",
    )
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


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


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
        "direct_asset_source": args.direct_asset_source,
        "local_asset_cache_dir": str(Path(args.local_asset_cache_dir).expanduser().resolve())
        if args.local_asset_cache_dir
        else None,
        "cdse_s3_endpoint_url": args.cdse_s3_endpoint_url,
        "cdse_s3_region": args.cdse_s3_region,
        "pixel_access_mode": args.pixel_access_mode,
        "allow_noncanonical_pixel_access": bool(args.allow_noncanonical_pixel_access),
        "cdse_process_url": args.cdse_process_url,
        "cdse_token_url": args.cdse_token_url,
        "cdse_resolution_m": float(args.cdse_resolution_m),
        "cdse_request_timeout": int(args.cdse_request_timeout),
        "fuse_s1_s2_per_bin": bool(args.fuse_s1_s2_per_bin),
        "chip_size": int(args.chip_size),
        "output_format": str(args.output_format),
        "zarr_shard_size": int(args.zarr_shard_size),
        "materialization_order": str(args.materialization_order),
        "log_every_frame_requests": int(args.log_every_frame_requests),
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

    if str(args.direct_asset_source) == "cdse_s3_cache" and not args.local_asset_cache_dir:
        raise ValueError(
            "--local-asset-cache-dir is required when --direct-asset-source=cdse_s3_cache."
        )

    configure_direct_asset_access(
        source=args.direct_asset_source,
        cache_dir=args.local_asset_cache_dir,
        cdse_s3_endpoint_url=args.cdse_s3_endpoint_url,
        cdse_s3_region=args.cdse_s3_region,
    )

    if str(args.pixel_access_mode) != "direct" and not bool(args.allow_noncanonical_pixel_access):
        raise ValueError(
            "Canonical dense temporal corpus builds require --pixel-access-mode=direct. "
            "cdse_process uses a different on-demand processing regime and is disabled unless "
            "--allow-noncanonical-pixel-access is set explicitly."
        )

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
    dropped_sequences = 0
    written_zarr_shards = 0
    zarr_buffer: list[dict[str, Any]] = []
    materialization_started_at = time.time()
    processed_frame_requests = 0
    processed_sequences = 0

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

    def format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def log_sequence_progress(
        *,
        sequence_id: str,
        status: str,
        frame_count: int | None = None,
        paired_frame_count: int | None = None,
        sequence_started_at: float | None = None,
    ) -> None:
        elapsed = format_duration(time.time() - materialization_started_at)
        message = (
            f"[dense-materialize] sequence={processed_sequences}/{len(sequence_frame)} "
            f"sequence_id={sequence_id} status={status} written={written_sequences} "
            f"reused={reused_sequences} dropped={dropped_sequences} failures={len(failure_rows)} "
            f"frame_requests={processed_frame_requests} elapsed={elapsed}"
        )
        if frame_count is not None:
            message += f" frame_count={int(frame_count)}"
        if paired_frame_count is not None:
            message += f" paired_frame_count={int(paired_frame_count)}"
        if sequence_started_at is not None:
            message += f" sequence_elapsed={format_duration(time.time() - sequence_started_at)}"
        print(message, flush=True)

    def scene_sort_key(request: dict[str, Any]) -> tuple[Any, ...]:
        def normalize_scene_key(found: bool, collection: Any, item_id: Any) -> tuple[int, str, str]:
            if found and collection and item_id:
                return (0, str(collection), str(item_id))
            return (1, "", "")

        return (
            normalize_scene_key(request["s2_found"], request["s2_collection"], request["s2_item_id"]),
            normalize_scene_key(request["s1_found"], request["s1_collection"], request["s1_item_id"]),
            int(request["bin_position"]),
            str(request["sequence_id"]),
        )

    def materialize_frame_request(request: dict[str, Any]) -> dict[str, Any]:
        sequence_id = str(request["sequence_id"])
        bin_index = int(request["bin_index"])
        latitude = float(request["latitude"])
        longitude = float(request["longitude"])
        sequence_dst_crs = request.get("sequence_dst_crs")
        sequence_dst_transform = request.get("sequence_dst_transform")
        s2_item_id = request.get("s2_item_id")
        s2_collection = request.get("s2_collection")
        s2_shortlist = parse_json_list(request.get("s2_candidate_items_json"))

        s2_ok = False
        s1_ok = False
        s2_data, s2_mask = empty_frame(band_count=len(s2_band_assets), chip_size=args.chip_size)
        s1_data, s1_mask = empty_frame(band_count=len(s1_band_assets), chip_size=args.chip_size)
        dst_crs = None
        dst_transform = None
        frame_meta: dict[str, Any] = {
            "bin_index": bin_index,
            "bin_start": str(request["bin_start"]),
            "bin_end": str(request["bin_end"]),
            "s2_found": bool(request["s2_found"]),
            "s1_found": bool(request["s1_found"]),
            "paired_found": bool(request["paired_found"]),
        }

        if bool(request["s2_found"]) and (not s2_shortlist):
            fallback_entry = {
                "item_id": s2_item_id,
                "collection": s2_collection,
                "datetime": request.get("s2_datetime"),
                "cloud_cover": request.get("s2_cloud_cover"),
            }
            if fallback_entry["item_id"] and fallback_entry["collection"]:
                s2_shortlist = [fallback_entry]

        if bool(request["s2_found"]) and s2_shortlist:
            ranked_s2_shortlist: list[dict[str, Any]] = []
            for shortlist_entry in s2_shortlist:
                shortlist_item_id = shortlist_entry.get("item_id")
                shortlist_collection = shortlist_entry.get("collection")
                if not shortlist_item_id or not shortlist_collection:
                    continue
                try:
                    s2_item = fetch_item_by_id(
                        catalog,
                        collection=str(shortlist_collection),
                        item_id=str(shortlist_item_id),
                        signing_mode=args.signing_mode,
                        cache=item_cache,
                    )
                    _mask, quality_meta, quality_dst_crs, quality_dst_transform = evaluate_s2_chip_quality(
                        s2_item,
                        lon=longitude,
                        lat=latitude,
                        chip_size=args.chip_size,
                        scl_asset=(args.s2_scl_asset or None),
                        invalid_scl_values=invalid_scl_values,
                        dst_crs=sequence_dst_crs,
                        dst_transform=sequence_dst_transform,
                        reference_asset_key=s2_band_assets[0],
                    )
                    ranked_s2_shortlist.append(
                        {
                            **shortlist_entry,
                            "item": s2_item,
                            "quality_meta": quality_meta,
                            "sequence_dst_crs": quality_dst_crs,
                            "sequence_dst_transform": quality_dst_transform,
                        }
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "sequence_id": sequence_id,
                            "bin_index": bin_index,
                            "modality": "s2_quality",
                            "item_id": str(shortlist_item_id),
                            "reason": str(exc),
                        }
                    )
            if ranked_s2_shortlist:
                ranked_s2_shortlist.sort(
                    key=lambda entry: (
                        -(entry["quality_meta"].get("clear_fraction") or 0.0),
                        entry["quality_meta"].get("cloud_cover")
                        if entry["quality_meta"].get("cloud_cover") is not None
                        else 999.0,
                    )
                )
                chosen_entry = ranked_s2_shortlist[0]
                s2_item_id = chosen_entry.get("item_id")
                s2_collection = chosen_entry.get("collection")
                request["s2_item_id"] = s2_item_id
                request["s2_collection"] = s2_collection
                frame_meta["s2_shortlist_size"] = len(ranked_s2_shortlist)
                frame_meta["s2_reranked"] = len(ranked_s2_shortlist) > 1
                frame_meta["s2_selected_quality"] = chosen_entry["quality_meta"]
                frame_meta["s2_shortlist_quality_json"] = [
                    {
                        "item_id": entry.get("item_id"),
                        "cloud_cover": entry["quality_meta"].get("cloud_cover"),
                        "clear_fraction": entry["quality_meta"].get("clear_fraction"),
                    }
                    for entry in ranked_s2_shortlist
                ]
                if sequence_dst_crs is None or sequence_dst_transform is None:
                    sequence_dst_crs = chosen_entry.get("sequence_dst_crs")
                    sequence_dst_transform = chosen_entry.get("sequence_dst_transform")

        fused_done = False
        if (
            str(args.pixel_access_mode) == "cdse_process"
            and bool(args.fuse_s1_s2_per_bin)
            and bool(request["s2_found"])
            and bool(request["s1_found"])
            and request["s2_item_id"]
            and request["s1_item_id"]
            and len(s2_shortlist) <= 1
        ):
            try:
                s2_item = fetch_item_by_id(
                    catalog,
                    collection=str(request["s2_collection"]),
                    item_id=str(request["s2_item_id"]),
                    signing_mode=args.signing_mode,
                    cache=item_cache,
                )
                s1_item = fetch_item_by_id(
                    catalog,
                    collection=str(request["s1_collection"]),
                    item_id=str(request["s1_item_id"]),
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
                        "item_id": f"{request['s2_item_id']}|{request['s1_item_id']}",
                        "reason": str(exc),
                    }
                )
                frame_meta["fused_error"] = str(exc)

        if (not fused_done) and bool(request["s2_found"]) and request["s2_item_id"]:
            try:
                s2_item = fetch_item_by_id(
                    catalog,
                    collection=str(request["s2_collection"]),
                    item_id=str(request["s2_item_id"]),
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
                        dst_crs=sequence_dst_crs,
                        dst_transform=sequence_dst_transform,
                    )
                frame_meta["s2"] = s2_meta
                s2_ok = bool(s2_mask.any())
            except Exception as exc:
                failure_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "bin_index": bin_index,
                        "modality": "s2",
                        "item_id": str(request["s2_item_id"]),
                        "reason": str(exc),
                    }
                )
                frame_meta["s2_error"] = str(exc)

        if (not fused_done) and bool(request["s1_found"]) and request["s1_item_id"]:
            try:
                s1_item = fetch_item_by_id(
                    catalog,
                    collection=str(request["s1_collection"]),
                    item_id=str(request["s1_item_id"]),
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
                        dst_crs = sequence_dst_crs
                        dst_transform = sequence_dst_transform
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
                        "item_id": str(request["s1_item_id"]),
                        "reason": str(exc),
                    }
                )
                frame_meta["s1_error"] = str(exc)

        return {
            "s2_data": s2_data,
            "s1_data": s1_data,
            "s2_mask": s2_mask,
            "s1_mask": s1_mask,
            "s2_ok": bool(s2_ok),
            "s1_ok": bool(s1_ok),
            "day_of_year": int(request["day_of_year"]),
            "bin_index": bin_index,
            "frame_meta": frame_meta,
        }

    def finalize_sequence_state(state: dict[str, Any]) -> None:
        nonlocal written_sequences, dropped_sequences, processed_sequences

        ordered_bins = [entry for entry in state["bins"] if entry is not None]
        frame_mask = [bool(entry["s2_ok"] and entry["s1_ok"]) for entry in ordered_bins]
        paired_frame_count = int(sum(frame_mask))
        if paired_frame_count < int(args.min_materialized_paired_bins):
            dropped_sequences += 1
            processed_sequences += 1
            failure_rows.append(
                {
                    "sequence_id": state["sequence_id"],
                    "bin_index": -1,
                    "modality": "sequence",
                    "item_id": None,
                    "reason": (
                        f"paired_frame_count={paired_frame_count} below min_materialized_paired_bins="
                        f"{args.min_materialized_paired_bins}"
                    ),
                }
            )
            log_sequence_progress(
                sequence_id=state["sequence_id"],
                status="dropped_low_paired_frames",
                frame_count=len(ordered_bins),
                paired_frame_count=paired_frame_count,
                sequence_started_at=state["sequence_started_at"],
            )
            return

        s2_array = np.stack([entry["s2_data"] for entry in ordered_bins], axis=0).astype(np.float32)
        s1_array = np.stack([entry["s1_data"] for entry in ordered_bins], axis=0).astype(np.float32)
        frame_mask_array = np.asarray(frame_mask, dtype=bool)
        s2_frame_mask = [bool(entry["s2_ok"]) for entry in ordered_bins]
        s1_frame_mask = [bool(entry["s1_ok"]) for entry in ordered_bins]
        s2_frame_mask_array = np.asarray(s2_frame_mask, dtype=bool)
        s1_frame_mask_array = np.asarray(s1_frame_mask, dtype=bool)
        s2_valid_mask_array = np.stack([entry["s2_mask"] for entry in ordered_bins], axis=0).astype(bool)
        s1_valid_mask_array = np.stack([entry["s1_mask"] for entry in ordered_bins], axis=0).astype(bool)
        day_of_year = [int(entry["day_of_year"]) for entry in ordered_bins]
        bin_indices = [int(entry["bin_index"]) for entry in ordered_bins]
        frame_metadata = [entry["frame_meta"] for entry in ordered_bins]

        if args.output_format in {"npz", "both"}:
            np.savez_compressed(
                state["sequence_path"],
                s2=s2_array,
                s1=s1_array,
                frame_mask=frame_mask_array,
                s2_frame_mask=s2_frame_mask_array,
                s1_frame_mask=s1_frame_mask_array,
                s2_valid_mask=s2_valid_mask_array,
                s1_valid_mask=s1_valid_mask_array,
                day_of_year=np.asarray(day_of_year, dtype=np.int16),
                bin_index=np.asarray(bin_indices, dtype=np.int16),
            )

        if args.output_format in {"zarr", "both"}:
            zarr_buffer.append(
                {
                    "sequence_id": state["sequence_id"],
                    "sample_id": state["sample_id"],
                    "search_group_id": state["search_group_id"],
                    "anchor_year": state["anchor_year"],
                    "latitude": state["latitude"],
                    "longitude": state["longitude"],
                    "frame_count": int(len(frame_mask)),
                    "paired_frame_count": paired_frame_count,
                    "s2": s2_array,
                    "s1": s1_array,
                    "frame_mask": frame_mask_array,
                    "s2_frame_mask": s2_frame_mask_array,
                    "s1_frame_mask": s1_frame_mask_array,
                    "s2_valid_mask": s2_valid_mask_array,
                    "s1_valid_mask": s1_valid_mask_array,
                    "day_of_year": np.asarray(day_of_year, dtype=np.int16),
                    "bin_index": np.asarray(bin_indices, dtype=np.int16),
                    "frame_metadata": frame_metadata,
                }
            )
            if len(zarr_buffer) >= int(args.zarr_shard_size):
                flush_zarr_buffer()

        written_sequences += 1
        processed_sequences += 1
        index_rows.append(
            {
                "sequence_id": state["sequence_id"],
                "sample_id": state["sample_id"],
                "anchor_year": state["anchor_year"],
                "search_group_id": state["search_group_id"],
                "latitude": state["latitude"],
                "longitude": state["longitude"],
                "sequence_path": str(state["sequence_path"]) if args.output_format in {"npz", "both"} else None,
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
        log_sequence_progress(
            sequence_id=state["sequence_id"],
            status="written",
            frame_count=len(frame_mask),
            paired_frame_count=paired_frame_count,
            sequence_started_at=state["sequence_started_at"],
        )

    group_columns = ["search_group_id", "anchor_year"] if "search_group_id" in sequence_frame.columns else ["anchor_year"]
    grouped_sequence_iter = sequence_frame.groupby(group_columns, dropna=False, sort=True)

    for group_key, group_frame in grouped_sequence_iter:
        group_label = group_key if isinstance(group_key, tuple) else (group_key,)
        group_name = ", ".join(f"{column}={value}" for column, value in zip(group_columns, group_label))
        sequence_states: dict[str, dict[str, Any]] = {}
        frame_requests: list[dict[str, Any]] = []

        for sequence_row in group_frame.itertuples(index=False):
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
                processed_sequences += 1
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
                log_sequence_progress(
                    sequence_id=sequence_id,
                    status="reused_existing",
                    frame_count=len(sequence_plan),
                    paired_frame_count=int(sequence_plan["paired_found"].fillna(False).astype(bool).sum()),
                )
                continue

            if sequence_plan.empty:
                dropped_sequences += 1
                processed_sequences += 1
                failure_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "bin_index": -1,
                        "modality": "sequence",
                        "item_id": None,
                        "reason": "sequence_plan_empty_after_filters",
                    }
                )
                log_sequence_progress(sequence_id=sequence_id, status="dropped_empty_plan")
                continue

            sequence_states[sequence_id] = {
                "sequence_id": sequence_id,
                "sample_id": sample_id,
                "anchor_year": anchor_year,
                "search_group_id": getattr(sequence_row, "search_group_id", None),
                "latitude": latitude,
                "longitude": longitude,
                "sequence_path": sequence_path,
                "frame_count": int(len(sequence_plan)),
                "completed_bins": 0,
                "bins": [None] * int(len(sequence_plan)),
                "sequence_started_at": time.time(),
            }
            if str(args.pixel_access_mode) == "direct":
                sequence_dst_crs, sequence_dst_transform = fixed_chip_target_grid(
                    lon=longitude,
                    lat=latitude,
                    chip_size=args.chip_size,
                    resolution_m=float(args.cdse_resolution_m),
                )
                sequence_states[sequence_id]["sequence_dst_crs"] = sequence_dst_crs
                sequence_states[sequence_id]["sequence_dst_transform"] = sequence_dst_transform

            for bin_position, plan_row in enumerate(sequence_plan.itertuples(index=False)):
                plan_data = plan_row._asdict()
                frame_requests.append(
                    {
                        "sequence_id": sequence_id,
                        "bin_position": int(bin_position),
                        "bin_index": int(plan_data["bin_index"]),
                        "bin_start": plan_data["bin_start"],
                        "bin_end": plan_data["bin_end"],
                        "latitude": latitude,
                        "longitude": longitude,
                        "day_of_year": parse_iso_day_of_year(plan_data["s2_datetime"] or plan_data["s1_datetime"]),
                        "s2_found": bool_from_value(plan_data["s2_found"]),
                        "s1_found": bool_from_value(plan_data["s1_found"]),
                        "paired_found": bool_from_value(plan_data["paired_found"]),
                        "s2_item_id": plan_data["s2_item_id"],
                        "s2_collection": plan_data["s2_collection"],
                        "s2_datetime": plan_data.get("s2_datetime"),
                        "s2_cloud_cover": plan_data.get("s2_cloud_cover"),
                        "s2_candidate_items_json": plan_data.get("s2_candidate_items_json"),
                        "s1_item_id": plan_data["s1_item_id"],
                        "s1_collection": plan_data["s1_collection"],
                        "sequence_dst_crs": sequence_states[sequence_id].get("sequence_dst_crs"),
                        "sequence_dst_transform": sequence_states[sequence_id].get("sequence_dst_transform"),
                    }
                )

        if not frame_requests:
            continue

        if str(args.materialization_order) == "scene":
            frame_requests.sort(key=scene_sort_key)
        else:
            frame_requests.sort(key=lambda request: (str(request["sequence_id"]), int(request["bin_position"])))

        unique_s2_scenes = len(
            {
                (str(request["s2_collection"]), str(request["s2_item_id"]))
                for request in frame_requests
                if request["s2_found"] and request["s2_item_id"]
            }
        )
        unique_s1_scenes = len(
            {
                (str(request["s1_collection"]), str(request["s1_item_id"]))
                for request in frame_requests
                if request["s1_found"] and request["s1_item_id"]
            }
        )
        print(
            f"[dense-materialize] group_start {group_name} sequences={len(sequence_states)} "
            f"frame_requests={len(frame_requests)} unique_s2_scenes={unique_s2_scenes} "
            f"unique_s1_scenes={unique_s1_scenes} order={args.materialization_order}",
            flush=True,
        )

        for group_request_idx, request in enumerate(frame_requests, start=1):
            result = materialize_frame_request(request)
            processed_frame_requests += 1
            sequence_state = sequence_states[str(request["sequence_id"])]
            sequence_state["bins"][int(request["bin_position"])] = result
            sequence_state["completed_bins"] += 1

            if (
                group_request_idx == 1
                or int(args.log_every_frame_requests) <= 1
                or group_request_idx % int(args.log_every_frame_requests) == 0
                or group_request_idx == len(frame_requests)
            ):
                print(
                    f"[dense-materialize] group_progress {group_name} frame_requests="
                    f"{group_request_idx}/{len(frame_requests)} failures={len(failure_rows)} "
                    f"elapsed={format_duration(time.time() - materialization_started_at)}",
                    flush=True,
                )

            if int(sequence_state["completed_bins"]) >= int(sequence_state["frame_count"]):
                finalize_sequence_state(sequence_state)
                del sequence_states[str(request["sequence_id"])]

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
            "dropped_sequences": int(dropped_sequences),
            "processed_sequences": int(processed_sequences),
            "processed_frame_requests": int(processed_frame_requests),
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
