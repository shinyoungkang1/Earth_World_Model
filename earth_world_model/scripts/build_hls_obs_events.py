#!/usr/bin/env python3
"""Build an `obs_events` parquet from the existing HLS chip index."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_BAND_SCHEMA = "hls6"
DEFAULT_SENSOR_ID = "hls"
DEFAULT_SOURCE_DATASET = "hls_chip_index_v1"
DEFAULT_SPATIAL_RESOLUTION_M = 30.0


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pandas is required to build HLS obs_events manifests.") from exc
    return pd


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("numpy is required to build HLS obs_events manifests.") from exc
    return np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an HLS chip index parquet into an obs_events parquet without duplicating chip tensors.",
    )
    parser.add_argument("--hls-index-path", required=True, help="Input HLS chip index parquet or CSV.")
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--chip-path-column", default="chip_path")
    parser.add_argument("--frame-count-column", default="frame_count")
    parser.add_argument("--valid-fraction-column", default="valid_fraction")
    parser.add_argument("--frame-datetimes-column", default="frame_datetimes_json")
    parser.add_argument("--frame-collections-column", default="frame_collections_json")
    parser.add_argument("--frame-item-ids-column", default="frame_item_ids_json")
    parser.add_argument("--frame-clear-fractions-column", default="frame_clear_fractions_json")
    parser.add_argument("--sample-key-column", default="sample_key")
    parser.add_argument("--sample-type-column", default="sample_type")
    parser.add_argument("--latitude-column", default="sample_latitude")
    parser.add_argument("--longitude-column", default="sample_longitude")
    parser.add_argument("--anchor-year-column", default="anchor_year")
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--band-schema", default=DEFAULT_BAND_SCHEMA)
    parser.add_argument("--spatial-resolution-m", type=float, default=DEFAULT_SPATIAL_RESOLUTION_M)
    parser.add_argument(
        "--resolve-missing-datetimes-from-chip",
        action="store_true",
        help="If frame datetimes are missing in the index, open the chip NPZ and derive timestamps from temporal_coords.",
    )
    parser.add_argument(
        "--require-existing-chip-paths",
        action="store_true",
        help="Drop rows whose chip_path does not exist on disk.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Drop events with quality_score below this threshold.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional row cap before event expansion.")
    return parser.parse_args()


def _read_table(path: Path):
    pd = require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported HLS index format for {path}. Expected .csv or .parquet")


def _json_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(value, list):
            return value
        return []
    try:
        pd = require_pandas()
        if pd.isna(raw):
            return []
    except SystemExit:
        pass
    return []


def _timestamp_to_iso(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return parsed.isoformat()


def _year_doy_to_iso(year_value: Any, doy_value: Any) -> str | None:
    try:
        year = int(round(float(year_value)))
        day_of_year = int(round(float(doy_value)))
    except (TypeError, ValueError):
        return None
    if year <= 0 or day_of_year <= 0:
        return None
    try:
        resolved = date(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return None
    return datetime.combine(resolved, datetime.min.time()).isoformat()


def _load_temporal_coord_datetimes(chip_path: Path) -> list[str]:
    np = require_numpy()
    raw_datetimes: list[str] = []
    with np.load(chip_path, allow_pickle=False) as payload:
        if "temporal_coords" not in payload.files:
            return raw_datetimes
        temporal_coords = payload["temporal_coords"]
        if temporal_coords.ndim != 2 or temporal_coords.shape[1] < 2:
            return raw_datetimes
        for row in temporal_coords:
            iso_value = _year_doy_to_iso(row[0], row[1])
            if iso_value is not None:
                raw_datetimes.append(iso_value)
    return raw_datetimes


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def _resolve_frame_count(row: dict[str, Any], *, frame_count_column: str, list_lengths: list[int]) -> int:
    frame_count = _safe_float(row.get(frame_count_column))
    if frame_count is not None and frame_count > 0:
        return int(frame_count)
    inferred = max(list_lengths) if list_lengths else 0
    if inferred <= 0:
        return 0
    return int(inferred)


def build_events(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pd = require_pandas()
    index_path = Path(args.hls_index_path)
    frame = _read_table(index_path).copy()
    required = [args.sample_id_column, args.chip_path_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"HLS index missing required columns {missing}: {index_path}")
    if args.max_samples > 0:
        frame = frame.iloc[: int(args.max_samples)].copy()
    if frame.empty:
        raise SystemExit(f"No HLS chip rows found in {index_path}")

    rows = frame.to_dict(orient="records")
    events: list[dict[str, Any]] = []
    skipped_missing_chip = 0
    skipped_missing_timestamp = 0
    filtered_low_quality = 0
    used_chip_temporal_coords = 0

    optional_passthrough = []
    for source_column, target_column in (
        (args.sample_key_column, "sample_key"),
        (args.sample_type_column, "sample_type"),
        (args.latitude_column, "latitude"),
        (args.longitude_column, "longitude"),
        (args.anchor_year_column, "anchor_year"),
    ):
        if source_column and source_column in frame.columns:
            optional_passthrough.append((source_column, target_column))

    for row in rows:
        sample_id = str(row.get(args.sample_id_column, "")).strip()
        if not sample_id:
            continue
        chip_path = Path(str(row.get(args.chip_path_column, "")).strip())
        if args.require_existing_chip_paths and not chip_path.exists():
            skipped_missing_chip += 1
            continue

        frame_datetimes = [_timestamp_to_iso(value) for value in _json_list(row.get(args.frame_datetimes_column))]
        if args.resolve_missing_datetimes_from_chip and (not frame_datetimes or any(value is None for value in frame_datetimes)):
            if chip_path.exists():
                resolved = _load_temporal_coord_datetimes(chip_path)
                if resolved:
                    frame_datetimes = resolved
                    used_chip_temporal_coords += 1
        frame_collections = [str(value) for value in _json_list(row.get(args.frame_collections_column))]
        frame_item_ids = [str(value) for value in _json_list(row.get(args.frame_item_ids_column))]
        frame_clear_fractions = [_safe_float(value) for value in _json_list(row.get(args.frame_clear_fractions_column))]
        valid_fraction = _safe_float(row.get(args.valid_fraction_column))
        frame_count = _resolve_frame_count(
            row,
            frame_count_column=args.frame_count_column,
            list_lengths=[
                len(frame_datetimes),
                len(frame_collections),
                len(frame_item_ids),
                len(frame_clear_fractions),
            ],
        )
        if frame_count <= 0:
            continue

        for frame_index in range(frame_count):
            timestamp = frame_datetimes[frame_index] if frame_index < len(frame_datetimes) else None
            if timestamp is None:
                skipped_missing_timestamp += 1
                continue
            clear_fraction = frame_clear_fractions[frame_index] if frame_index < len(frame_clear_fractions) else None
            quality_score = clear_fraction if clear_fraction is not None else valid_fraction
            if quality_score is None:
                quality_score = 0.0
            if quality_score < float(args.min_quality_score):
                filtered_low_quality += 1
                continue
            event = {
                "event_id": f"{sample_id}:{args.sensor_id}:{frame_index:02d}",
                "sample_id": sample_id,
                "sensor_id": str(args.sensor_id),
                "timestamp": timestamp,
                "date": timestamp[:10],
                "tensor_path": str(chip_path),
                "tensor_container": "npz",
                "frame_index": int(frame_index),
                "mask_path": "",
                "mask_metadata_kind": "embedded_valid_mask",
                "band_schema": str(args.band_schema),
                "spatial_resolution_m": float(args.spatial_resolution_m),
                "quality_score": float(quality_score),
                "valid_fraction": valid_fraction,
                "clear_fraction": clear_fraction,
                "source_dataset": str(args.source_dataset),
                "source_collection": frame_collections[frame_index] if frame_index < len(frame_collections) else None,
                "source_item_id": frame_item_ids[frame_index] if frame_index < len(frame_item_ids) else None,
                "frame_count": int(frame_count),
            }
            for source_column, target_column in optional_passthrough:
                event[target_column] = row.get(source_column)
            events.append(event)

    metadata = {
        "dataset": "hls_obs_events_v1",
        "source_index_path": str(index_path),
        "source_dataset": str(args.source_dataset),
        "sensor_id": str(args.sensor_id),
        "band_schema": str(args.band_schema),
        "spatial_resolution_m": float(args.spatial_resolution_m),
        "row_count_input": int(len(frame)),
        "event_count_output": int(len(events)),
        "skipped_missing_chip": int(skipped_missing_chip),
        "skipped_missing_timestamp": int(skipped_missing_timestamp),
        "filtered_low_quality": int(filtered_low_quality),
        "resolved_missing_datetimes_from_chip_rows": int(used_chip_temporal_coords),
        "min_quality_score": float(args.min_quality_score),
        "require_existing_chip_paths": bool(args.require_existing_chip_paths),
        "resolve_missing_datetimes_from_chip": bool(args.resolve_missing_datetimes_from_chip),
        "sample_count_output": int(pd.Series([event["sample_id"] for event in events]).nunique()) if events else 0,
        "timestamp_min": min((event["timestamp"] for event in events), default=None),
        "timestamp_max": max((event["timestamp"] for event in events), default=None),
    }
    return events, metadata


def main() -> None:
    args = parse_args()
    pd = require_pandas()
    events, metadata = build_events(args)
    output_parquet = Path(args.output_parquet)
    output_metadata_json = Path(args.output_metadata_json)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(events).sort_values(["sample_id", "timestamp", "frame_index"]).reset_index(drop=True)
    frame.to_parquet(output_parquet, index=False)
    output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
