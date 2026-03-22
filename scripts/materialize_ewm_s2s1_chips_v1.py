#!/usr/bin/env python3
"""Materialize basin-scoped Sentinel-2 + Sentinel-1 chips for frozen EWM gas runs.

This is the data-prep step that Phase 4 was missing.

Outputs:

- `data/features/<basin_id>/ewm_s2s1_chips_v1/*.npz`
- `data/features/<basin_id>/ewm_s2s1_chip_index_v1.parquet`
- `data/features/<basin_id>/ewm_s2s1_chip_index_v1_metadata.json`
- `data/features/<basin_id>/ewm_s2s1_selection_manifest_v1.json`

The emitted `.npz` files match the `s2s1_npz_manifest` mode expected by
`scripts/run_ewm_gas_pipeline_v1.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pyproj import Transformer
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import reproject


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dataset import SSL4EO_S1GRD_MEAN, SSL4EO_S2L2A_MEAN  # noqa: E402


CHIP_SIZE = 128
CHIP_RADIUS_PX = CHIP_SIZE // 2
S2_COLLECTION = "sentinel-2-l2a"
S1_COLLECTION = "sentinel-1-rtc"
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["vv", "vh"]
S2_INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2))
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".parquet",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def write_npz_atomic(
    path: Path,
    *,
    s2: np.ndarray,
    s1: np.ndarray,
    dates: np.ndarray,
    mask: np.ndarray,
    location_coords: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz.tmp",
        delete=False,
    ) as handle:
        np.savez_compressed(
            handle,
            s2=s2,
            s1=s1,
            dates=dates,
            mask=mask,
            location_coords=location_coords,
        )
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def load_basin_config(repo_root: Path, basin_id: str) -> dict[str, Any]:
    return load_json(repo_root / "config" / "basins" / f"{basin_id}.json")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table extension for {path}")


def sanitize_id(value: str) -> str:
    return value.replace("::", "__").replace("/", "_").replace(":", "_")


def shard_tag(shard_index: int, shard_count: int) -> str:
    if shard_count <= 1:
        return ""
    width = max(2, len(str(shard_count - 1)))
    return f"__shard_{shard_index:0{width}d}_of_{shard_count:0{width}d}"


def search_group_columns(samples: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    if "region_id" in samples.columns:
        columns.append("region_id")
    columns.append("anchor_year")
    return columns


def apply_sample_shard(
    samples: pd.DataFrame,
    shard_index: int,
    shard_count: int,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    if shard_count <= 1:
        return samples.copy().reset_index(drop=True)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")

    if not group_columns:
        ordered = samples.sort_values("sample_id").reset_index(drop=True).copy()
        shard_rows = ordered.iloc[shard_index::shard_count].copy().reset_index(drop=True)
        return shard_rows

    group_sizes = (
        samples.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["row_count", *group_columns], ascending=[False, *([True] * len(group_columns))])
        .reset_index(drop=True)
    )
    shard_loads = [0 for _ in range(shard_count)]
    shard_assignments: dict[tuple[Any, ...], int] = {}
    for row in group_sizes.itertuples(index=False):
        group_key = tuple(getattr(row, column) for column in group_columns)
        target_shard = min(range(shard_count), key=lambda idx: (shard_loads[idx], idx))
        shard_assignments[group_key] = target_shard
        shard_loads[target_shard] += int(getattr(row, "row_count"))

    assigned_keys = {group_key for group_key, assigned_shard in shard_assignments.items() if assigned_shard == shard_index}
    key_frame = samples[group_columns].copy()
    row_keys = list(key_frame.itertuples(index=False, name=None))
    mask = [group_key in assigned_keys for group_key in row_keys]
    shard_rows = samples.loc[mask].sort_values(group_columns + ["sample_id"]).reset_index(drop=True).copy()
    return shard_rows


def point_in_bbox(lon: float, lat: float, bbox: list[float]) -> bool:
    minx, miny, maxx, maxy = bbox
    return minx <= lon <= maxx and miny <= lat <= maxy


def year_doy_from_timestamp(ts: pd.Timestamp) -> tuple[int, int]:
    stamp = pd.Timestamp(ts)
    return int(stamp.year), int(stamp.dayofyear)


def choose_temporal_split(
    frame: pd.DataFrame,
    date_col: str,
    label_col: str,
    preferred_quantile: float,
) -> tuple[float, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    candidates = [preferred_quantile]
    for delta in [0.05, 0.1, 0.15, 0.2]:
        for direction in [-1, 1]:
            quantile = round(preferred_quantile + (delta * direction), 4)
            if 0.5 <= quantile < 0.98:
                candidates.append(quantile)
    ordered: list[float] = []
    seen: set[float] = set()
    for quantile in candidates:
        if quantile not in seen:
            seen.add(quantile)
            ordered.append(quantile)

    data = frame.sort_values(date_col).copy()
    for quantile in ordered:
        split_date = data[date_col].quantile(quantile)
        train = data[data[date_col] <= split_date].copy()
        test = data[data[date_col] > split_date].copy()
        if train.empty or test.empty:
            continue
        if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
            continue
        return quantile, split_date, train, test

    split_date = data[date_col].quantile(preferred_quantile)
    train = data[data[date_col] <= split_date].copy()
    test = data[data[date_col] > split_date].copy()
    return preferred_quantile, split_date, train, test


def stratified_binary_downsample(
    frame: pd.DataFrame,
    label_col: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    positives = frame[frame[label_col] == True].copy()
    negatives = frame[frame[label_col] == False].copy()
    if positives.empty or negatives.empty:
        return frame.sample(n=max_rows, random_state=random_state).copy()

    target_each = max_rows // 2
    pos_n = min(len(positives), target_each)
    neg_n = min(len(negatives), target_each)
    remainder = max_rows - pos_n - neg_n
    if remainder > 0:
        if len(positives) - pos_n > len(negatives) - neg_n:
            pos_n = min(len(positives), pos_n + remainder)
        else:
            neg_n = min(len(negatives), neg_n + remainder)

    sampled = pd.concat(
        [
            positives.sample(n=pos_n, random_state=random_state),
            negatives.sample(n=neg_n, random_state=random_state),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_windows(year: int) -> list[tuple[str, str, str]]:
    return [
        ("early_spring", f"{year}-03-01", f"{year}-04-30"),
        ("late_spring", f"{year}-05-01", f"{year}-06-30"),
        ("summer", f"{year}-07-01", f"{year}-08-31"),
        ("fall", f"{year}-09-01", f"{year}-10-31"),
    ]


def resolve_output_dir(
    repo_root: Path,
    explicit_output_dir: str | None,
    cohort_path: Path | None,
    basin_id: str,
) -> Path:
    if explicit_output_dir:
        path = Path(explicit_output_dir)
        return path if path.is_absolute() else (repo_root / path)
    if cohort_path is not None:
        return cohort_path.parent / f"{cohort_path.stem}_ewm_s2s1"
    return repo_root / "data" / "features" / basin_id


def compute_anchor_years(
    frame: pd.DataFrame,
    imagery_anchor_mode: str,
    sample_year: int,
    anchor_date_column: str,
    anchor_year_column: str,
    anchor_year_offset: int,
) -> pd.Series:
    if imagery_anchor_mode == "fixed_year":
        return pd.Series(sample_year, index=frame.index, dtype="Int64")
    if imagery_anchor_mode == "column":
        return pd.to_numeric(frame[anchor_year_column], errors="coerce").astype("Int64")
    anchor_dates = parse_date(frame[anchor_date_column])
    return (anchor_dates.dt.year + anchor_year_offset).astype("Int64")


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    cloud_cover = candidate["cloud_cover"] if candidate["cloud_cover"] is not None else 999.0
    return (-candidate["coverage_count"], cloud_cover, -pd.Timestamp(candidate["datetime"]).value)


def has_required_assets(item, asset_keys: list[str]) -> bool:
    return all(key in item.assets and item.assets[key].href for key in asset_keys)


def search_candidates(
    catalog: Client,
    collection: str,
    bbox: list[float],
    sample_points: pd.DataFrame,
    windows: list[tuple[str, str, str]],
    max_items: int,
    cloud_cover_max: float | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_rows: list[dict[str, Any]] = []
    required_assets = S2_BANDS + ["SCL"] if collection == S2_COLLECTION else S1_BANDS

    for window_id, start_date, end_date in windows:
        search_kwargs: dict[str, Any] = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "max_items": max_items,
        }
        if cloud_cover_max is not None:
            search_kwargs["query"] = {"eo:cloud_cover": {"lt": cloud_cover_max}}

        candidates: list[dict[str, Any]] = []
        for item in search_items_with_backoff(catalog, search_kwargs):
            if not has_required_assets(item, required_assets):
                continue
            item_bbox = list(item.bbox) if item.bbox else None
            if not item_bbox:
                continue
            coverage_count = int(
                sample_points.apply(
                    lambda row: point_in_bbox(
                        float(row["sample_longitude"]),
                        float(row["sample_latitude"]),
                        item_bbox,
                    ),
                    axis=1,
                ).sum()
            )
            if coverage_count <= 0:
                continue
            candidate = {
                "window_id": window_id,
                "collection": collection,
                "item_id": item.id,
                "datetime": pd.Timestamp(item.datetime).isoformat() if item.datetime is not None else None,
                "cloud_cover": (
                    float(item.properties.get("eo:cloud_cover"))
                    if item.properties.get("eo:cloud_cover") is not None
                    else None
                ),
                "coverage_count": coverage_count,
                "bbox": item_bbox,
                "item": planetary_computer.sign(item),
            }
            candidates.append(candidate)

        candidates.sort(key=candidate_sort_key)
        selected[window_id] = candidates
        for candidate in candidates:
            manifest_rows.append(
                {
                    "window_id": candidate["window_id"],
                    "collection": candidate["collection"],
                    "item_id": candidate["item_id"],
                    "datetime": candidate["datetime"],
                    "cloud_cover": candidate["cloud_cover"],
                    "coverage_count": candidate["coverage_count"],
                    "bbox": candidate["bbox"],
                }
            )

    return selected, manifest_rows


def search_items_with_backoff(
    catalog: Client,
    search_kwargs: dict[str, Any],
    *,
    max_attempts: int = 6,
    base_delay_seconds: float = 5.0,
) -> list[Any]:
    for attempt in range(max_attempts):
        try:
            return list(catalog.search(**search_kwargs).items())
        except APIError as exc:
            message = str(exc).lower()
            if "rate limit" not in message and "429" not in message:
                raise
            if attempt >= max_attempts - 1:
                raise
            delay = base_delay_seconds * (2**attempt) + random.uniform(0.0, 1.0)
            print(
                f"STAC rate limited for collections={search_kwargs.get('collections')} "
                f"datetime={search_kwargs.get('datetime')}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    return []


def pick_best_candidate(
    candidates_by_window: dict[str, list[dict[str, Any]]],
    window_id: str,
    lon: float,
    lat: float,
) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in candidates_by_window.get(window_id, [])
        if candidate["bbox"] and point_in_bbox(lon, lat, candidate["bbox"])
    ]
    if not matches:
        return None
    return sorted(matches, key=candidate_sort_key)[0]


def reference_window_and_transform(asset_href: str, lon: float, lat: float) -> tuple[Any, Window, Affine]:
    with rasterio.open(asset_href) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = transformer.transform(float(lon), float(lat))
        row, col = ds.index(x, y)
        window = Window(col - CHIP_RADIUS_PX, row - CHIP_RADIUS_PX, CHIP_SIZE, CHIP_SIZE)
        return ds.crs, window, ds.window_transform(window)


def reproject_asset_to_grid(
    asset_href: str,
    dst_crs,
    dst_transform: Affine,
    width: int,
    height: int,
    resampling: Resampling,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    destination = np.full((height, width), dst_nodata, dtype=np.float32)
    with rasterio.open(asset_href) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return destination


def read_s2_window(
    item: dict[str, Any],
    lon: float,
    lat: float,
    s2_fill_values: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any], Any, Affine]:
    signed_item = item["item"]
    dst_crs, window, dst_transform = reference_window_and_transform(signed_item.assets["B02"].href, lon, lat)

    scl = reproject_asset_to_grid(
        signed_item.assets["SCL"].href,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        width=CHIP_SIZE,
        height=CHIP_SIZE,
        resampling=Resampling.nearest,
    )
    scl_int = np.where(np.isfinite(scl), np.rint(scl), -9999).astype(np.int16)
    clear_mask = np.isfinite(scl) & ~np.isin(scl_int, list(S2_INVALID_SCL))
    clear_fraction = float(clear_mask.mean())

    bands: list[np.ndarray] = []
    for band_idx, band_key in enumerate(S2_BANDS):
        band = reproject_asset_to_grid(
            signed_item.assets[band_key].href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            resampling=Resampling.bilinear,
        )
        valid_mask = np.isfinite(band) & clear_mask
        filled = np.where(valid_mask, band, s2_fill_values[band_idx]).astype(np.float32)
        bands.append(filled)

    metadata = {
        "s2_item_id": item["item_id"],
        "s2_datetime": item["datetime"],
        "s2_cloud_cover": item["cloud_cover"],
        "s2_clear_fraction": clear_fraction,
    }
    return np.stack(bands, axis=0), clear_fraction, metadata, dst_crs, dst_transform


def read_s1_window(
    item: dict[str, Any],
    dst_crs,
    dst_transform: Affine,
    s1_fill_values: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    signed_item = item["item"]
    bands: list[np.ndarray] = []
    valid_mask_all: np.ndarray | None = None

    for band_idx, band_key in enumerate(S1_BANDS):
        band = reproject_asset_to_grid(
            signed_item.assets[band_key].href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            resampling=Resampling.bilinear,
        )
        valid_mask = np.isfinite(band)
        valid_mask_all = valid_mask if valid_mask_all is None else (valid_mask_all & valid_mask)
        filled = np.where(valid_mask, band, s1_fill_values[band_idx]).astype(np.float32)
        bands.append(filled)

    valid_fraction = float(valid_mask_all.mean()) if valid_mask_all is not None else 0.0
    metadata = {
        "s1_item_id": item["item_id"],
        "s1_datetime": item["datetime"],
        "s1_valid_fraction": valid_fraction,
    }
    return np.stack(bands, axis=0), valid_fraction, metadata


def build_existing_index_row(
    sample_id: str,
    sample_type: str,
    sample_key: str,
    lon: float,
    lat: float,
    chip_path: Path,
    anchor_year: int,
    windows: list[tuple[str, str, str]],
) -> dict[str, Any]:
    with np.load(chip_path, allow_pickle=False) as payload:
        dates_arr = np.asarray(payload["dates"], dtype=np.int32) if "dates" in payload.files else np.asarray([], dtype=np.int32)
        mask_arr = np.asarray(payload["mask"], dtype=bool) if "mask" in payload.files else np.asarray([], dtype=bool)
        s2 = np.asarray(payload["s2"], dtype=np.float32)

    frame_count = int(s2.shape[0]) if s2.ndim >= 1 else int(len(dates_arr))
    if mask_arr.size == 0:
        mask_arr = np.ones(frame_count, dtype=bool)

    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "sample_key": sample_key,
        "sample_longitude": lon,
        "sample_latitude": lat,
        "sample_path": str(chip_path),
        "frame_count": frame_count,
        "valid_frame_count": int(mask_arr.sum()),
        "frame_mask_json": json.dumps(mask_arr.astype(int).tolist()),
        "dates_json": json.dumps(dates_arr.tolist()),
        # Existing chips are reused as-is; detailed source item metadata is not reconstructed here.
        "frame_metadata_json": json.dumps([]),
        "anchor_year": int(anchor_year),
        "window_ids_json": json.dumps([window_id for window_id, _start, _end in windows]),
        "window_start_dates_json": json.dumps([start_date for _window_id, start_date, _end in windows]),
        "window_end_dates_json": json.dumps([end_date for _window_id, _start, end_date in windows]),
    }


def group_has_all_existing_chips(samples: pd.DataFrame, chips_dir: Path) -> bool:
    if samples.empty:
        return False
    for sample_id in samples["sample_id"].tolist():
        chip_path = chips_dir / f"{sanitize_id(str(sample_id))}.npz"
        if not chip_path.exists():
            return False
    return True


def materialize_s2s1_chips(
    samples: pd.DataFrame,
    s2_candidates: dict[str, list[dict[str, Any]]],
    s1_candidates: dict[str, list[dict[str, Any]]],
    windows: list[tuple[str, str, str]],
    chips_dir: Path,
    min_s2_clear_fraction: float,
    min_s1_valid_fraction: float,
    anchor_year: int,
    skip_existing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    chips_dir.mkdir(parents=True, exist_ok=True)
    s2_fill_values = SSL4EO_S2L2A_MEAN.view(-1).cpu().numpy().astype(np.float32)
    s1_fill_values = SSL4EO_S1GRD_MEAN.view(-1).cpu().numpy().astype(np.float32)

    index_columns = [
        "sample_id",
        "sample_type",
        "sample_key",
        "sample_longitude",
        "sample_latitude",
        "sample_path",
        "frame_count",
        "valid_frame_count",
        "frame_mask_json",
        "dates_json",
        "frame_metadata_json",
        "anchor_year",
        "window_ids_json",
        "window_start_dates_json",
        "window_end_dates_json",
    ]
    rows: list[dict[str, Any]] = []
    failures = 0
    reused_existing = 0
    attempts = 0

    for row in samples[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]].itertuples(index=False):
        attempts += 1
        sample_id = str(row.sample_id)
        lon = float(row.sample_longitude)
        lat = float(row.sample_latitude)
        chip_path = chips_dir / f"{sanitize_id(sample_id)}.npz"

        if skip_existing and chip_path.exists():
            try:
                rows.append(
                    build_existing_index_row(
                        sample_id=sample_id,
                        sample_type=str(row.sample_type),
                        sample_key=str(row.sample_key),
                        lon=lon,
                        lat=lat,
                        chip_path=chip_path,
                        anchor_year=anchor_year,
                        windows=windows,
                    )
                )
                reused_existing += 1
                if attempts == 1 or attempts % 10 == 0:
                    print(
                        f"[materialize] anchor_year={anchor_year} sample={attempts}/{len(samples)} "
                        f"reused_existing={reused_existing} successes={len(rows)} failures={failures}",
                        flush=True,
                    )
                continue
            except Exception:
                # Fall back to recomputing if the cached chip is unreadable.
                pass

        s2_frames: list[np.ndarray] = []
        s1_frames: list[np.ndarray] = []
        dates: list[int] = []
        frame_mask: list[bool] = []
        frame_metadata: list[dict[str, Any]] = []

        success = True
        for window_id, _start, _end in windows:
            s2_item = pick_best_candidate(s2_candidates, window_id, lon, lat)
            s1_item = pick_best_candidate(s1_candidates, window_id, lon, lat)
            if s2_item is None or s1_item is None:
                success = False
                break

            try:
                s2_frame, s2_clear_fraction, s2_meta, dst_crs, dst_transform = read_s2_window(
                    s2_item,
                    lon,
                    lat,
                    s2_fill_values=s2_fill_values,
                )
                s1_frame, s1_valid_fraction, s1_meta = read_s1_window(
                    s1_item,
                    dst_crs=dst_crs,
                    dst_transform=dst_transform,
                    s1_fill_values=s1_fill_values,
                )
            except Exception:
                success = False
                break

            s2_frames.append(s2_frame)
            s1_frames.append(s1_frame)
            dates.append(year_doy_from_timestamp(pd.Timestamp(s2_item["datetime"]))[1])
            frame_mask.append(
                bool(s2_clear_fraction >= min_s2_clear_fraction and s1_valid_fraction >= min_s1_valid_fraction)
            )
            frame_metadata.append({"window_id": window_id, **s2_meta, **s1_meta})

        if not success or len(s2_frames) != len(windows) or len(s1_frames) != len(windows):
            failures += 1
            if attempts == 1 or attempts % 10 == 0:
                print(
                    f"[materialize] anchor_year={anchor_year} sample={attempts}/{len(samples)} "
                    f"successes={len(rows)} failures={failures}",
                    flush=True,
                )
            continue

        s2 = np.stack(s2_frames, axis=0).astype(np.float32)
        s1 = np.stack(s1_frames, axis=0).astype(np.float32)
        dates_arr = np.asarray(dates, dtype=np.int32)
        mask_arr = np.asarray(frame_mask, dtype=bool)
        write_npz_atomic(
            chip_path,
            s2=s2,
            s1=s1,
            dates=dates_arr,
            mask=mask_arr,
            location_coords=np.asarray([lat, lon], dtype=np.float32),
        )

        rows.append(
            {
                "sample_id": sample_id,
                "sample_type": row.sample_type,
                "sample_key": row.sample_key,
                "sample_longitude": lon,
                "sample_latitude": lat,
                "sample_path": str(chip_path),
                "frame_count": len(windows),
                "valid_frame_count": int(mask_arr.sum()),
                "frame_mask_json": json.dumps(mask_arr.astype(int).tolist()),
                "dates_json": json.dumps(dates_arr.tolist()),
                "frame_metadata_json": json.dumps(frame_metadata),
                "anchor_year": int(anchor_year),
                "window_ids_json": json.dumps([window_id for window_id, _start, _end in windows]),
                "window_start_dates_json": json.dumps([start_date for _window_id, start_date, _end in windows]),
                "window_end_dates_json": json.dumps([end_date for _window_id, _start, end_date in windows]),
            }
        )
        if attempts == 1 or attempts % 10 == 0:
            print(
                f"[materialize] anchor_year={anchor_year} sample={attempts}/{len(samples)} "
                f"successes={len(rows)} failures={failures}",
                flush=True,
            )

    if rows:
        index = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    else:
        index = pd.DataFrame(columns=index_columns)
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "ewm_s2s1_chip_index_v1",
        "row_count": int(len(index)),
        "failed_sample_count": int(failures),
        "frame_count": int(len(windows)),
        "chip_size_px": CHIP_SIZE,
        "modalities": ["Sentinel-2 L2A", "Sentinel-1 RTC"],
        "s2_band_order": S2_BANDS,
        "s1_band_order": S1_BANDS,
        "reused_existing_chip_count": int(reused_existing),
        "min_s2_clear_fraction": float(min_s2_clear_fraction),
        "min_s1_valid_fraction": float(min_s1_valid_fraction),
        "sample_type_counts": (
            {key: int(value) for key, value in index["sample_type"].value_counts().items()}
            if not index.empty
            else {}
        ),
    }
    return index, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sentinel-2 + Sentinel-1 chip inputs for the frozen EWM gas pipeline.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--cohort-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--max-training-wells", type=int, default=1600)
    parser.add_argument("--max-grid-cells", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-year", type=int, default=2024)
    parser.add_argument("--imagery-anchor-mode", choices=["fixed_year", "year_from_date_column", "column"], default="year_from_date_column")
    parser.add_argument("--anchor-date-column", default="first_prod_date")
    parser.add_argument("--anchor-year-column", default="anchor_year")
    parser.add_argument("--anchor-year-offset", type=int, default=-1)
    parser.add_argument("--s2-cloud-cover-max", type=float, default=35.0)
    parser.add_argument("--max-items-per-window", type=int, default=200)
    parser.add_argument("--min-s2-clear-fraction", type=float, default=0.5)
    parser.add_argument("--min-s1-valid-fraction", type=float, default=0.9)
    parser.add_argument("--stac-api-url", default="https://planetarycomputer.microsoft.com/api/stac/v1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cohort_path = Path(args.cohort_path).resolve() if args.cohort_path else None
    output_dir = resolve_output_dir(repo_root, args.output_dir, cohort_path, args.basin_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    scope_id = args.basin_id
    row_count_training_wells_selected = None
    if cohort_path is not None:
        scope_id = cohort_path.stem
        cohort = read_table(cohort_path)
        required_columns = {
            "sample_id",
            "sample_type",
            "sample_key",
            "sample_longitude",
            "sample_latitude",
        }
        missing_columns = sorted(required_columns - set(cohort.columns))
        if missing_columns:
            raise SystemExit(f"Cohort is missing required columns: {missing_columns}")
        combined_samples = cohort.copy()
        if args.skip_grid:
            combined_samples = combined_samples[combined_samples["sample_type"] != "grid_cell"].copy()
        combined_samples["sample_longitude"] = pd.to_numeric(combined_samples["sample_longitude"], errors="coerce")
        combined_samples["sample_latitude"] = pd.to_numeric(combined_samples["sample_latitude"], errors="coerce")
        combined_samples["anchor_year"] = compute_anchor_years(
            combined_samples,
            imagery_anchor_mode=args.imagery_anchor_mode,
            sample_year=args.sample_year,
            anchor_date_column=args.anchor_date_column,
            anchor_year_column=args.anchor_year_column,
            anchor_year_offset=args.anchor_year_offset,
        )
    else:
        basin = load_basin_config(repo_root, args.basin_id)
        features_dir = repo_root / "data" / "features" / args.basin_id
        derived_dir = repo_root / "data" / "derived" / args.basin_id
        features_dir.mkdir(parents=True, exist_ok=True)

        training_table = pd.read_csv(features_dir / "gas_training_table_v2.csv", low_memory=False)
        if args.label_column not in training_table.columns:
            raise SystemExit(f"Unknown label column: {args.label_column}")

        training_table["first_prod_date"] = parse_date(training_table["first_prod_date"])
        wells = training_table.copy()
        wells["sample_id"] = wells["well_api"].map(lambda value: f"well::{value}")
        wells["sample_type"] = "well"
        wells["sample_key"] = wells["well_api"]
        wells["sample_longitude"] = wells["longitude_decimal"]
        wells["sample_latitude"] = wells["latitude_decimal"]

        wells_available = wells[
            wells["label_f12_available"].eq(True)
            & wells[args.label_column].notna()
            & wells["first_prod_date"].notna()
            & wells["sample_latitude"].notna()
            & wells["sample_longitude"].notna()
        ].copy()
        wells_available[args.label_column] = wells_available[args.label_column].astype(bool)

        wells_selected = stratified_binary_downsample(
            wells_available,
            args.label_column,
            max_rows=args.max_training_wells,
            random_state=args.random_state,
        )
        wells_selected = wells_selected.sort_values("first_prod_date").reset_index(drop=True)
        row_count_training_wells_selected = int(len(wells_selected))

        combined_samples = wells_selected[
            ["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude", "first_prod_date"]
        ].copy()

        if not args.skip_grid:
            grid_features_path = derived_dir / "gas_prospect_cells_v1.csv"
            if grid_features_path.exists():
                grid = pd.read_csv(grid_features_path)
                grid["sample_id"] = grid["cell_id"].map(lambda value: f"cell::{value}")
                grid["sample_type"] = "grid_cell"
                grid["sample_key"] = grid["cell_id"]
                grid["sample_longitude"] = grid["center_longitude"]
                grid["sample_latitude"] = grid["center_latitude"]
                if args.max_grid_cells and args.max_grid_cells > 0 and len(grid) > args.max_grid_cells:
                    grid = grid.head(args.max_grid_cells).copy()
                combined_samples = pd.concat(
                    [
                        combined_samples,
                        grid[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]],
                    ],
                    ignore_index=True,
                ).drop_duplicates("sample_id")

        combined_samples["anchor_year"] = compute_anchor_years(
            combined_samples,
            imagery_anchor_mode="fixed_year",
            sample_year=args.sample_year,
            anchor_date_column=args.anchor_date_column,
            anchor_year_column=args.anchor_year_column,
            anchor_year_offset=args.anchor_year_offset,
        )
        output_dir = features_dir
        scope_id = basin["basin_id"]

    combined_samples = combined_samples[
        combined_samples["sample_longitude"].notna()
        & combined_samples["sample_latitude"].notna()
        & combined_samples["anchor_year"].notna()
    ].copy()
    if combined_samples.empty:
        raise SystemExit("No samples selected for S2/S1 chip materialization.")
    combined_samples["anchor_year"] = combined_samples["anchor_year"].astype(int)

    grouping_columns = search_group_columns(combined_samples)
    combined_samples_before_shard = len(combined_samples)
    combined_samples = apply_sample_shard(
        combined_samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        group_columns=grouping_columns,
    )
    if combined_samples.empty:
        raise SystemExit(
            f"No samples selected after applying shard_index={args.shard_index} shard_count={args.shard_count}."
        )

    catalog = Client.open(args.stac_api_url)
    chips_dir = output_dir / "ewm_s2s1_chips_v1"
    manifest_rows_by_year: list[dict[str, Any]] = []
    s2_manifest_rows_all: list[dict[str, Any]] = []
    s1_manifest_rows_all: list[dict[str, Any]] = []
    chip_index_parts: list[pd.DataFrame] = []
    total_failures = 0
    total_reused_existing = 0

    for group_key, anchor_frame in combined_samples.groupby(grouping_columns, sort=True, dropna=False):
        group_samples = anchor_frame.copy().reset_index(drop=True)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(grouping_columns, group_key))
        anchor_year = int(group_values["anchor_year"])
        bbox = [
            float(group_samples["sample_longitude"].min()) - 0.05,
            float(group_samples["sample_latitude"].min()) - 0.05,
            float(group_samples["sample_longitude"].max()) + 0.05,
            float(group_samples["sample_latitude"].max()) + 0.05,
        ]
        windows = build_windows(int(anchor_year))
        sample_points = group_samples[["sample_longitude", "sample_latitude"]].drop_duplicates().reset_index(drop=True)
        print(
            f"[group-start] shard={args.shard_index}/{args.shard_count} "
            + " ".join(f"{column}={group_values[column]}" for column in grouping_columns)
            + f" rows={len(group_samples)} bbox={bbox}",
            flush=True,
        )

        reuse_group_existing = bool(args.skip_existing and group_has_all_existing_chips(group_samples, chips_dir))
        if reuse_group_existing:
            s2_candidates: dict[str, list[dict[str, Any]]] = {}
            s1_candidates: dict[str, list[dict[str, Any]]] = {}
            s2_manifest_rows: list[dict[str, Any]] = []
            s1_manifest_rows: list[dict[str, Any]] = []
            print(
                f"[group-reuse-existing] shard={args.shard_index}/{args.shard_count} "
                + " ".join(f"{column}={group_values[column]}" for column in grouping_columns)
                + " using cached chips only",
                flush=True,
            )
        else:
            s2_candidates, s2_manifest_rows = search_candidates(
                catalog=catalog,
                collection=S2_COLLECTION,
                bbox=bbox,
                sample_points=sample_points,
                windows=windows,
                max_items=args.max_items_per_window,
                cloud_cover_max=args.s2_cloud_cover_max,
            )
            s1_candidates, s1_manifest_rows = search_candidates(
                catalog=catalog,
                collection=S1_COLLECTION,
                bbox=bbox,
                sample_points=sample_points,
                windows=windows,
                max_items=args.max_items_per_window,
                cloud_cover_max=None,
            )
        print(
            f"[group-candidates] shard={args.shard_index}/{args.shard_count} "
            + " ".join(f"{column}={group_values[column]}" for column in grouping_columns)
            + f" s2_candidates={len(s2_manifest_rows)} s1_candidates={len(s1_manifest_rows)}",
            flush=True,
        )

        for row in s2_manifest_rows:
            s2_manifest_rows_all.append({**group_values, "anchor_year": int(anchor_year), **row})
        for row in s1_manifest_rows:
            s1_manifest_rows_all.append({**group_values, "anchor_year": int(anchor_year), **row})

        chip_index_part, chip_meta_part = materialize_s2s1_chips(
            samples=group_samples,
            s2_candidates=s2_candidates,
            s1_candidates=s1_candidates,
            windows=windows,
            chips_dir=chips_dir,
            min_s2_clear_fraction=args.min_s2_clear_fraction,
            min_s1_valid_fraction=args.min_s1_valid_fraction,
            anchor_year=int(anchor_year),
            skip_existing=args.skip_existing,
        )
        chip_index_parts.append(chip_index_part)
        total_failures += int(chip_meta_part["failed_sample_count"])
        total_reused_existing += int(chip_meta_part["reused_existing_chip_count"])
        manifest_rows_by_year.append(
            {
                **group_values,
                "anchor_year": int(anchor_year),
                "bbox": bbox,
                "row_count_samples": int(len(group_samples)),
                "row_count_materialized": int(len(chip_index_part)),
                "row_count_failed": int(chip_meta_part["failed_sample_count"]),
                "windows": [
                    {"window_id": window_id, "start_date": start_date, "end_date": end_date}
                    for window_id, start_date, end_date in windows
                ],
            }
        )

    chip_index = (
        pd.concat(chip_index_parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        if chip_index_parts
        else pd.DataFrame()
    )

    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "ewm_s2s1_selection_manifest_v1",
        "scope_id": scope_id,
        "cohort_path": str(cohort_path) if cohort_path is not None else None,
        "imagery_anchor_mode": args.imagery_anchor_mode if cohort_path is not None else "fixed_year",
        "anchor_date_column": args.anchor_date_column if cohort_path is not None else None,
        "anchor_year_column": args.anchor_year_column if cohort_path is not None else None,
        "anchor_year_offset": int(args.anchor_year_offset) if cohort_path is not None else None,
        "sample_year": int(args.sample_year) if cohort_path is None else None,
        "search_group_columns": grouping_columns,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "row_count_samples_before_shard": int(combined_samples_before_shard),
        "row_count_samples": int(len(combined_samples)),
        "row_count_training_wells_selected": row_count_training_wells_selected,
        "row_count_by_anchor_year": {
            str(key): int(value) for key, value in combined_samples["anchor_year"].value_counts().sort_index().items()
        },
        "sample_groups": manifest_rows_by_year,
        "s2_collection": S2_COLLECTION,
        "s1_collection": S1_COLLECTION,
        "s2_candidates": s2_manifest_rows_all,
        "s1_candidates": s1_manifest_rows_all,
    }
    manifest_tag = shard_tag(args.shard_index, args.shard_count)
    manifest_path = output_dir / f"ewm_s2s1_selection_manifest_v1{manifest_tag}.json"
    write_json(manifest_path, manifest)

    chip_meta = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "ewm_s2s1_chip_index_v1",
        "row_count": int(len(chip_index)),
        "failed_sample_count": int(total_failures),
        "frame_count": 4,
        "chip_size_px": CHIP_SIZE,
        "modalities": ["Sentinel-2 L2A", "Sentinel-1 RTC"],
        "s2_band_order": S2_BANDS,
        "s1_band_order": S1_BANDS,
        "reused_existing_chip_count": int(total_reused_existing),
        "min_s2_clear_fraction": float(args.min_s2_clear_fraction),
        "min_s1_valid_fraction": float(args.min_s1_valid_fraction),
        "sample_type_counts": (
            {key: int(value) for key, value in chip_index["sample_type"].value_counts().items()}
            if not chip_index.empty
            else {}
        ),
        "anchor_year_counts": (
            {str(key): int(value) for key, value in chip_index["anchor_year"].value_counts().sort_index().items()}
            if (not chip_index.empty and "anchor_year" in chip_index.columns)
            else {}
        ),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "row_count_samples_before_shard": int(combined_samples_before_shard),
    }
    output_tag = shard_tag(args.shard_index, args.shard_count)
    out_path = output_dir / f"ewm_s2s1_chip_index_v1{output_tag}.parquet"
    write_parquet_atomic(chip_index, out_path)
    chip_meta["output_path"] = str(out_path)
    chip_meta["chips_dir"] = str(chips_dir)
    chip_meta["selection_manifest_path"] = str(manifest_path)
    metadata_path = output_dir / f"ewm_s2s1_chip_index_v1_metadata{output_tag}.json"
    write_json(metadata_path, chip_meta)
    print(json.dumps(chip_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
