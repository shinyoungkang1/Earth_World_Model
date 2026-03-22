#!/usr/bin/env python3
"""Run a basin-scoped HLS -> Prithvi -> XGBoost gas prospectivity pipeline."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import torch
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import Window
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import prithvi_eo_v2_300_tl
from xgboost import XGBClassifier

from gas_v1_common import load_basin_config, polygon_from_bbox, prospect_tier


CHIP_SIZE = 224
CHIP_RADIUS_PX = CHIP_SIZE // 2
PRITHVI_MEAN = np.asarray([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
PRITHVI_STD = np.asarray([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)
PRITHVI_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]
PRITHVI_BAND_LABELS = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
HLS_COLLECTIONS = ["hls2-s30", "hls2-l30"]
HLS_ASSET_MAP = {
    "hls2-s30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B8A",
        "swir1": "B11",
        "swir2": "B12",
        "fmask": "Fmask",
    },
    "hls2-l30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir1": "B06",
        "swir2": "B07",
        "fmask": "Fmask",
    },
}
COLLECTION_PREFERENCE = {collection: rank for rank, collection in enumerate(HLS_COLLECTIONS)}


def build_hls_windows(year: int) -> list[tuple[str, str, str]]:
    return [
        ("early_spring", f"{year}-03-01", f"{year}-04-30"),
        ("late_spring", f"{year}-05-01", f"{year}-06-30"),
        ("summer", f"{year}-07-01", f"{year}-08-31"),
        ("fall", f"{year}-09-01", f"{year}-10-31"),
    ]


HLS_WINDOWS = build_hls_windows(2024)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def sanitize_id(value: str) -> str:
    return value.replace("::", "__").replace("/", "_").replace(":", "_")


def point_in_bbox(lon: float, lat: float, bbox: list[float]) -> bool:
    minx, miny, maxx, maxy = bbox
    return minx <= lon <= maxx and miny <= lat <= maxy


def tile_id_from_item_id(item_id: str) -> str:
    parts = item_id.split(".")
    if len(parts) < 3:
        return "unknown"
    return parts[2]


def year_doy_from_timestamp(ts: pd.Timestamp) -> tuple[float, float]:
    ts = pd.Timestamp(ts)
    return float(ts.year), float(ts.dayofyear)


def choose_best_item(group: list[dict]) -> dict:
    return sorted(
        group,
        key=lambda item: (
            COLLECTION_PREFERENCE.get(item["collection"], 999),
            item["cloud_cover"],
            -pd.Timestamp(item["datetime"]).value,
        ),
    )[0]


def temporal_split_candidates(preferred_quantile: float) -> list[float]:
    candidates = [preferred_quantile]
    for delta in [0.05, 0.1, 0.15, 0.2]:
        for direction in [-1, 1]:
            q = round(preferred_quantile + (delta * direction), 4)
            if 0.5 <= q < 0.98:
                candidates.append(q)
    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def choose_temporal_split(
    frame: pd.DataFrame,
    date_col: str,
    label_col: str,
    preferred_quantile: float,
) -> tuple[float, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    data = frame.sort_values(date_col).copy()
    for quantile in temporal_split_candidates(preferred_quantile):
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


def stratified_binary_downsample(frame: pd.DataFrame, label_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
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


def compute_life_window_labels(
    training_table: pd.DataFrame,
    production: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    prod = production[
        (production["production_scope"] == "unconventional") & (production["permit_num"].isin(training_table["well_api"]))
    ].copy()
    prod["production_period_start_date"] = parse_date(prod["production_period_start_date"])
    prod["production_period_end_date"] = parse_date(prod["production_period_end_date"])
    prod["gas_quantity"] = pd.to_numeric(prod["gas_quantity"], errors="coerce").fillna(0.0)

    rows = []
    for well_api, frame in prod.groupby("permit_num", sort=False):
        frame = frame.sort_values("production_period_start_date")
        first_prod = frame["production_period_start_date"].min()
        if pd.isna(first_prod):
            continue
        last_prod = frame["production_period_end_date"].max()
        for horizon_days, horizon_name in [(365, "f12"), (730, "f24")]:
            window_end = first_prod + pd.Timedelta(days=horizon_days - 1)
            overlap_start = frame["production_period_start_date"].clip(lower=first_prod)
            overlap_end = frame["production_period_end_date"].clip(upper=window_end)
            overlap_days = (overlap_end - overlap_start).dt.days + 1
            period_days = (frame["production_period_end_date"] - frame["production_period_start_date"]).dt.days + 1
            valid = overlap_days.gt(0) & period_days.gt(0)
            covered_days = float(overlap_days.where(valid, 0).sum())
            cum_gas = float(
                (
                    frame["gas_quantity"]
                    * (overlap_days.where(valid, 0) / period_days.where(valid, 1))
                )
                .fillna(0.0)
                .sum()
            )
            rows.append(
                {
                    "well_api": well_api,
                    "horizon": horizon_name,
                    "first_prod_date": first_prod,
                    "last_prod_date": last_prod,
                    "covered_days": covered_days,
                    "cum_gas": cum_gas,
                }
            )

    summary = pd.DataFrame(rows)
    wide = summary.pivot(index="well_api", columns="horizon", values=["first_prod_date", "covered_days", "cum_gas"])
    wide.columns = ["_".join(column) for column in wide.columns]
    wide = wide.reset_index()

    labels = training_table[["well_api"]].merge(wide, on="well_api", how="left")
    labels["first_prod_date"] = parse_date(labels["first_prod_date_f12"])
    labels["label_f12_available"] = labels["covered_days_f12"].fillna(0).ge(330)
    labels["label_f24_available"] = labels["covered_days_f24"].fillna(0).ge(695)

    for threshold in [100000, 250000, 500000, 1000000, 2000000]:
        labels[f"label_f12_ge_{threshold}"] = (labels["cum_gas_f12"] >= threshold).where(labels["label_f12_available"])
        labels[f"label_f24_ge_{threshold}"] = (labels["cum_gas_f24"] >= threshold).where(labels["label_f24_available"])

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_labels_v2",
        "label_anchor": "first production date",
        "window_method": "Prorated cumulative gas over the first 365 or 730 days after first reported production.",
        "row_count_total": int(len(labels)),
        "row_count_f12_available": int(labels["label_f12_available"].sum()),
        "row_count_f24_available": int(labels["label_f24_available"].sum()),
        "positive_counts": {
            column: int(labels[column].astype("boolean").fillna(False).sum())
            for column in labels.columns
            if column.startswith("label_f")
            and column.endswith(("100000", "250000", "500000", "1000000", "2000000"))
        },
    }
    return labels, metadata


def search_hls_items(
    catalog: Client,
    bbox: list[float],
    sample_points: pd.DataFrame,
    windows: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, dict[str, dict]], dict]:
    windows = windows or HLS_WINDOWS
    selected_items: dict[str, dict[str, dict]] = defaultdict(dict)
    manifest_rows: list[dict] = []

    for window_id, start_date, end_date in windows:
        candidates_by_tile: dict[str, list[dict]] = defaultdict(list)
        for collection in HLS_COLLECTIONS:
            search = catalog.search(
                collections=[collection],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": 35}},
                max_items=200,
            )
            for item in search.items():
                tile_id = tile_id_from_item_id(item.id)
                candidate = {
                    "window_id": window_id,
                    "collection": collection,
                    "item_id": item.id,
                    "tile_id": tile_id,
                    "datetime": pd.Timestamp(item.datetime).isoformat(),
                    "cloud_cover": float(item.properties.get("eo:cloud_cover", 100.0)),
                    "bbox": list(item.bbox) if item.bbox else None,
                    "item": planetary_computer.sign(item),
                }
                candidates_by_tile[tile_id].append(candidate)

        for tile_id, candidates in candidates_by_tile.items():
            for candidate in candidates:
                candidate["coverage_count"] = int(
                    sample_points.apply(
                        lambda row: point_in_bbox(float(row["sample_longitude"]), float(row["sample_latitude"]), candidate["bbox"]),
                        axis=1,
                    ).sum()
                )
            best = sorted(
                candidates,
                key=lambda item: (
                    -item["coverage_count"],
                    COLLECTION_PREFERENCE.get(item["collection"], 999),
                    item["cloud_cover"],
                    -pd.Timestamp(item["datetime"]).value,
                ),
            )[0]
            selected_items[window_id][tile_id] = best
            manifest_rows.append(
                {
                    "window_id": window_id,
                    "tile_id": tile_id,
                    "collection": best["collection"],
                    "item_id": best["item_id"],
                    "datetime": best["datetime"],
                    "cloud_cover": best["cloud_cover"],
                    "coverage_count": best["coverage_count"],
                    "bbox": best["bbox"],
                }
            )

    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "hls_selection_manifest_v1",
        "windows": [
            {"window_id": window_id, "start_date": start_date, "end_date": end_date}
            for window_id, start_date, end_date in windows
        ],
        "selected_items": manifest_rows,
    }
    return selected_items, manifest


def pick_items_for_point(
    selected_items: dict[str, dict[str, dict]],
    lon: float,
    lat: float,
    windows: list[tuple[str, str, str]] | None = None,
) -> list[dict] | None:
    windows = windows or HLS_WINDOWS
    frames = []
    for window_id, _, _ in windows:
        window_items = selected_items.get(window_id, {})
        matches = [item for item in window_items.values() if item["bbox"] and point_in_bbox(lon, lat, item["bbox"])]
        if not matches:
            return None
        frames.append(choose_best_item(matches))
    return frames


def read_hls_window(item: dict, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray, dict]:
    collection = item["collection"]
    asset_map = HLS_ASSET_MAP[collection]
    band_arrays: list[np.ndarray] = []
    nodata_mask: np.ndarray | None = None
    frame_meta: dict = {
        "window_id": item["window_id"],
        "collection": collection,
        "item_id": item["item_id"],
        "tile_id": item["tile_id"],
        "datetime": item["datetime"],
        "cloud_cover": item["cloud_cover"],
    }

    signed_item = item["item"]
    fmask_asset = signed_item.assets[asset_map["fmask"]].href
    with rasterio.open(fmask_asset) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = transformer.transform(float(lon), float(lat))
        row, col = ds.index(x, y)
        window = Window(col - CHIP_RADIUS_PX, row - CHIP_RADIUS_PX, CHIP_SIZE, CHIP_SIZE)
        fill_value = 255
        fmask = ds.read(1, window=window, boundless=True, fill_value=fill_value)
        valid_pixels = fmask != fill_value
        cloud_or_shadow = (
            ((fmask & (1 << 1)) > 0)
            | ((fmask & (1 << 2)) > 0)
            | ((fmask & (1 << 3)) > 0)
            | ((fmask & (1 << 4)) > 0)
        )
        clear_mask = valid_pixels & ~cloud_or_shadow
        frame_meta["clear_fraction"] = float(clear_mask.sum() / max(1, valid_pixels.sum()))

    for band_label in PRITHVI_BAND_LABELS:
        band_key = asset_map[band_label]
        with rasterio.open(signed_item.assets[band_key].href) as ds:
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = transformer.transform(float(lon), float(lat))
            row, col = ds.index(x, y)
            window = Window(col - CHIP_RADIUS_PX, row - CHIP_RADIUS_PX, CHIP_SIZE, CHIP_SIZE)
            nodata = ds.nodata if ds.nodata is not None else -9999
            band = ds.read(1, window=window, boundless=True, fill_value=nodata).astype(np.float32)
            band_mask = (band != float(nodata)) & clear_mask
            masked = np.where(band_mask, band, np.nan).astype(np.float32)
            band_arrays.append(masked)
            nodata_mask = band_mask if nodata_mask is None else (nodata_mask & band_mask)

    return np.stack(band_arrays, axis=0), nodata_mask.astype(bool), frame_meta


def materialize_hls_chips(
    samples: pd.DataFrame,
    selected_items: dict[str, dict[str, dict]],
    chips_dir: Path,
    windows: list[tuple[str, str, str]] | None = None,
    skip_existing: bool = False,
    anchor_year: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    windows = windows or HLS_WINDOWS
    chips_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    failures = 0

    for row in samples[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]].itertuples(index=False):
        frames = pick_items_for_point(selected_items, float(row.sample_longitude), float(row.sample_latitude), windows=windows)
        if frames is None:
            failures += 1
            continue

        chip_path = chips_dir / f"{sanitize_id(row.sample_id)}.npz"
        temporal_coords = np.asarray(
            [year_doy_from_timestamp(pd.Timestamp(item["datetime"])) for item in frames],
            dtype=np.float32,
        )
        expected_item_ids = [item["item_id"] for item in frames]
        expected_datetimes = [item["datetime"] for item in frames]
        expected_collections = [item["collection"] for item in frames]

        if skip_existing and chip_path.exists():
            with np.load(chip_path) as payload:
                chip = payload["chip"]
                location_coords = payload["location_coords"]
                if (
                    chip.shape == (len(PRITHVI_BAND_LABELS), len(windows), CHIP_SIZE, CHIP_SIZE)
                    and location_coords.shape == (2,)
                    and np.allclose(
                        location_coords,
                        np.asarray([row.sample_latitude, row.sample_longitude], dtype=np.float32),
                        atol=1e-5,
                    )
                ):
                    valid_fraction = float(np.isfinite(chip).sum() / chip.size)
                    index_rows.append(
                        {
                            "sample_id": row.sample_id,
                            "sample_type": row.sample_type,
                            "sample_key": row.sample_key,
                            "sample_longitude": float(row.sample_longitude),
                            "sample_latitude": float(row.sample_latitude),
                            "chip_path": str(chip_path),
                            "frame_count": len(frames),
                            "valid_fraction": valid_fraction,
                            "clear_fraction_mean": np.nan,
                            "frame_item_ids_json": json.dumps(expected_item_ids),
                            "frame_datetimes_json": json.dumps(expected_datetimes),
                            "frame_collections_json": json.dumps(expected_collections),
                            "frame_clear_fractions_json": json.dumps([]),
                            "anchor_year": int(anchor_year) if anchor_year is not None else None,
                        }
                    )
                    continue

        chip_frames = []
        valid_masks = []
        frame_meta_rows = []
        frame_read_failed = False
        for item in frames:
            try:
                chip_frame, valid_mask, frame_meta = read_hls_window(item, float(row.sample_longitude), float(row.sample_latitude))
            except Exception as exc:
                print(f"  WARNING: Failed to read HLS tile {item.get('item_id', '?')} for {row.sample_id}: {exc}")
                frame_read_failed = True
                break
            chip_frames.append(chip_frame)
            valid_masks.append(valid_mask)
            frame_meta_rows.append(frame_meta)
        if frame_read_failed:
            failures += 1
            continue

        chip = np.stack(chip_frames, axis=1).astype(np.float32)
        valid_fraction = float(np.isfinite(chip).sum() / chip.size)
        clear_fraction_mean = float(np.mean([meta["clear_fraction"] for meta in frame_meta_rows]))
        np.savez_compressed(
            chip_path,
            chip=chip,
            temporal_coords=temporal_coords,
            location_coords=np.asarray([row.sample_latitude, row.sample_longitude], dtype=np.float32),
            valid_mask=np.stack(valid_masks, axis=0).astype(np.uint8),
        )
        index_rows.append(
            {
                "sample_id": row.sample_id,
                "sample_type": row.sample_type,
                "sample_key": row.sample_key,
                "sample_longitude": float(row.sample_longitude),
                "sample_latitude": float(row.sample_latitude),
                "chip_path": str(chip_path),
                "frame_count": len(frame_meta_rows),
                "valid_fraction": valid_fraction,
                "clear_fraction_mean": clear_fraction_mean,
                "frame_item_ids_json": json.dumps([meta["item_id"] for meta in frame_meta_rows]),
                "frame_datetimes_json": json.dumps([meta["datetime"] for meta in frame_meta_rows]),
                "frame_collections_json": json.dumps([meta["collection"] for meta in frame_meta_rows]),
                "frame_clear_fractions_json": json.dumps([meta["clear_fraction"] for meta in frame_meta_rows]),
                "anchor_year": int(anchor_year) if anchor_year is not None else None,
            }
        )

    index = pd.DataFrame(index_rows).sort_values("sample_id").reset_index(drop=True)
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "hls_chip_index_v1",
        "chip_size_px": CHIP_SIZE,
        "frame_count": len(windows),
        "row_count": int(len(index)),
        "sample_type_counts": {key: int(value) for key, value in index["sample_type"].value_counts().items()},
        "valid_fraction_mean": float(index["valid_fraction"].mean()) if not index.empty else None,
        "clear_fraction_mean": (
            float(index["clear_fraction_mean"].dropna().mean())
            if (not index.empty and index["clear_fraction_mean"].notna().any())
            else None
        ),
        "failed_sample_count": failures,
        "anchor_year": int(anchor_year) if anchor_year is not None else None,
    }
    return index, metadata


def normalize_chip_tensor(chip: np.ndarray) -> np.ndarray:
    filled = chip.copy()
    for band_idx in range(filled.shape[0]):
        band = filled[band_idx]
        band[np.isnan(band)] = PRITHVI_MEAN[band_idx]
        filled[band_idx] = band
    return (filled - PRITHVI_MEAN[:, None, None, None]) / PRITHVI_STD[:, None, None, None]


def extract_prithvi_embeddings(
    chip_index: pd.DataFrame,
    batch_size: int,
    device: str,
    existing_embeddings_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    cached = pd.DataFrame()
    if existing_embeddings_path and existing_embeddings_path.exists():
        cached = pd.read_parquet(existing_embeddings_path)
        cached = cached[cached["sample_id"].isin(chip_index["sample_id"])].copy()

    missing_ids = sorted(set(chip_index["sample_id"]) - set(cached["sample_id"])) if not cached.empty else chip_index["sample_id"].tolist()
    pending = chip_index[chip_index["sample_id"].isin(missing_ids)].copy().reset_index(drop=True)
    if pending.empty:
        frame = cached.sort_values("sample_id").reset_index(drop=True)
        metadata = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "dataset": "prithvi_embeddings_v1",
            "model_name": "Prithvi-EO-2.0-300M-TL",
            "embedding_dim": int(frame.filter(regex=r"^embedding_").shape[1]),
            "batch_size": batch_size,
            "device": device,
            "row_count": int(len(frame)),
            "sample_type_counts": {key: int(value) for key, value in frame["sample_type"].value_counts().items()},
            "reused_row_count": int(len(frame)),
            "new_row_count": 0,
        }
        return frame, metadata

    model = prithvi_eo_v2_300_tl(pretrained=True, bands=PRITHVI_BANDS, num_frames=len(HLS_WINDOWS))
    model = model.to(device).eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch_frame = pending.iloc[start : start + batch_size].copy()
            chips = []
            temporal_coords = []
            location_coords = []
            sample_ids = []
            sample_types = []
            sample_keys = []

            for row in batch_frame.itertuples(index=False):
                payload = np.load(row.chip_path)
                chips.append(normalize_chip_tensor(payload["chip"]).astype(np.float32))
                temporal_coords.append(payload["temporal_coords"].astype(np.float32))
                location_coords.append(payload["location_coords"].astype(np.float32))
                sample_ids.append(row.sample_id)
                sample_types.append(row.sample_type)
                sample_keys.append(row.sample_key)

            chip_tensor = torch.from_numpy(np.stack(chips, axis=0)).to(device=device, dtype=torch.float32)
            temporal_tensor = torch.from_numpy(np.stack(temporal_coords, axis=0)).to(device=device, dtype=torch.float32)
            location_tensor = torch.from_numpy(np.stack(location_coords, axis=0)).to(device=device, dtype=torch.float32)
            features = model.forward_features(
                chip_tensor,
                temporal_coords=temporal_tensor,
                location_coords=location_tensor,
            )
            embeddings = features[-1][:, 0, :].detach().cpu().numpy().astype(np.float32)

            for idx, sample_id in enumerate(sample_ids):
                record = {
                    "sample_id": sample_id,
                    "sample_type": sample_types[idx],
                    "sample_key": sample_keys[idx],
                }
                for col_idx in range(embeddings.shape[1]):
                    record[f"embedding_{col_idx:04d}"] = float(embeddings[idx, col_idx])
                rows.append(record)

    new_frame = pd.DataFrame(rows)
    frame = (
        pd.concat([cached, new_frame], ignore_index=True)
        .drop_duplicates(subset=["sample_id"], keep="last")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "prithvi_embeddings_v1",
        "model_name": "Prithvi-EO-2.0-300M-TL",
        "embedding_dim": int(frame.filter(regex=r"^embedding_").shape[1]),
        "batch_size": batch_size,
        "device": device,
        "row_count": int(len(frame)),
        "sample_type_counts": {key: int(value) for key, value in frame["sample_type"].value_counts().items()},
        "reused_row_count": int(len(cached)),
        "new_row_count": int(len(new_frame)),
    }
    return frame, metadata


def build_model_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    class_weight_scale: float,
    random_state: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        scale_pos_weight=class_weight_scale,
        n_jobs=0,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def score_grid_cells(
    basin: dict,
    grid_features: pd.DataFrame,
    grid_embeddings: pd.DataFrame,
    model_pipeline: Pipeline,
    feature_columns: list[str],
    label_column: str,
    model_path: Path,
    metrics_path: Path,
    derived_dir: Path,
) -> tuple[Path, Path, Path]:
    merged = grid_features.merge(grid_embeddings, on=["sample_id", "sample_type", "sample_key"], how="inner")
    merged["score"] = model_pipeline.predict_proba(merged[feature_columns])[:, 1]
    merged["score_percentile"] = merged["score"].rank(method="average", pct=True)
    merged["prospect_tier"] = merged["score_percentile"].map(prospect_tier)
    merged = merged.sort_values(["score"], ascending=False).reset_index(drop=True)
    merged["score_rank"] = np.arange(1, len(merged) + 1)

    score_csv_path = derived_dir / "gas_prospect_cells_prithvi_v1.csv"
    merged.to_csv(score_csv_path, index=False)

    features = []
    for row in merged.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "geometry": polygon_from_bbox(row.bbox_west, row.bbox_south, row.bbox_east, row.bbox_north),
                "properties": {
                    "cell_id": row.sample_key,
                    "score": round(float(row.score), 6),
                    "score_percentile": round(float(row.score_percentile), 6),
                    "score_rank": int(row.score_rank),
                    "prospect_tier": row.prospect_tier,
                    "county_name": row.county_name,
                    "geology_name": row.geology_name,
                    "geology_lith1": row.geology_lith1,
                    "fault_distance_km": round(float(row.fault_distance_km), 3) if pd.notna(row.fault_distance_km) else None,
                    "well_count_5km": int(row.well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                },
            }
        )

    geojson_path = derived_dir / "gas_prospect_cells_prithvi_v1.geojson"
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")

    top_cols = [
        "sample_key",
        "score_rank",
        "score",
        "score_percentile",
        "prospect_tier",
        "county_name",
        "geology_name",
        "geology_lith1",
        "fault_distance_km",
        "well_count_5km",
        "permit_count_5km",
        "sample_longitude",
        "sample_latitude",
    ]
    top_path = derived_dir / "top_prospects_prithvi_v1.csv"
    merged[top_cols].head(100).to_csv(top_path, index=False)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prospect_cells_prithvi_v1",
        "basin_id": basin["basin_id"],
        "basin_name": basin["display_name"],
        "row_count": int(len(merged)),
        "score_min": float(merged["score"].min()),
        "score_p50": float(merged["score"].median()),
        "score_p90": float(merged["score"].quantile(0.9)),
        "score_max": float(merged["score"].max()),
        "tier_counts": {tier: int(count) for tier, count in merged["prospect_tier"].value_counts().sort_index().items()},
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "label_column": label_column,
        "output_csv_path": str(score_csv_path),
        "output_geojson_path": str(geojson_path),
        "top_prospects_path": str(top_path),
    }
    metadata_path = derived_dir / "gas_prospect_cells_prithvi_v1_metadata.json"
    write_json(metadata_path, metadata)
    return score_csv_path, geojson_path, top_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--max-training-wells", type=int, default=1600)
    parser.add_argument("--max-grid-cells", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-results-publish", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    features_dir = repo_root / "data" / "features" / args.basin_id
    derived_dir = repo_root / "data" / "derived" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    raw_hls_dir = repo_root / "data" / "raw" / "hls" / args.basin_id
    chips_dir = features_dir / "hls_chips_v1"
    features_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    raw_hls_dir.mkdir(parents=True, exist_ok=True)

    training_table = pd.read_csv(features_dir / "gas_training_table_v1.csv")
    training_meta = load_json(features_dir / "gas_training_table_v1_metadata.json")
    production = pd.read_csv(repo_root / "data" / "canonical" / "pa_mvp" / "production.csv")
    grid_features = pd.read_csv(derived_dir / "gas_prospect_cells_v1.csv")

    labels, labels_meta = compute_life_window_labels(training_table, production)
    labels_path = features_dir / "gas_labels_v2.parquet"
    labels.to_parquet(labels_path, index=False)
    labels_meta["output_path"] = str(labels_path)
    labels_meta_path = features_dir / "gas_labels_v2_metadata.json"
    write_json(labels_meta_path, labels_meta)

    wells = training_table.merge(labels, on="well_api", how="left")
    wells["sample_id"] = wells["well_api"].map(lambda value: f"well::{value}")
    wells["sample_type"] = "well"
    wells["sample_key"] = wells["well_api"]
    wells["sample_longitude"] = wells["longitude_decimal"]
    wells["sample_latitude"] = wells["latitude_decimal"]
    wells["first_prod_date"] = parse_date(wells["first_prod_date"])

    grid = grid_features.copy()
    grid["sample_id"] = grid["cell_id"].map(lambda value: f"cell::{value}")
    grid["sample_type"] = "grid_cell"
    grid["sample_key"] = grid["cell_id"]
    grid["sample_longitude"] = grid["center_longitude"]
    grid["sample_latitude"] = grid["center_latitude"]
    grid["longitude_decimal"] = grid["center_longitude"]
    grid["latitude_decimal"] = grid["center_latitude"]
    if args.max_grid_cells and args.max_grid_cells > 0 and len(grid) > args.max_grid_cells:
        grid = grid.head(args.max_grid_cells).copy()

    wells_available = wells[
        wells[args.label_column].notna()
        & wells["sample_latitude"].notna()
        & wells["sample_longitude"].notna()
        & wells["first_prod_date"].notna()
    ].copy()
    wells_selected = stratified_binary_downsample(
        wells_available,
        args.label_column,
        max_rows=args.max_training_wells,
        random_state=args.random_state,
    )
    wells_selected = wells_selected.sort_values("first_prod_date").reset_index(drop=True)

    combined_samples = pd.concat(
        [
            wells_selected[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]],
            grid[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]],
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")

    min_lon = float(combined_samples["sample_longitude"].min()) - 0.05
    min_lat = float(combined_samples["sample_latitude"].min()) - 0.05
    max_lon = float(combined_samples["sample_longitude"].max()) + 0.05
    max_lat = float(combined_samples["sample_latitude"].max()) + 0.05
    stac_client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    selected_items, hls_manifest = search_hls_items(
        stac_client,
        [min_lon, min_lat, max_lon, max_lat],
        combined_samples[["sample_longitude", "sample_latitude"]].drop_duplicates().reset_index(drop=True),
    )
    write_json(raw_hls_dir / "hls_selection_manifest_v1.json", hls_manifest)

    chip_index, chip_meta = materialize_hls_chips(combined_samples, selected_items, chips_dir)
    chip_index_path = features_dir / "hls_chip_index_v1.parquet"
    chip_index.to_parquet(chip_index_path, index=False)
    chip_meta["output_path"] = str(chip_index_path)
    write_json(features_dir / "hls_chip_index_v1_metadata.json", chip_meta)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, embedding_meta = extract_prithvi_embeddings(
        chip_index=chip_index,
        batch_size=args.embedding_batch_size,
        device=device,
        existing_embeddings_path=features_dir / "prithvi_embeddings_v1.parquet",
    )
    embeddings_path = features_dir / "prithvi_embeddings_v1.parquet"
    embeddings.to_parquet(embeddings_path, index=False)
    embedding_meta["output_path"] = str(embeddings_path)
    write_json(features_dir / "prithvi_embeddings_v1_metadata.json", embedding_meta)

    feature_columns_numeric = training_meta["feature_columns_numeric"]
    feature_columns_categorical = training_meta["feature_columns_categorical"]

    training_frame = wells_selected.merge(
        embeddings,
        on=["sample_id", "sample_type", "sample_key"],
        how="inner",
    ).copy()
    embedding_columns = [column for column in training_frame.columns if column.startswith("embedding_")]
    feature_columns = feature_columns_numeric + feature_columns_categorical + embedding_columns
    training_frame[args.label_column] = training_frame[args.label_column].astype(bool)

    chosen_quantile, split_date, train, test = choose_temporal_split(
        training_frame,
        date_col="first_prod_date",
        label_col=args.label_column,
        preferred_quantile=args.holdout_quantile,
    )
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set for the Prithvi pipeline.")

    positives = int(train[args.label_column].sum())
    negatives = int((~train[args.label_column]).sum())
    class_weight_scale = float(negatives / max(1, positives))
    model_pipeline = build_model_pipeline(
        numeric_columns=feature_columns_numeric + embedding_columns,
        categorical_columns=feature_columns_categorical,
        class_weight_scale=class_weight_scale,
        random_state=args.random_state,
    )
    model_pipeline.fit(train[feature_columns], train[args.label_column])
    test_scores = model_pipeline.predict_proba(test[feature_columns])[:, 1]

    final_pipeline = build_model_pipeline(
        numeric_columns=feature_columns_numeric + embedding_columns,
        categorical_columns=feature_columns_categorical,
        class_weight_scale=float((~training_frame[args.label_column]).sum() / max(1, training_frame[args.label_column].sum())),
        random_state=args.random_state,
    )
    final_pipeline.fit(training_frame[feature_columns], training_frame[args.label_column])

    model_path = models_dir / f"gas_prithvi_xgboost_v1_{args.label_column}.joblib"
    metrics_path = models_dir / f"gas_prithvi_xgboost_v1_{args.label_column}_metrics.json"
    joblib.dump(final_pipeline, model_path)

    fused_features_path = features_dir / "fused_features_v1.parquet"
    training_frame.to_parquet(fused_features_path, index=False)

    roc_auc_test = (
        float(roc_auc_score(test[args.label_column], test_scores))
        if test[args.label_column].nunique() > 1
        else None
    )
    average_precision_test = (
        float(average_precision_score(test[args.label_column], test_scores))
        if test[args.label_column].nunique() > 1
        else None
    )

    metrics = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prithvi_xgboost_v1",
        "basin_id": basin["basin_id"],
        "basin_name": basin["display_name"],
        "label_column": args.label_column,
        "row_count_training_available": int(len(wells_available)),
        "row_count_training_selected": int(len(wells_selected)),
        "row_count_with_embeddings": int(len(training_frame)),
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(train[args.label_column].mean()),
        "positive_rate_test": float(test[args.label_column].mean()),
        "roc_auc_test": roc_auc_test,
        "average_precision_test": average_precision_test,
        "temporal_holdout_anchor": "first_prod_date",
        "temporal_holdout_quantile": chosen_quantile,
        "temporal_holdout_cutoff_date": split_date.strftime("%Y-%m-%d"),
        "train_first_prod_start": train["first_prod_date"].min().strftime("%Y-%m-%d"),
        "train_first_prod_end": train["first_prod_date"].max().strftime("%Y-%m-%d"),
        "test_first_prod_start": test["first_prod_date"].min().strftime("%Y-%m-%d"),
        "test_first_prod_end": test["first_prod_date"].max().strftime("%Y-%m-%d"),
        "model_type": "xgboost_classifier_gpu",
        "embedding_model": "Prithvi-EO-2.0-300M-TL",
        "embedding_dim": int(len(embedding_columns)),
        "embedding_columns": embedding_columns,
        "feature_columns": feature_columns,
        "tabular_feature_columns_numeric": feature_columns_numeric,
        "tabular_feature_columns_categorical": feature_columns_categorical,
        "fused_feature_count": int(len(feature_columns)),
        "class_weight_scale": class_weight_scale,
        "model_path": str(model_path),
        "fused_features_path": str(fused_features_path),
    }
    write_json(metrics_path, metrics)

    tabular_wells_path = features_dir / "tabular_features_v2.parquet"
    tabular_cells_path = features_dir / "tabular_candidate_cells_v2.parquet"
    wells.to_parquet(tabular_wells_path, index=False)
    grid.to_parquet(tabular_cells_path, index=False)

    score_csv_path, score_geojson_path, top_path = score_grid_cells(
        basin=basin,
        grid_features=grid,
        grid_embeddings=embeddings[embeddings["sample_type"] == "grid_cell"].copy(),
        model_pipeline=final_pipeline,
        feature_columns=feature_columns,
        label_column=args.label_column,
        model_path=model_path,
        metrics_path=metrics_path,
        derived_dir=derived_dir,
    )

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "prithvi_gas_pipeline_v1",
        "basin_id": basin["basin_id"],
        "label_column": args.label_column,
        "selected_training_wells": int(len(wells_selected)),
        "selected_grid_cells": int(len(grid)),
        "chip_index_path": str(chip_index_path),
        "embedding_path": str(embeddings_path),
        "model_metrics_path": str(metrics_path),
        "prospect_csv_path": str(score_csv_path),
        "prospect_geojson_path": str(score_geojson_path),
        "top_prospects_path": str(top_path),
    }
    write_json(derived_dir / "prithvi_gas_pipeline_v1_summary.json", summary)

    if not args.skip_results_publish:
        subprocess.run(["python", "scripts/publish_results_snapshot.py"], cwd=repo_root, check=True)

    print(json.dumps({**summary, "model_metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
