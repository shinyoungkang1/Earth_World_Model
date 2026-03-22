#!/usr/bin/env python3
"""Plan dense temporal Sentinel-2 + Sentinel-1 location-year sequences from STAC.

This is the first reusable step for the post-SSL4EO dense temporal corpus:

1. Expand input locations into location-year sequences.
2. Build weekly or bi-weekly bins for each sequence.
3. Search STAC once per search-group-year for Sentinel-2 and Sentinel-1 items.
4. Pick the best candidate per bin and write a reusable manifest.

The output is a planning artifact, not a pixel corpus yet. Later materialization
can read the emitted parquet and fetch/render the selected items into chips or
Zarr sequences.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pystac_client import Client
from pystac_client.exceptions import APIError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAC_API_URL = "https://stac.dataspace.copernicus.eu/v1"
DEFAULT_S2_COLLECTION = "sentinel-2-l2a"
DEFAULT_S1_COLLECTION = "sentinel-1-grd"
DEFAULT_S2_ASSETS = [
    "B02_10m",
    "B03_10m",
    "B04_10m",
    "B05_20m",
    "B06_20m",
    "B07_20m",
    "B08_10m",
    "B8A_20m",
    "B11_20m",
    "B12_20m",
    "SCL_20m",
]
DEFAULT_S1_ASSETS = ["VV", "VH"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan weekly or bi-weekly Sentinel-2 + Sentinel-1 location-year sequences from STAC."
    )
    parser.add_argument("--locations-path", required=True, help="CSV or parquet containing at least lat/lon columns.")
    parser.add_argument("--output-dir", default=None, help="Directory for parquet/json outputs.")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument(
        "--year-column",
        default=None,
        help="Optional column containing the anchor year for each location. Ignored if --years is supplied.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated explicit years to explode each location into, e.g. 2022,2023,2024.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional column to group searches before year is appended. Useful for curated regions.",
    )
    parser.add_argument(
        "--grid-size-deg",
        type=float,
        default=1.0,
        help="Fallback spatial grouping grid when --group-column is not supplied.",
    )
    parser.add_argument(
        "--search-padding-deg",
        type=float,
        default=0.10,
        help="Degrees to pad the group bbox before STAC search.",
    )
    parser.add_argument("--cadence-days", type=int, default=7, help="7 for weekly, 14 for bi-weekly.")
    parser.add_argument("--sequence-limit", type=int, default=0, help="Optional cap on location-year sequences.")
    parser.add_argument("--dry-run", action="store_true", help="Only write location-year and group summaries.")
    parser.add_argument("--stac-api-url", default=DEFAULT_STAC_API_URL)
    parser.add_argument("--s2-collection", default=DEFAULT_S2_COLLECTION)
    parser.add_argument("--s1-collection", default=DEFAULT_S1_COLLECTION)
    parser.add_argument(
        "--s2-required-assets",
        default=",".join(DEFAULT_S2_ASSETS),
        help="Comma-separated asset keys required for Sentinel-2 candidates.",
    )
    parser.add_argument(
        "--s1-required-assets",
        default=",".join(DEFAULT_S1_ASSETS),
        help="Comma-separated asset keys required for Sentinel-1 candidates.",
    )
    parser.add_argument("--s2-cloud-cover-max", type=float, default=40.0)
    parser.add_argument("--max-items-per-search", type=int, default=2000)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--base-delay-seconds", type=float, default=5.0)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table extension for {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(locations_path: Path, explicit_output_dir: str | None, cadence_days: int) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
        return output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir)
    return REPO_ROOT / "data" / "raw" / "dense_temporal_plans" / f"{locations_path.stem}_{cadence_days}d"


def parse_years(years_arg: str | None) -> list[int]:
    if not years_arg:
        return []
    years = [int(piece.strip()) for piece in years_arg.split(",") if piece.strip()]
    if not years:
        raise ValueError("--years was provided but no usable values were found")
    return sorted(set(years))


def build_location_year_sequences(
    frame: pd.DataFrame,
    sample_id_column: str,
    latitude_column: str,
    longitude_column: str,
    year_column: str | None,
    years: list[int],
) -> pd.DataFrame:
    data = frame.copy()
    required = [latitude_column, longitude_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if sample_id_column not in data.columns:
        data[sample_id_column] = [f"sample_{idx:07d}" for idx in range(len(data))]

    data[latitude_column] = pd.to_numeric(data[latitude_column], errors="coerce")
    data[longitude_column] = pd.to_numeric(data[longitude_column], errors="coerce")
    data = data.dropna(subset=[sample_id_column, latitude_column, longitude_column]).copy()

    if years:
        exploded_rows: list[pd.DataFrame] = []
        for year in years:
            year_frame = data.copy()
            year_frame["anchor_year"] = year
            exploded_rows.append(year_frame)
        result = pd.concat(exploded_rows, ignore_index=True)
    else:
        if not year_column:
            raise ValueError("Provide either --years or --year-column")
        if year_column not in data.columns:
            raise ValueError(f"--year-column={year_column} was not found in {list(data.columns)}")
        data["anchor_year"] = pd.to_numeric(data[year_column], errors="coerce").astype("Int64")
        result = data.dropna(subset=["anchor_year"]).copy()
        result["anchor_year"] = result["anchor_year"].astype(int)

    result["sequence_id"] = result.apply(
        lambda row: f"{row[sample_id_column]}__year_{int(row['anchor_year'])}",
        axis=1,
    )
    return result.reset_index(drop=True)


def grid_group_id(latitude: float, longitude: float, grid_size_deg: float) -> str:
    lat_bucket = math.floor((latitude + 90.0) / grid_size_deg)
    lon_bucket = math.floor((longitude + 180.0) / grid_size_deg)
    return f"lat_{lat_bucket:03d}_lon_{lon_bucket:03d}"


def assign_search_groups(
    sequences: pd.DataFrame,
    *,
    group_column: str | None,
    latitude_column: str,
    longitude_column: str,
    grid_size_deg: float,
) -> pd.DataFrame:
    result = sequences.copy()
    if group_column and group_column in result.columns:
        base_group = result[group_column].fillna("unknown_group").astype(str)
    else:
        base_group = result.apply(
            lambda row: grid_group_id(float(row[latitude_column]), float(row[longitude_column]), grid_size_deg),
            axis=1,
        )
    result["search_group_id"] = [
        f"{group}__year_{int(year)}" for group, year in zip(base_group, result["anchor_year"], strict=False)
    ]
    return result


def build_bins_for_year(year: int, cadence_days: int) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    current = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    bin_index = 0
    while current < end:
        next_start = min(current + pd.Timedelta(days=cadence_days), end)
        bins.append(
            {
                "bin_index": bin_index,
                "bin_start": current,
                "bin_end": next_start,
                "bin_midpoint": current + ((next_start - current) / 2),
            }
        )
        current = next_start
        bin_index += 1
    return bins


def point_in_bbox(longitude: float, latitude: float, bbox: list[float] | None) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    minx, miny, maxx, maxy = bbox
    return minx <= longitude <= maxx and miny <= latitude <= maxy


def normalize_required_assets(text: str) -> list[str]:
    return [asset.strip() for asset in text.split(",") if asset.strip()]


def has_required_assets(item: Any, required_assets: list[str]) -> bool:
    if not required_assets:
        return True
    present = {key.lower() for key in item.assets.keys()}
    return all(asset.lower() in present for asset in required_assets)


def extract_cloud_cover(item: Any) -> float | None:
    for key in ("eo:cloud_cover", "s2:cloud_cover", "cloud_cover"):
        value = item.properties.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def search_items_with_backoff(
    catalog: Client,
    search_kwargs: dict[str, Any],
    *,
    max_attempts: int,
    base_delay_seconds: float,
) -> list[Any]:
    for attempt in range(max_attempts):
        try:
            return list(catalog.search(**search_kwargs).items())
        except APIError as exc:
            message = str(exc).lower()
            if "429" not in message and "rate limit" not in message:
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


def search_candidates(
    catalog: Client,
    *,
    collection: str,
    bbox: list[float],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_items: int,
    required_assets: list[str],
    cloud_cover_max: float | None,
    max_attempts: int,
    base_delay_seconds: float,
) -> list[dict[str, Any]]:
    search_kwargs: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start_dt.isoformat()}/{end_dt.isoformat()}",
        "max_items": max_items,
    }
    if cloud_cover_max is not None:
        search_kwargs["query"] = {"eo:cloud_cover": {"lt": cloud_cover_max}}

    candidates: list[dict[str, Any]] = []
    for item in search_items_with_backoff(
        catalog,
        search_kwargs,
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
    ):
        if item.datetime is None:
            continue
        if not has_required_assets(item, required_assets):
            continue
        candidates.append(
            {
                "item_id": item.id,
                "collection": collection,
                "datetime": pd.Timestamp(item.datetime),
                "cloud_cover": extract_cloud_cover(item),
                "bbox": list(item.bbox) if item.bbox else None,
            }
        )
    return candidates


def candidate_sort_key(candidate: dict[str, Any], *, modality: str, bin_midpoint: pd.Timestamp) -> tuple[float, float, float]:
    dt = candidate["datetime"]
    time_distance_seconds = abs((dt - bin_midpoint).total_seconds())
    recency_value = -float(dt.value)
    if modality == "s2":
        cloud_cover = candidate["cloud_cover"] if candidate["cloud_cover"] is not None else 999.0
        return (cloud_cover, time_distance_seconds, recency_value)
    return (time_distance_seconds, 0.0, recency_value)


def pick_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    modality: str,
    longitude: float,
    latitude: float,
    bin_start: pd.Timestamp,
    bin_end: pd.Timestamp,
    bin_midpoint: pd.Timestamp,
) -> dict[str, Any] | None:
    eligible = [
        candidate
        for candidate in candidates
        if bin_start <= candidate["datetime"] < bin_end and point_in_bbox(longitude, latitude, candidate["bbox"])
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda candidate: candidate_sort_key(candidate, modality=modality, bin_midpoint=bin_midpoint))


def plan_group_sequences(
    group_frame: pd.DataFrame,
    *,
    latitude_column: str,
    longitude_column: str,
    sample_id_column: str,
    cadence_days: int,
    s2_candidates: list[dict[str, Any]],
    s1_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in group_frame.itertuples(index=False):
        latitude = float(getattr(row, latitude_column))
        longitude = float(getattr(row, longitude_column))
        anchor_year = int(getattr(row, "anchor_year"))
        bins = build_bins_for_year(anchor_year, cadence_days)

        s2_found = 0
        s1_found = 0
        paired_found = 0
        for bin_spec in bins:
            s2_match = pick_best_candidate(
                s2_candidates,
                modality="s2",
                longitude=longitude,
                latitude=latitude,
                bin_start=bin_spec["bin_start"],
                bin_end=bin_spec["bin_end"],
                bin_midpoint=bin_spec["bin_midpoint"],
            )
            s1_match = pick_best_candidate(
                s1_candidates,
                modality="s1",
                longitude=longitude,
                latitude=latitude,
                bin_start=bin_spec["bin_start"],
                bin_end=bin_spec["bin_end"],
                bin_midpoint=bin_spec["bin_midpoint"],
            )
            if s2_match is not None:
                s2_found += 1
            if s1_match is not None:
                s1_found += 1
            if s2_match is not None and s1_match is not None:
                paired_found += 1

            plan_rows.append(
                {
                    "sequence_id": getattr(row, "sequence_id"),
                    "sample_id": getattr(row, sample_id_column),
                    "anchor_year": anchor_year,
                    "search_group_id": getattr(row, "search_group_id"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "bin_index": int(bin_spec["bin_index"]),
                    "bin_start": bin_spec["bin_start"].isoformat(),
                    "bin_end": bin_spec["bin_end"].isoformat(),
                    "cadence_days": cadence_days,
                    "s2_found": s2_match is not None,
                    "s2_item_id": s2_match["item_id"] if s2_match is not None else None,
                    "s2_collection": s2_match["collection"] if s2_match is not None else None,
                    "s2_datetime": s2_match["datetime"].isoformat() if s2_match is not None else None,
                    "s2_cloud_cover": s2_match["cloud_cover"] if s2_match is not None else None,
                    "s1_found": s1_match is not None,
                    "s1_item_id": s1_match["item_id"] if s1_match is not None else None,
                    "s1_collection": s1_match["collection"] if s1_match is not None else None,
                    "s1_datetime": s1_match["datetime"].isoformat() if s1_match is not None else None,
                    "paired_found": s2_match is not None and s1_match is not None,
                }
            )

        summary_rows.append(
            {
                "sequence_id": getattr(row, "sequence_id"),
                "sample_id": getattr(row, sample_id_column),
                "anchor_year": anchor_year,
                "search_group_id": getattr(row, "search_group_id"),
                "latitude": latitude,
                "longitude": longitude,
                "bin_count": len(bins),
                "s2_found_bins": s2_found,
                "s1_found_bins": s1_found,
                "paired_found_bins": paired_found,
                "paired_coverage_fraction": (paired_found / len(bins)) if bins else 0.0,
            }
        )
    return plan_rows, summary_rows


def main() -> int:
    args = parse_args()

    locations_path = Path(args.locations_path)
    output_dir = resolve_output_dir(locations_path, args.output_dir, args.cadence_days)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_frame = read_table(locations_path)
    years = parse_years(args.years)
    sequences = build_location_year_sequences(
        input_frame,
        sample_id_column=args.sample_id_column,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        year_column=args.year_column,
        years=years,
    )
    sequences = assign_search_groups(
        sequences,
        group_column=args.group_column,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        grid_size_deg=args.grid_size_deg,
    )
    sequences = sequences.sort_values(["search_group_id", "sequence_id"]).reset_index(drop=True)

    if args.sequence_limit > 0:
        sequences = sequences.head(args.sequence_limit).copy()

    sequences_path = output_dir / "location_year_sequences.parquet"
    sequences.to_parquet(sequences_path, index=False)

    group_rows: list[dict[str, Any]] = []
    for group_id, group_frame in sequences.groupby("search_group_id", dropna=False):
        year = int(group_frame["anchor_year"].iloc[0])
        latitudes = pd.to_numeric(group_frame[args.latitude_column], errors="coerce")
        longitudes = pd.to_numeric(group_frame[args.longitude_column], errors="coerce")
        bbox = [
            float(longitudes.min() - args.search_padding_deg),
            float(latitudes.min() - args.search_padding_deg),
            float(longitudes.max() + args.search_padding_deg),
            float(latitudes.max() + args.search_padding_deg),
        ]
        group_rows.append(
            {
                "search_group_id": group_id,
                "anchor_year": year,
                "sequence_count": int(len(group_frame)),
                "bbox_json": json.dumps(bbox),
            }
        )
    groups_frame = pd.DataFrame(group_rows)
    groups_path = output_dir / "search_groups.parquet"
    groups_frame.to_parquet(groups_path, index=False)

    metadata: dict[str, Any] = {
        "locations_path": str(locations_path),
        "sequences_path": str(sequences_path),
        "search_groups_path": str(groups_path),
        "output_dir": str(output_dir),
        "stac_api_url": args.stac_api_url,
        "cadence_days": int(args.cadence_days),
        "sequence_count": int(len(sequences)),
        "search_group_count": int(len(groups_frame)),
        "years": sorted(sequences["anchor_year"].unique().tolist()) if not sequences.empty else [],
        "s2_collection": args.s2_collection,
        "s1_collection": args.s1_collection,
        "s2_required_assets": normalize_required_assets(args.s2_required_assets),
        "s1_required_assets": normalize_required_assets(args.s1_required_assets),
        "s2_cloud_cover_max": args.s2_cloud_cover_max,
        "max_items_per_search": int(args.max_items_per_search),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2))
        return 0

    if sequences.empty:
        write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2))
        return 0

    catalog = Client.open(args.stac_api_url)
    s2_required_assets = normalize_required_assets(args.s2_required_assets)
    s1_required_assets = normalize_required_assets(args.s1_required_assets)

    plan_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_stats: list[dict[str, Any]] = []

    for group_row in groups_frame.itertuples(index=False):
        group_id = str(group_row.search_group_id)
        group_frame = sequences[sequences["search_group_id"] == group_id].copy()
        year = int(group_row.anchor_year)
        bbox = json.loads(group_row.bbox_json)
        start_dt = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end_dt = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")

        print(f"Planning {group_id} with {len(group_frame)} sequences", flush=True)

        s2_candidates = search_candidates(
            catalog,
            collection=args.s2_collection,
            bbox=bbox,
            start_dt=start_dt,
            end_dt=end_dt,
            max_items=args.max_items_per_search,
            required_assets=s2_required_assets,
            cloud_cover_max=args.s2_cloud_cover_max,
            max_attempts=args.max_attempts,
            base_delay_seconds=args.base_delay_seconds,
        )
        s1_candidates = search_candidates(
            catalog,
            collection=args.s1_collection,
            bbox=bbox,
            start_dt=start_dt,
            end_dt=end_dt,
            max_items=args.max_items_per_search,
            required_assets=s1_required_assets,
            cloud_cover_max=None,
            max_attempts=args.max_attempts,
            base_delay_seconds=args.base_delay_seconds,
        )

        planned_rows, planned_summaries = plan_group_sequences(
            group_frame,
            latitude_column=args.latitude_column,
            longitude_column=args.longitude_column,
            sample_id_column=args.sample_id_column,
            cadence_days=args.cadence_days,
            s2_candidates=s2_candidates,
            s1_candidates=s1_candidates,
        )
        plan_rows.extend(planned_rows)
        summary_rows.extend(planned_summaries)
        group_stats.append(
            {
                "search_group_id": group_id,
                "anchor_year": year,
                "sequence_count": int(len(group_frame)),
                "s2_candidate_count": int(len(s2_candidates)),
                "s1_candidate_count": int(len(s1_candidates)),
            }
        )

    plan_frame = pd.DataFrame(plan_rows)
    summary_frame = pd.DataFrame(summary_rows)
    group_stats_frame = pd.DataFrame(group_stats)

    plan_path = output_dir / "sequence_plan.parquet"
    summary_path = output_dir / "sequence_summary.parquet"
    group_stats_path = output_dir / "search_group_stats.parquet"
    plan_frame.to_parquet(plan_path, index=False)
    summary_frame.to_parquet(summary_path, index=False)
    group_stats_frame.to_parquet(group_stats_path, index=False)

    paired_bins = int(plan_frame["paired_found"].sum()) if not plan_frame.empty else 0
    total_bins = int(len(plan_frame))
    paired_fraction = (paired_bins / total_bins) if total_bins else 0.0
    coverage_histogram = Counter()
    if not summary_frame.empty:
        rounded = (summary_frame["paired_coverage_fraction"] * 100.0).round(-1).fillna(0).astype(int)
        coverage_histogram.update(rounded.tolist())

    metadata.update(
        {
            "plan_path": str(plan_path),
            "summary_path": str(summary_path),
            "search_group_stats_path": str(group_stats_path),
            "paired_bins": paired_bins,
            "total_bins": total_bins,
            "paired_fraction": paired_fraction,
            "coverage_histogram_pct_rounded": dict(sorted(coverage_histogram.items())),
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
