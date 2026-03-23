#!/usr/bin/env python3
"""Generate a dense location pool from region bounding boxes.

This is the first reusable step for moving beyond hand-curated seed CSVs.
It expands broad regions into many distinct local chip centers using a simple
lat/lon grid with optional per-region caps.

The output is a CSV or parquet with:

- sample_id
- region_id
- latitude
- longitude
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dense temporal location pool from region bounding boxes."
    )
    parser.add_argument(
        "--regions-path",
        required=True,
        help="CSV or parquet with region_id,min_lon,min_lat,max_lon,max_lat columns.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sample-prefix", default="dense")
    parser.add_argument(
        "--spacing-km",
        type=float,
        default=25.0,
        help="Approximate spacing between chip centers in kilometers.",
    )
    parser.add_argument(
        "--max-points-per-region",
        type=int,
        default=0,
        help="Optional cap per region. Zero means no cap.",
    )
    parser.add_argument(
        "--target-total-points",
        type=int,
        default=0,
        help="Optional global cap after concatenation. Zero means no cap.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table extension: {path}")


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output extension: {path}")


def km_per_degree_lat() -> float:
    return 111.32


def km_per_degree_lon(latitude_deg: float) -> float:
    return max(111.32 * math.cos(math.radians(latitude_deg)), 1e-6)


def generate_region_points(
    *,
    region_id: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    spacing_km: float,
    max_points_per_region: int,
    sample_prefix: str,
    sample_counter_start: int,
) -> tuple[list[dict[str, float | str]], int]:
    rows: list[dict[str, float | str]] = []
    counter = sample_counter_start

    lat_step = spacing_km / km_per_degree_lat()
    lat = float(min_lat)
    while lat <= float(max_lat):
        lon_step = spacing_km / km_per_degree_lon(lat)
        lon = float(min_lon)
        while lon <= float(max_lon):
            sample_id = f"{sample_prefix}_{counter:08d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "region_id": str(region_id),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )
            counter += 1
            if max_points_per_region > 0 and len(rows) >= max_points_per_region:
                return rows, counter
            lon += lon_step
        lat += lat_step
    return rows, counter


def main() -> int:
    args = parse_args()
    regions_path = Path(args.regions_path)
    output_path = Path(args.output_path)

    frame = read_table(regions_path)
    required = {"region_id", "min_lon", "min_lat", "max_lon", "max_lat"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required region columns: {missing}")

    all_rows: list[dict[str, float | str]] = []
    counter = 0
    for row in frame.itertuples(index=False):
        region_rows, counter = generate_region_points(
            region_id=str(getattr(row, "region_id")),
            min_lon=float(getattr(row, "min_lon")),
            min_lat=float(getattr(row, "min_lat")),
            max_lon=float(getattr(row, "max_lon")),
            max_lat=float(getattr(row, "max_lat")),
            spacing_km=float(args.spacing_km),
            max_points_per_region=int(args.max_points_per_region),
            sample_prefix=str(args.sample_prefix),
            sample_counter_start=counter,
        )
        all_rows.extend(region_rows)

    result = pd.DataFrame(all_rows)
    if args.target_total_points > 0 and not result.empty:
        result = result.head(int(args.target_total_points)).copy()

    write_table(output_path, result)
    print(
        {
            "regions_path": str(regions_path),
            "output_path": str(output_path),
            "region_count": int(len(frame)),
            "point_count": int(len(result)),
            "spacing_km": float(args.spacing_km),
            "max_points_per_region": int(args.max_points_per_region),
            "target_total_points": int(args.target_total_points),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
