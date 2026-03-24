#!/usr/bin/env python3
"""Generate a stable, diversified pool of candidate dense-temporal regions.

The output unit is a region planning container, not a final training chip.
Each region is represented by:

- a stable region_id
- a center point
- a geographic bounding box for downstream point generation
- diversification buckets such as continent and latitude band
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import site
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


REPO_ROOT = Path(__file__).resolve().parents[2]
WGS84 = "EPSG:4326"
DEFAULT_EQUAL_AREA_CRS = "EPSG:6933"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stable diversified candidate-region pool over global land."
    )
    parser.add_argument("--output-path", required=True, help="CSV or parquet path for the generated candidate regions.")
    parser.add_argument(
        "--land-path",
        default=None,
        help="Optional vector file containing global land/country polygons. "
        "If omitted, the script will try to use a local Natural Earth fallback.",
    )
    parser.add_argument(
        "--target-total-regions",
        type=int,
        default=30000,
        help="Target number of final candidate regions after diversification-aware downsampling.",
    )
    parser.add_argument(
        "--region-side-km",
        type=float,
        default=15.0,
        help="Side length of each region planning container in kilometers.",
    )
    parser.add_argument(
        "--region-spacing-km",
        type=float,
        default=50.0,
        help="Spacing between candidate region centers on the equal-area grid in kilometers.",
    )
    parser.add_argument(
        "--latitude-band-width-deg",
        type=float,
        default=15.0,
        help="Latitude bucket width used for diversification accounting.",
    )
    parser.add_argument(
        "--equal-bucket-fraction",
        type=float,
        default=0.5,
        help="Fraction of the target allocated evenly across occupied continent x latitude buckets. "
        "The remainder is allocated proportional to available candidate count.",
    )
    parser.add_argument(
        "--projected-crs",
        default=DEFAULT_EQUAL_AREA_CRS,
        help="Projected CRS used for region spacing and region box construction.",
    )
    parser.add_argument(
        "--exclude-antarctica",
        action="store_true",
        help="Exclude Antarctica from the land polygons if a continent column is available.",
    )
    parser.add_argument(
        "--selection-seed",
        default="earth_world_model_region_v1",
        help="Stable seed string used for deterministic within-bucket ordering.",
    )
    parser.add_argument(
        "--sample-prefix",
        default="region",
        help="Prefix for generated region identifiers.",
    )
    return parser.parse_args()


def detect_default_land_path() -> Path | None:
    candidates: list[Path] = []
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        base = Path(root)
        candidates.append(base / "pyogrio" / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported output extension: {path}")


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


def load_land_polygons(path: Path, *, exclude_antarctica: bool) -> gpd.GeoDataFrame:
    land = gpd.read_file(path)
    if land.empty:
        raise ValueError(f"Land polygon file is empty: {path}")
    if land.crs is None:
        raise ValueError(f"Land polygon file has no CRS: {path}")
    land = land.to_crs(WGS84)
    land = land[land.geometry.notna()].copy()
    land = land[~land.geometry.is_empty].copy()

    if exclude_antarctica:
        lowered = {column.lower(): column for column in land.columns}
        continent_col = lowered.get("continent")
        name_col = lowered.get("name")
        if continent_col is not None:
            land = land[land[continent_col].fillna("") != "Antarctica"].copy()
        elif name_col is not None:
            land = land[land[name_col].fillna("") != "Antarctica"].copy()

    land["continent_bucket"] = normalize_bucket_column(land, ["continent"], default="unknown")
    land["country_bucket"] = normalize_bucket_column(land, ["iso_a3", "name"], default="unknown")
    land = land[land["continent_bucket"] != "seven_seas_(open_ocean)"].copy()
    return land


def normalize_bucket_value(value: object, default: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    return (
        text.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_bucket_column(frame: gpd.GeoDataFrame, names: list[str], *, default: str) -> pd.Series:
    lowered = {column.lower(): column for column in frame.columns}
    for name in names:
        column = lowered.get(name.lower())
        if column is not None:
            return frame[column].map(lambda value: normalize_bucket_value(value, default))
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def make_latitude_band(latitude_deg: float, width_deg: float) -> str:
    start = int(math.floor(latitude_deg / width_deg) * width_deg)
    end = start + int(width_deg)
    return f"lat_{start:+03d}_{end:+03d}".replace("+", "p").replace("-", "m")


def generate_equal_area_grid(bounds: tuple[float, float, float, float], spacing_m: float) -> pd.DataFrame:
    minx, miny, maxx, maxy = bounds
    x0 = math.floor(minx / spacing_m) * spacing_m
    y0 = math.floor(miny / spacing_m) * spacing_m
    x1 = math.ceil(maxx / spacing_m) * spacing_m
    y1 = math.ceil(maxy / spacing_m) * spacing_m

    xs = np.arange(x0, x1 + 0.5 * spacing_m, spacing_m, dtype=np.float64)
    ys = np.arange(y0, y1 + 0.5 * spacing_m, spacing_m, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    grid = pd.DataFrame(
        {
            "x_m": xx.ravel(),
            "y_m": yy.ravel(),
        }
    )
    grid["grid_x"] = np.rint(grid["x_m"] / spacing_m).astype(np.int64)
    grid["grid_y"] = np.rint(grid["y_m"] / spacing_m).astype(np.int64)
    return grid


def stable_rank_key(text: str, seed: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def make_region_id(sample_prefix: str, spacing_km: float, grid_x: int, grid_y: int) -> str:
    return f"{sample_prefix}_{int(round(spacing_km)):03d}km_x{grid_x:+07d}_y{grid_y:+07d}".replace("+", "p").replace("-", "m")


def distribute_quota(
    available_counts: pd.Series,
    *,
    target_total: int,
    equal_fraction: float,
) -> dict[str, int]:
    counts = available_counts[available_counts > 0].sort_index()
    if counts.empty:
        return {}
    if target_total <= 0 or target_total >= int(counts.sum()):
        return {str(bucket): int(value) for bucket, value in counts.items()}

    quotas = {str(bucket): 0 for bucket in counts.index}
    bucket_ids = list(counts.index)
    equal_total = min(target_total, int(round(target_total * equal_fraction)))
    if equal_total > 0:
        base = equal_total // len(bucket_ids)
        remainder = equal_total % len(bucket_ids)
        for index, bucket in enumerate(bucket_ids):
            want = base + (1 if index < remainder else 0)
            take = min(int(counts.loc[bucket]), want)
            quotas[str(bucket)] += take

    def remaining_capacity() -> pd.Series:
        return counts - pd.Series(quotas)

    remaining = target_total - sum(quotas.values())
    while remaining > 0:
        capacity = remaining_capacity()
        capacity = capacity[capacity > 0]
        if capacity.empty:
            break

        weights = capacity.astype(float) / float(capacity.sum())
        raw = weights * remaining
        floor = np.floor(raw).astype(int)
        assigned = int(floor.sum())
        for bucket, value in floor.items():
            if value > 0:
                quotas[str(bucket)] += int(value)

        leftovers = remaining - assigned
        if leftovers > 0:
            fractional = (raw - floor).sort_values(ascending=False)
            for bucket in fractional.index:
                key = str(bucket)
                if leftovers <= 0:
                    break
                if quotas[key] >= int(counts.loc[bucket]):
                    continue
                quotas[key] += 1
                leftovers -= 1

        new_remaining = target_total - sum(quotas.values())
        if new_remaining == remaining:
            # If rounding made no progress, fill greedily by largest remaining capacity.
            greedy = capacity.sort_values(ascending=False)
            for bucket in greedy.index:
                key = str(bucket)
                if new_remaining <= 0:
                    break
                if quotas[key] >= int(counts.loc[bucket]):
                    continue
                quotas[key] += 1
                new_remaining -= 1
        remaining = target_total - sum(quotas.values())

    return quotas


def build_selected_region_boxes(
    selected: gpd.GeoDataFrame,
    *,
    region_side_m: float,
    projected_crs: str,
) -> gpd.GeoDataFrame:
    half_side = 0.5 * region_side_m
    boxes = [
        box(x - half_side, y - half_side, x + half_side, y + half_side)
        for x, y in zip(selected["x_m"], selected["y_m"], strict=True)
    ]
    box_gdf = gpd.GeoDataFrame(selected.drop(columns=["geometry"]).copy(), geometry=boxes, crs=projected_crs)
    return box_gdf.to_crs(WGS84)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    land_path = Path(args.land_path) if args.land_path else detect_default_land_path()
    if land_path is None or not land_path.exists():
        raise SystemExit(
            "No land polygon file found. Pass --land-path with a global land/country vector file."
        )

    land_wgs84 = load_land_polygons(land_path, exclude_antarctica=bool(args.exclude_antarctica))
    land_projected = land_wgs84.to_crs(args.projected_crs)

    spacing_m = float(args.region_spacing_km) * 1000.0
    region_side_m = float(args.region_side_km) * 1000.0
    grid = generate_equal_area_grid(tuple(land_projected.total_bounds), spacing_m)
    point_gdf = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid["x_m"], grid["y_m"]),
        crs=args.projected_crs,
    )

    join_cols = ["continent_bucket", "country_bucket", "geometry"]
    joined = gpd.sjoin(point_gdf, land_projected[join_cols], how="inner", predicate="intersects")
    if joined.empty:
        raise SystemExit("No land candidate points were generated. Check the land polygons and CRS.")

    joined = joined.sort_values(["grid_y", "grid_x", "continent_bucket", "country_bucket"]).drop_duplicates(
        subset=["grid_x", "grid_y"],
        keep="first",
    )
    joined = joined.drop(columns=[column for column in ["index_right"] if column in joined.columns])

    centers_wgs84 = joined.to_crs(WGS84)
    joined["center_lon"] = centers_wgs84.geometry.x.astype(np.float64)
    joined["center_lat"] = centers_wgs84.geometry.y.astype(np.float64)
    joined["latitude_band"] = joined["center_lat"].map(
        lambda lat: make_latitude_band(float(lat), float(args.latitude_band_width_deg))
    )
    joined["bucket_id"] = joined["continent_bucket"] + "__" + joined["latitude_band"]
    joined["region_id"] = [
        make_region_id(str(args.sample_prefix), float(args.region_spacing_km), int(grid_x), int(grid_y))
        for grid_x, grid_y in zip(joined["grid_x"], joined["grid_y"], strict=True)
    ]
    joined["stable_rank"] = joined["region_id"].map(lambda value: stable_rank_key(str(value), str(args.selection_seed)))

    available_counts = joined["bucket_id"].value_counts().sort_index()
    quotas = distribute_quota(
        available_counts,
        target_total=int(args.target_total_regions),
        equal_fraction=float(args.equal_bucket_fraction),
    )

    selected_frames: list[gpd.GeoDataFrame] = []
    for bucket_id, quota in sorted(quotas.items()):
        bucket_frame = joined[joined["bucket_id"] == bucket_id].sort_values(["stable_rank", "region_id"])
        selected_frames.append(bucket_frame.head(int(quota)).copy())

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else joined.head(0).copy()
    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=args.projected_crs)
    region_boxes_wgs84 = build_selected_region_boxes(
        selected,
        region_side_m=region_side_m,
        projected_crs=str(args.projected_crs),
    )
    bounds = region_boxes_wgs84.bounds

    output = pd.DataFrame(
        {
            "region_id": region_boxes_wgs84["region_id"].astype(str),
            "continent_bucket": region_boxes_wgs84["continent_bucket"].astype(str),
            "country_bucket": region_boxes_wgs84["country_bucket"].astype(str),
            "latitude_band": region_boxes_wgs84["latitude_band"].astype(str),
            "bucket_id": region_boxes_wgs84["bucket_id"].astype(str),
            "center_latitude": region_boxes_wgs84["center_lat"].astype(float),
            "center_longitude": region_boxes_wgs84["center_lon"].astype(float),
            "min_lon": bounds["minx"].astype(float),
            "min_lat": bounds["miny"].astype(float),
            "max_lon": bounds["maxx"].astype(float),
            "max_lat": bounds["maxy"].astype(float),
            "grid_x": region_boxes_wgs84["grid_x"].astype(np.int64),
            "grid_y": region_boxes_wgs84["grid_y"].astype(np.int64),
            "region_side_km": float(args.region_side_km),
            "region_spacing_km": float(args.region_spacing_km),
            "projected_crs": str(args.projected_crs),
            "land_source": str(land_path),
            "selection_seed": str(args.selection_seed),
        }
    ).sort_values("region_id").reset_index(drop=True)

    write_table(output_path, output)

    summary = {
        "output_path": str(output_path),
        "land_path": str(land_path),
        "available_land_centers": int(len(joined)),
        "selected_region_count": int(len(output)),
        "region_side_km": float(args.region_side_km),
        "region_spacing_km": float(args.region_spacing_km),
        "equal_bucket_fraction": float(args.equal_bucket_fraction),
        "bucket_count": int(output["bucket_id"].nunique()),
        "continent_counts": {
            key: int(value)
            for key, value in output["continent_bucket"].value_counts().sort_index().items()
        },
        "bucket_counts_preview": {
            key: int(value)
            for key, value in output["bucket_id"].value_counts().sort_index().head(12).items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
