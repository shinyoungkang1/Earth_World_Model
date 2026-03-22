"""Shared helpers for the basin-scoped gas v1 pipeline."""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.strtree import STRtree
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_basin_config(repo_root: Path, basin_id: str) -> dict:
    return load_json(repo_root / "config" / "basins" / f"{basin_id}.json")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def build_geology_lookup(geology_path: Path):
    payload = load_json(geology_path)
    geoms = []
    props = []
    for feature in payload.get("features", []):
        geom = shape(feature["geometry"])
        geoms.append(geom)
        props.append(feature.get("properties", {}))
    tree = STRtree(geoms)
    return geoms, props, tree


def assign_geology(lon: float, lat: float, geoms, props, tree) -> dict:
    point = Point(float(lon), float(lat))
    for idx in tree.query(point):
        if geoms[int(idx)].covers(point):
            match = props[int(idx)]
            return {
                "geology_map_symbol": match.get("map_symbol"),
                "geology_name": match.get("name"),
                "geology_age": match.get("age"),
                "geology_lith1": match.get("lith1"),
            }
    return {
        "geology_map_symbol": None,
        "geology_name": None,
        "geology_age": None,
        "geology_lith1": None,
    }


def load_fault_index(fault_zip_path: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(fault_zip_path) as zf:
            zf.extractall(temp_dir)
        temp_path = Path(temp_dir)
        prj_text = (temp_path / "pafaults_lcc.prj").read_text(encoding="utf-8")
        fault_crs = CRS.from_wkt(prj_text)
        reader = shapefile.Reader(str(temp_path / "pafaults_lcc.shp"))
        geometries = []
        for shp in reader.iterShapes():
            points = shp.points
            parts = list(shp.parts) + [len(points)]
            segments = []
            for idx in range(len(parts) - 1):
                segment = points[parts[idx] : parts[idx + 1]]
                if len(segment) >= 2:
                    segments.append(LineString(segment))
            if len(segments) == 1:
                geometries.append(segments[0])
            elif segments:
                geometries.append(MultiLineString(segments))
    tree = STRtree(geometries)
    transformer = Transformer.from_crs(4326, fault_crs, always_xy=True)
    return geometries, tree, transformer


def nearest_fault_distance_km(
    lon: float,
    lat: float,
    fault_geometries,
    fault_tree: STRtree,
    transformer: Transformer,
) -> float:
    x, y = transformer.transform(float(lon), float(lat))
    point = Point(x, y)
    nearest_idx = int(fault_tree.nearest(point))
    return point.distance(fault_geometries[nearest_idx]) / 1000.0


def load_dem_datasets(elevation_dir: Path):
    datasets = []
    for tif_path in sorted(elevation_dir.glob("*.tif")):
        ds = rasterio.open(tif_path)
        datasets.append(ds)
    return datasets


def close_dem_datasets(datasets) -> None:
    for ds in datasets:
        ds.close()


def dataset_for_point(lon: float, lat: float, datasets):
    for ds in datasets:
        bounds = ds.bounds
        if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
            return ds
    return None


def _horn_slope(window: np.ndarray, lat: float, res_x_deg: float, res_y_deg: float) -> float:
    if window.shape != (3, 3) or np.isnan(window).any():
        return float("nan")
    dx = res_x_deg * 111320.0 * math.cos(math.radians(lat))
    dy = res_y_deg * 111320.0
    if dx <= 0 or dy <= 0:
        return float("nan")
    dzdx = ((window[0, 2] + 2 * window[1, 2] + window[2, 2]) - (window[0, 0] + 2 * window[1, 0] + window[2, 0])) / (
        8 * dx
    )
    dzdy = ((window[2, 0] + 2 * window[2, 1] + window[2, 2]) - (window[0, 0] + 2 * window[0, 1] + window[0, 2])) / (
        8 * dy
    )
    return math.degrees(math.atan(math.sqrt((dzdx**2) + (dzdy**2))))


def sample_dem_features(lon: float, lat: float, datasets) -> dict:
    ds = dataset_for_point(lon, lat, datasets)
    if ds is None:
        return {"elevation_m": np.nan, "slope_deg": np.nan, "relief_3px_m": np.nan}
    row, col = ds.index(float(lon), float(lat))
    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return {"elevation_m": np.nan, "slope_deg": np.nan, "relief_3px_m": np.nan}
    value = float(next(ds.sample([(float(lon), float(lat))]))[0])
    window = ds.read(1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan).astype(float)
    slope = _horn_slope(window, lat=float(lat), res_x_deg=abs(ds.res[0]), res_y_deg=abs(ds.res[1]))
    relief = float(np.nanmax(window) - np.nanmin(window)) if not np.isnan(window).all() else float("nan")
    return {"elevation_m": value, "slope_deg": slope, "relief_3px_m": relief}


def build_balltree(frame: pd.DataFrame, lat_col: str, lon_col: str):
    clean = frame.dropna(subset=[lat_col, lon_col]).copy().reset_index(drop=True)
    coords_rad = np.deg2rad(clean[[lat_col, lon_col]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    return clean, tree


def count_neighbors_before_date(
    target_lat: float,
    target_lon: float,
    target_date: pd.Timestamp,
    source_frame: pd.DataFrame,
    source_tree: BallTree,
    date_col: str,
    lat_col: str,
    lon_col: str,
    radius_km: float,
) -> int:
    if pd.isna(target_date):
        return 0
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs = source_tree.query_radius(coords_rad, r=radius_km / EARTH_RADIUS_KM, return_distance=False)[0]
    if len(idxs) == 0:
        return 0
    subset = source_frame.iloc[idxs]
    mask = parse_date(subset[date_col]) < target_date
    return int(mask.sum())


def nearest_neighbor_before_date_km(
    target_lat: float,
    target_lon: float,
    target_date: pd.Timestamp,
    source_frame: pd.DataFrame,
    source_tree: BallTree,
    date_col: str,
    lat_col: str,
    lon_col: str,
    max_radius_km: float = 20.0,
) -> float:
    if pd.isna(target_date):
        return float("nan")
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs, dists = source_tree.query_radius(
        coords_rad,
        r=max_radius_km / EARTH_RADIUS_KM,
        return_distance=True,
        sort_results=True,
    )
    if len(idxs[0]) == 0:
        return float("nan")
    subset = source_frame.iloc[idxs[0]].copy()
    subset["__dist_km"] = np.asarray(dists[0]) * EARTH_RADIUS_KM
    subset = subset[parse_date(subset[date_col]) < target_date]
    if subset.empty:
        return float("nan")
    return float(subset["__dist_km"].min())


def count_neighbors_current(
    target_lat: float,
    target_lon: float,
    source_tree: BallTree,
    radius_km: float,
) -> int:
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs = source_tree.query_radius(coords_rad, r=radius_km / EARTH_RADIUS_KM, return_distance=False)[0]
    return int(len(idxs))


def nearest_neighbor_current_km(
    target_lat: float,
    target_lon: float,
    source_tree: BallTree,
    exclude_self: bool = False,
    max_radius_km: float = 20.0,
) -> float:
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs, dists = source_tree.query_radius(
        coords_rad,
        r=max_radius_km / EARTH_RADIUS_KM,
        return_distance=True,
        sort_results=True,
    )
    distances_km = np.asarray(dists[0]) * EARTH_RADIUS_KM
    if exclude_self:
        distances_km = distances_km[distances_km > 1e-6]
    if len(distances_km) == 0:
        return float("nan")
    return float(distances_km.min())


def polygon_from_bbox(minx: float, miny: float, maxx: float, maxy: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def prospect_tier(percentile: float) -> str:
    if percentile >= 0.9:
        return "very_high"
    if percentile >= 0.7:
        return "high"
    if percentile >= 0.4:
        return "medium"
    return "background"
