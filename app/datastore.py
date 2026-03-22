"""In-memory access to canonical PA MVP datasets for the local API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import settings


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def geometry_bbox(geometry: dict | None) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None
    coords = geometry.get("coordinates")
    if coords is None:
        return None
    xs: list[float] = []
    ys: list[float] = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                xs.append(float(node[0]))
                ys.append(float(node[1]))
            else:
                for child in node:
                    walk(child)

    walk(coords)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must contain 4 comma-separated numbers")
    return (parts[0], parts[1], parts[2], parts[3])


def bbox_polygon(minx: float, miny: float, maxx: float, maxy: float) -> dict:
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


class DataStore:
    def __init__(self) -> None:
        repo_root = settings()["repo_root"]
        canonical = Path(repo_root) / "data" / "canonical" / "pa_mvp"
        derived = Path(repo_root) / "data" / "derived" / "prospect_layers"
        basin_derived = Path(repo_root) / "data" / "derived" / "swpa_core_washington_greene"
        basin_models = Path(repo_root) / "models" / "swpa_core_washington_greene"
        basin_config_path = Path(repo_root) / "config" / "basins" / "swpa_core_washington_greene.json"
        self._wells = pd.read_csv(canonical / "wells.csv")
        self._permits = pd.read_csv(canonical / "permits.csv")
        self._sentinel = pd.read_csv(canonical / "raster_scenes_sentinel2.csv")
        self._landsat = pd.read_csv(canonical / "raster_scenes_landsat.csv")
        planet_geojson_path = canonical / "planet_scenes_psscene.geojson"
        self._planet_scene_features = []
        self._gas_prospect_features = []
        self._gas_prospect_v1_features = []
        self._gas_prospect_prithvi_v1_features = []
        self._gas_top_prospects_v1 = pd.DataFrame()
        self._gas_top_prospects_prithvi_v1 = pd.DataFrame()
        self._gas_model_v1_metrics = {}
        self._gas_model_prithvi_v1_metrics = {}
        self._gas_prospect_v1_metadata = {}
        self._gas_prospect_prithvi_v1_metadata = {}
        self._basin_config = json.loads(basin_config_path.read_text(encoding="utf-8")) if basin_config_path.exists() else {}
        if planet_geojson_path.exists():
            planet_payload = json.loads(planet_geojson_path.read_text(encoding="utf-8"))
            for feature in planet_payload.get("features", []):
                bbox = geometry_bbox(feature.get("geometry"))
                if bbox is None:
                    continue
                item = dict(feature)
                item["_bbox"] = bbox
                self._planet_scene_features.append(item)
        composite_path = canonical / "raster_composites.csv"
        self._composites = pd.read_csv(composite_path) if composite_path.exists() else pd.DataFrame()
        self._registry = json.loads((canonical / "source_registry.json").read_text(encoding="utf-8"))

        geology_payload = json.loads((canonical / "bedrock_geology.geojson").read_text(encoding="utf-8"))
        self._geology_features = []
        for feature in geology_payload.get("features", []):
            bbox = geometry_bbox(feature.get("geometry"))
            if bbox is None:
                continue
            item = dict(feature)
            item["_bbox"] = bbox
            self._geology_features.append(item)

        prospect_path = derived / "gas_prospect_cells_v0.geojson"
        if prospect_path.exists():
            prospect_payload = json.loads(prospect_path.read_text(encoding="utf-8"))
            for feature in prospect_payload.get("features", []):
                bbox = geometry_bbox(feature.get("geometry"))
                if bbox is None:
                    continue
                item = dict(feature)
                item["_bbox"] = bbox
                self._gas_prospect_features.append(item)
        prospect_v1_path = basin_derived / "gas_prospect_cells_v1.geojson"
        if prospect_v1_path.exists():
            prospect_v1_payload = json.loads(prospect_v1_path.read_text(encoding="utf-8"))
            for feature in prospect_v1_payload.get("features", []):
                bbox = geometry_bbox(feature.get("geometry"))
                if bbox is None:
                    continue
                item = dict(feature)
                item["_bbox"] = bbox
                self._gas_prospect_v1_features.append(item)
        top_v1_path = basin_derived / "top_prospects_v1.csv"
        if top_v1_path.exists():
            self._gas_top_prospects_v1 = pd.read_csv(top_v1_path)
        metrics_v1_path = basin_models / "gas_baseline_v1_label_recent12_ge_250000_metrics.json"
        if metrics_v1_path.exists():
            self._gas_model_v1_metrics = json.loads(metrics_v1_path.read_text(encoding="utf-8"))
        meta_v1_path = basin_derived / "gas_prospect_cells_v1_metadata.json"
        if meta_v1_path.exists():
            self._gas_prospect_v1_metadata = json.loads(meta_v1_path.read_text(encoding="utf-8"))
        prospect_prithvi_v1_path = basin_derived / "gas_prospect_cells_prithvi_v1.geojson"
        if prospect_prithvi_v1_path.exists():
            prospect_prithvi_v1_payload = json.loads(prospect_prithvi_v1_path.read_text(encoding="utf-8"))
            for feature in prospect_prithvi_v1_payload.get("features", []):
                bbox = geometry_bbox(feature.get("geometry"))
                if bbox is None:
                    continue
                item = dict(feature)
                item["_bbox"] = bbox
                self._gas_prospect_prithvi_v1_features.append(item)
        top_prithvi_v1_path = basin_derived / "top_prospects_prithvi_v1.csv"
        if top_prithvi_v1_path.exists():
            self._gas_top_prospects_prithvi_v1 = pd.read_csv(top_prithvi_v1_path)
        prithvi_metric_candidates = sorted(basin_models.glob("gas_prithvi_xgboost_v1_*_metrics.json"))
        if prithvi_metric_candidates:
            self._gas_model_prithvi_v1_metrics = json.loads(prithvi_metric_candidates[-1].read_text(encoding="utf-8"))
        meta_prithvi_v1_path = basin_derived / "gas_prospect_cells_prithvi_v1_metadata.json"
        if meta_prithvi_v1_path.exists():
            self._gas_prospect_prithvi_v1_metadata = json.loads(meta_prithvi_v1_path.read_text(encoding="utf-8"))
        self._summary = self._build_summary()

    def _build_summary(self) -> dict:
        outputs = {item["dataset"]: item for item in self._registry["outputs"]}
        return {
            "wells": outputs.get("wells", {}).get("row_count", 0),
            "production_records": outputs.get("production", {}).get("row_count", 0),
            "permits": outputs.get("permits", {}).get("row_count", 0),
            "bedrock_polygons": outputs.get("bedrock_geology", {}).get("feature_count", 0),
            "sentinel_scenes": outputs.get("raster_scenes_sentinel2", {}).get("row_count", 0),
            "landsat_scenes": outputs.get("raster_scenes_landsat", {}).get("row_count", 0),
            "planet_scenes": outputs.get("planet_scenes_psscene", {}).get("row_count", 0),
            "raster_assets": outputs.get("raster_assets", {}).get("row_count", 0),
            "raster_composites": outputs.get("raster_composites", {}).get("row_count", 0),
            "gas_prospect_cells_v0": len(self._gas_prospect_features),
            "gas_prospect_cells_v1": len(self._gas_prospect_v1_features),
            "gas_prospect_cells_prithvi_v1": len(self._gas_prospect_prithvi_v1_features),
            "normalized_at_utc": self._registry.get("normalized_at_utc"),
        }

    def summary(self) -> dict:
        return self._summary

    def layer_catalog(self) -> list[dict]:
        return [
            {"id": "wells", "kind": "point", "default": True},
            {"id": "permits", "kind": "point", "default": True},
            {"id": "geology", "kind": "polygon", "default": False},
            {"id": "sentinel_scenes", "kind": "bbox_polygon", "default": False},
            {"id": "landsat_scenes", "kind": "bbox_polygon", "default": False},
            {"id": "planet_scenes", "kind": "polygon", "default": False},
            {"id": "gas_prospect_cells_v0", "kind": "polygon", "default": False},
            {"id": "gas_prospect_cells_v1", "kind": "polygon", "default": True},
            {"id": "gas_prospect_cells_prithvi_v1", "kind": "polygon", "default": False},
            {"id": "sentinel_quicklook_composite", "kind": "image", "default": False},
        ]

    def point_geojson(
        self,
        frame: pd.DataFrame,
        lon_col: str,
        lat_col: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        prop_cols: list[str],
    ) -> dict:
        data = frame.dropna(subset=[lon_col, lat_col]).copy()
        if bbox:
            data = data[
                data[lon_col].between(bbox[0], bbox[2]) & data[lat_col].between(bbox[1], bbox[3])
            ]
        truncated = len(data) > limit
        data = data.head(limit)
        features = []
        for row in data.itertuples(index=False):
            props = {col: getattr(row, col) for col in prop_cols}
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [getattr(row, lon_col), getattr(row, lat_col)]},
                    "properties": props,
                }
            )
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def wells_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        return self.point_geojson(
            self._wells,
            "longitude_decimal",
            "latitude_decimal",
            bbox,
            limit,
            ["well_api", "operator_name", "well_status", "well_type", "county_name", "municipality_name"],
        )

    def permits_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        return self.point_geojson(
            self._permits,
            "longitude",
            "latitude",
            bbox,
            limit,
            ["authorization_id", "well_api", "operator", "permit_issued_date", "spud_date", "county", "well_type"],
        )

    def scene_geojson(
        self, frame: pd.DataFrame, source_id: str, bbox: tuple[float, float, float, float] | None, limit: int
    ) -> dict:
        data = frame.dropna(subset=["bbox_west", "bbox_south", "bbox_east", "bbox_north"]).copy()
        if bbox:
            data = data[
                ~(
                    (data["bbox_east"] < bbox[0])
                    | (data["bbox_west"] > bbox[2])
                    | (data["bbox_north"] < bbox[1])
                    | (data["bbox_south"] > bbox[3])
                )
            ]
        truncated = len(data) > limit
        data = data.head(limit)
        features = []
        for row in data.itertuples(index=False):
            features.append(
                {
                    "type": "Feature",
                    "geometry": bbox_polygon(row.bbox_west, row.bbox_south, row.bbox_east, row.bbox_north),
                    "properties": {
                        "scene_id": row.scene_id,
                        "source_id": source_id,
                        "collection": row.collection,
                        "datetime": row.datetime,
                        "cloud_cover": row.cloud_cover,
                        "platform": row.platform,
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def sentinel_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        return self.scene_geojson(self._sentinel, "satellite_sentinel2_catalog", bbox, limit)

    def landsat_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        return self.scene_geojson(self._landsat, "satellite_landsat_catalog", bbox, limit)

    def geology_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        features = []
        truncated = False
        for feature in self._geology_features:
            if bbox and not intersects(feature["_bbox"], bbox):
                continue
            item = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": feature.get("properties", {}),
            }
            features.append(item)
            if len(features) >= limit:
                truncated = True
                break
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def planet_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        features = []
        truncated = False
        for feature in self._planet_scene_features:
            if bbox and not intersects(feature["_bbox"], bbox):
                continue
            item = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": feature.get("properties", {}),
            }
            features.append(item)
            if len(features) >= limit:
                truncated = True
                break
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def gas_prospect_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        features = []
        truncated = False
        for feature in self._gas_prospect_features:
            if bbox and not intersects(feature["_bbox"], bbox):
                continue
            item = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": feature.get("properties", {}),
            }
            features.append(item)
            if len(features) >= limit:
                truncated = True
                break
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def gas_prospect_v1_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        features = []
        truncated = False
        for feature in self._gas_prospect_v1_features:
            if bbox and not intersects(feature["_bbox"], bbox):
                continue
            item = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": feature.get("properties", {}),
            }
            features.append(item)
            if len(features) >= limit:
                truncated = True
                break
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def gas_prospect_prithvi_v1_geojson(self, bbox: tuple[float, float, float, float] | None, limit: int) -> dict:
        features = []
        truncated = False
        for feature in self._gas_prospect_prithvi_v1_features:
            if bbox and not intersects(feature["_bbox"], bbox):
                continue
            item = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": feature.get("properties", {}),
            }
            features.append(item)
            if len(features) >= limit:
                truncated = True
                break
        return {"type": "FeatureCollection", "features": features, "meta": {"truncated": truncated}}

    def gas_model_v1_summary(self) -> dict:
        return {
            "basin": self._basin_config,
            "metrics": self._gas_model_v1_metrics,
            "prospect_layer": self._gas_prospect_v1_metadata,
        }

    def gas_model_prithvi_v1_summary(self) -> dict:
        return {
            "basin": self._basin_config,
            "metrics": self._gas_model_prithvi_v1_metrics,
            "prospect_layer": self._gas_prospect_prithvi_v1_metadata,
        }

    def top_prospects_v1(self, limit: int) -> list[dict]:
        if self._gas_top_prospects_v1.empty:
            return []
        rows = []
        for row in self._gas_top_prospects_v1.head(limit).itertuples(index=False):
            rows.append(
                {
                    "cell_id": row.cell_id,
                    "score_rank": int(row.score_rank),
                    "score": float(row.score),
                    "score_percentile": float(row.score_percentile),
                    "prospect_tier": row.prospect_tier,
                    "county_name": row.county_name,
                    "geology_name": row.geology_name,
                    "geology_lith1": row.geology_lith1,
                    "fault_distance_km": float(row.fault_distance_km) if pd.notna(row.fault_distance_km) else None,
                    "well_count_5km": int(row.well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                    "center_longitude": float(row.center_longitude),
                    "center_latitude": float(row.center_latitude),
                }
            )
        return rows

    def top_prospects_prithvi_v1(self, limit: int) -> list[dict]:
        if self._gas_top_prospects_prithvi_v1.empty:
            return []
        rows = []
        for row in self._gas_top_prospects_prithvi_v1.head(limit).itertuples(index=False):
            rows.append(
                {
                    "cell_id": row.sample_key,
                    "score_rank": int(row.score_rank),
                    "score": float(row.score),
                    "score_percentile": float(row.score_percentile),
                    "prospect_tier": row.prospect_tier,
                    "county_name": row.county_name,
                    "geology_name": row.geology_name,
                    "geology_lith1": row.geology_lith1,
                    "fault_distance_km": float(row.fault_distance_km) if pd.notna(row.fault_distance_km) else None,
                    "well_count_5km": int(row.well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                    "center_longitude": float(row.sample_longitude),
                    "center_latitude": float(row.sample_latitude),
                }
            )
        return rows

    def composites(self) -> list[dict]:
        if self._composites.empty:
            return []
        items = []
        for row in self._composites.itertuples(index=False):
            items.append(
                {
                    "composite_id": row.composite_id,
                    "source_id": row.source_id,
                    "collection": row.collection,
                    "composite_kind": row.composite_kind,
                    "scene_count": int(row.scene_count),
                    "image_url": f"/api/composites/{row.composite_id}/image",
                    "bbox_west": float(row.bbox_west),
                    "bbox_south": float(row.bbox_south),
                    "bbox_east": float(row.bbox_east),
                    "bbox_north": float(row.bbox_north),
                    "image_width": int(row.image_width),
                    "image_height": int(row.image_height),
                    "generated_at_utc": row.generated_at_utc,
                }
            )
        return items

    def composite_path(self, composite_id: str) -> Path | None:
        if self._composites.empty:
            return None
        matches = self._composites[self._composites["composite_id"] == composite_id]
        if matches.empty:
            return None
        return Path(matches.iloc[0]["image_path"])


@lru_cache(maxsize=1)
def datastore() -> DataStore:
    return DataStore()
