#!/usr/bin/env python3
"""Download a bounded set of Sentinel-2 quicklooks and build a preview mosaic."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


REPO_ROOT = Path("/home/shin/Mineral_Gas_Locator")


@dataclass
class SceneRecord:
    scene_id: str
    datetime_utc: datetime
    grid_code: str
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    thumbnail_href: str
    collection: str
    platform: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_scenes(catalog_path: Path) -> list[SceneRecord]:
    payload = load_json(catalog_path)
    scenes: list[SceneRecord] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        assets = feature.get("assets", {})
        thumb = assets.get("thumbnail") or {}
        href = thumb.get("href")
        dt = parse_datetime(props.get("datetime"))
        bbox = feature.get("bbox") or [None, None, None, None]
        if (
            not href
            or not href.startswith("http")
            or dt is None
            or any(value is None for value in bbox[:4])
        ):
            continue
        cloud = props.get("eo:cloud_cover")
        try:
            cloud_value = float(cloud) if cloud is not None else None
        except (TypeError, ValueError):
            cloud_value = None
        scenes.append(
            SceneRecord(
                scene_id=feature.get("id"),
                datetime_utc=dt,
                grid_code=props.get("grid:code") or feature.get("id"),
                cloud_cover=cloud_value,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                thumbnail_href=href,
                collection=feature.get("collection", "sentinel-2-l2a"),
                platform=props.get("platform"),
            )
        )
    return scenes


def scene_sort_key(scene: SceneRecord) -> tuple[float, float]:
    cloud = scene.cloud_cover if scene.cloud_cover is not None else 999.0
    return (cloud, -scene.datetime_utc.timestamp())


def select_scenes(scenes: list[SceneRecord], lookback_days: int, max_cloud_cover: float) -> list[SceneRecord]:
    if not scenes:
        return []
    latest_dt = max(scene.datetime_utc for scene in scenes)
    window_start = latest_dt - timedelta(days=lookback_days)
    filtered = [
        scene
        for scene in scenes
        if scene.datetime_utc >= window_start
        and (scene.cloud_cover is None or scene.cloud_cover <= max_cloud_cover)
    ]
    grouped: dict[str, list[SceneRecord]] = {}
    for scene in filtered:
        grouped.setdefault(scene.grid_code, []).append(scene)
    selected = []
    for _, group in sorted(grouped.items()):
        selected.append(sorted(group, key=scene_sort_key)[0])
    return selected


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "codex-cli/1.0"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def pixel_box(
    scene_bbox: tuple[float, float, float, float],
    composite_bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    west, south, east, north = scene_bbox
    c_west, c_south, c_east, c_north = composite_bbox
    x0 = int(round((west - c_west) / (c_east - c_west) * width))
    x1 = int(round((east - c_west) / (c_east - c_west) * width))
    y0 = int(round((c_north - north) / (c_north - c_south) * height))
    y1 = int(round((c_north - south) / (c_north - c_south) * height))
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def build_composite(
    records: list[dict],
    output_path: Path,
) -> dict | None:
    downloaded = [record for record in records if record["status"] in {"downloaded", "reused"}]
    if not downloaded:
        return None

    composite_bbox = (
        min(record["bbox_west"] for record in downloaded),
        min(record["bbox_south"] for record in downloaded),
        max(record["bbox_east"] for record in downloaded),
        max(record["bbox_north"] for record in downloaded),
    )
    lon_range = composite_bbox[2] - composite_bbox[0]
    lat_range = composite_bbox[3] - composite_bbox[1]
    canvas_width = 1600
    canvas_height = max(600, int(round(canvas_width * (lat_range / lon_range))))
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (248, 244, 234, 255))

    paste_order = sorted(
        downloaded,
        key=lambda item: (
            item["cloud_cover"] if item["cloud_cover"] is not None else 999.0,
            item["datetime"],
        ),
        reverse=True,
    )
    for record in paste_order:
        image = Image.open(record["local_path"]).convert("RGBA")
        image.putalpha(212)
        x0, y0, x1, y1 = pixel_box(
            (record["bbox_west"], record["bbox_south"], record["bbox_east"], record["bbox_north"]),
            composite_bbox,
            canvas_width,
            canvas_height,
        )
        resized = image.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.Resampling.BILINEAR)
        canvas.paste(resized, (x0, y0), resized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "composite_id": "sentinel2_recent_quicklook_mosaic",
        "source_id": "satellite_sentinel2_quicklooks",
        "collection": "sentinel-2-l2a",
        "composite_kind": "preview_mosaic",
        "rendering_method": "bbox_paste_quicklook",
        "scene_count": len(downloaded),
        "scene_ids": [record["scene_id"] for record in downloaded],
        "image_path": str(output_path),
        "image_width": canvas_width,
        "image_height": canvas_height,
        "bbox_west": composite_bbox[0],
        "bbox_south": composite_bbox[1],
        "bbox_east": composite_bbox[2],
        "bbox_north": composite_bbox[3],
        "generated_at_utc": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--max-cloud-cover", type=float, default=30.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    catalog_path = (
        repo_root / "data" / "raw" / "satellite_catalog" / "satellite_sentinel2_catalog" / "catalog_items.json"
    )
    output_dir = repo_root / "data" / "raw" / "satellite_assets" / "satellite_sentinel2_quicklooks"
    manifest_path = output_dir / "download_manifest.json"
    composite_image_path = (
        repo_root / "data" / "derived" / "raster_composites" / "sentinel2_recent_quicklook_mosaic.png"
    )
    scenes = parse_scenes(catalog_path)
    selected = select_scenes(scenes, args.lookback_days, args.max_cloud_cover)

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    downloaded_count = 0
    reused_count = 0
    for scene in selected:
        ext = ".jpg"
        local_path = output_dir / f"{scene.scene_id}_thumbnail{ext}"
        action = "reused"
        if not local_path.exists():
            local_path.write_bytes(fetch_bytes(scene.thumbnail_href))
            action = "downloaded"
        with Image.open(local_path) as image:
            width, height = image.size
        if action == "downloaded":
            downloaded_count += 1
        else:
            reused_count += 1
        records.append(
            {
                "scene_id": scene.scene_id,
                "grid_code": scene.grid_code,
                "datetime": scene.datetime_utc.date().isoformat(),
                "cloud_cover": scene.cloud_cover,
                "collection": scene.collection,
                "platform": scene.platform,
                "asset_kind": "thumbnail",
                "asset_role": "overview",
                "status": action,
                "href": scene.thumbnail_href,
                "local_path": str(local_path),
                "file_size_bytes": local_path.stat().st_size,
                "image_width": width,
                "image_height": height,
                "bbox_west": scene.bbox[0],
                "bbox_south": scene.bbox[1],
                "bbox_east": scene.bbox[2],
                "bbox_north": scene.bbox[3],
            }
        )

    composite = build_composite(records, composite_image_path)
    manifest = {
        "checked_at_utc": utc_now(),
        "source_id": "satellite_sentinel2_quicklooks",
        "selection_strategy": "best_recent_per_grid_code",
        "lookback_days": args.lookback_days,
        "max_cloud_cover": args.max_cloud_cover,
        "selected_scene_count": len(records),
        "downloaded_count": downloaded_count,
        "reused_count": reused_count,
        "downloads": records,
        "composites": [composite] if composite else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
