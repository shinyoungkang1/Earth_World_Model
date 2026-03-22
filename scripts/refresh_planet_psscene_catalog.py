#!/usr/bin/env python3
"""Refresh a bounded Planet PSScene catalog and generate non-submitted order drafts."""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


REPO_ROOT = Path("/home/shin/Mineral_Gas_Locator")
PLANET_QUICK_SEARCH_URL = "https://api.planet.com/data/v1/quick-search"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


def derive_bbox(repo_root: Path, padding_deg: float) -> tuple[float, float, float, float]:
    wells = pd.read_csv(
        repo_root / "data" / "canonical" / "pa_mvp" / "wells.csv",
        usecols=["longitude_decimal", "latitude_decimal"],
    ).dropna()
    minx = float(wells["longitude_decimal"].min()) - padding_deg
    miny = float(wells["latitude_decimal"].min()) - padding_deg
    maxx = float(wells["longitude_decimal"].max()) + padding_deg
    maxy = float(wells["latitude_decimal"].max()) + padding_deg
    return (minx, miny, maxx, maxy)


def auth_headers() -> dict[str, str]:
    api_key = os.getenv("PLANETLAB_API_KEY")
    if not api_key:
        raise RuntimeError("PLANETLAB_API_KEY missing")
    basic = base64.b64encode(f"{api_key}:".encode()).decode()
    return {
        "Authorization": f"Basic {basic}",
        "User-Agent": "codex-cli/1.0",
        "Content-Type": "application/json",
    }


def search_planet(body: dict, page_size: int) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        f"{PLANET_QUICK_SEARCH_URL}?_page_size={page_size}&_sort=acquired%20desc",
        data=payload,
        headers=auth_headers(),
        method="POST",
    )
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def feature_cloud(feature: dict) -> float:
    value = (feature.get("properties") or {}).get("cloud_cover")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 999.0


def feature_acquired(feature: dict) -> datetime:
    value = (feature.get("properties") or {}).get("acquired")
    parsed = parse_datetime(value)
    return parsed or datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_order_draft(
    name: str,
    item_ids: list[str],
    product_bundle: str,
    aoi: dict,
    include_file_format: bool,
) -> dict:
    tools = [{"clip": {"aoi": aoi}}]
    if include_file_format:
        tools.append({"file_format": {"format": "COG"}})
    return {
        "name": name,
        "source_type": "scenes",
        "order_type": "partial",
        "products": [
            {
                "item_ids": item_ids,
                "item_type": "PSScene",
                "product_bundle": product_bundle,
            }
        ],
        "tools": tools,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--bootstrap-days", type=int, default=30)
    parser.add_argument("--overlap-days", type=int, default=3)
    parser.add_argument("--max-cloud-cover", type=float, default=0.2)
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--max-order-items", type=int, default=25)
    parser.add_argument("--padding-deg", type=float, default=0.15)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    load_env_file(repo_root / ".env")
    output_dir = repo_root / "data" / "raw" / "planet_catalog" / "planet_psscene_catalog"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "catalog_items.json"
    manifest_path = output_dir / "download_manifest.json"

    existing_items: dict[str, dict] = {}
    if catalog_path.exists():
        payload = load_json(catalog_path)
        existing_items = {feature["id"]: feature for feature in payload.get("features", [])}

    latest_seen = None
    if existing_items:
        latest_seen = max(feature_acquired(feature) for feature in existing_items.values())

    now = datetime.now(timezone.utc)
    if latest_seen is None:
        query_start = now - timedelta(days=args.bootstrap_days)
    else:
        query_start = latest_seen - timedelta(days=args.overlap_days)
    query_end = now

    bbox = derive_bbox(repo_root, args.padding_deg)
    aoi = bbox_polygon(*bbox)
    body = {
        "item_types": ["PSScene"],
        "geometry": aoi,
        "filter": {
            "type": "AndFilter",
            "config": [
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {
                        "gte": query_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lte": query_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                },
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {"lte": args.max_cloud_cover},
                },
            ],
        },
    }
    payload = search_planet(body, args.page_size)
    fetched = payload.get("features", [])
    merged = {**existing_items}
    for feature in fetched:
        merged[feature["id"]] = feature

    merged_features = sorted(
        merged.values(),
        key=lambda feature: feature_acquired(feature),
        reverse=True,
    )
    catalog_payload = {
        "type": "FeatureCollection",
        "features": merged_features,
    }
    catalog_path.write_text(json.dumps(catalog_payload, indent=2), encoding="utf-8")

    selected_for_draft = sorted(
        merged_features,
        key=lambda feature: (feature_cloud(feature), -feature_acquired(feature).timestamp()),
    )[: args.max_order_items]
    item_ids = [feature["id"] for feature in selected_for_draft]

    orders_dir = repo_root / "data" / "raw" / "planet_orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    visual_draft = build_order_draft(
        "subterra-pa-psscene-visual-draft",
        item_ids,
        "visual",
        aoi,
        include_file_format=False,
    )
    analytic_draft = build_order_draft(
        "subterra-pa-psscene-analytic-draft",
        item_ids,
        "analytic_8b_udm2,analytic_udm2",
        aoi,
        include_file_format=True,
    )
    visual_path = orders_dir / "planet_psscene_visual_order_draft.json"
    analytic_path = orders_dir / "planet_psscene_analytic_order_draft.json"
    visual_path.write_text(json.dumps(visual_draft, indent=2), encoding="utf-8")
    analytic_path.write_text(json.dumps(analytic_draft, indent=2), encoding="utf-8")

    manifest = {
        "checked_at_utc": utc_now(),
        "source_id": "planet_psscene_catalog",
        "item_type": "PSScene",
        "bbox": list(bbox),
        "query_start": query_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query_end": query_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_cloud_cover": args.max_cloud_cover,
        "page_size": args.page_size,
        "fetched_item_count": len(fetched),
        "new_item_count": sum(1 for feature in fetched if feature["id"] not in existing_items),
        "total_catalog_count": len(merged_features),
        "latest_acquired": (
            feature_acquired(merged_features[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
            if merged_features
            else None
        ),
        "catalog_items_path": str(catalog_path),
        "order_draft_paths": [str(visual_path), str(analytic_path)],
        "draft_item_count": len(item_ids),
        "draft_item_ids": item_ids,
        "action": "incremental_refresh",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
