#!/usr/bin/env python3
"""Refresh a CDSE STAC scene catalog incrementally for the PA MVP."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bbox_from_inventory(path: Path, padding_degrees: float) -> list[float]:
    lats = []
    lons = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lat = row.get("LATITUDE_DECIMAL")
            lon = row.get("LONGITUDE_DECIMAL")
            if not lat or not lon:
                continue
            lats.append(float(lat))
            lons.append(float(lon))
    if not lats or not lons:
        raise RuntimeError(f"No coordinates found in {path}")
    return [
        min(lons) - padding_degrees,
        min(lats) - padding_degrees,
        max(lons) + padding_degrees,
        max(lats) + padding_degrees,
    ]


def load_existing_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("features", [])


def latest_datetime(features: list[dict]) -> datetime | None:
    seen = []
    for feature in features:
        value = feature.get("properties", {}).get("datetime")
        if value:
            seen.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
    return max(seen) if seen else None


def search_url(
    collection: str,
    bbox: list[float],
    start_dt: datetime,
    end_dt: datetime,
    cloud_cover_max: float | None,
    limit: int,
) -> str:
    params = {
        "collections": collection,
        "bbox": ",".join(str(x) for x in bbox),
        "datetime": f"{isoformat_z(start_dt)}/{isoformat_z(end_dt)}",
        "limit": str(limit),
    }
    if cloud_cover_max is not None:
        params["filter-lang"] = "cql2-text"
        params["filter"] = f"eo:cloud_cover<{cloud_cover_max}"
    return f"{STAC_SEARCH_URL}?{urlencode(params)}"


def fetch_all_pages(first_url: str) -> list[dict]:
    items = []
    url = first_url
    while url:
        payload = fetch_json(url)
        items.extend(payload.get("features", []))
        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
    return items


def merge_items(existing: list[dict], fresh: list[dict]) -> tuple[list[dict], list[str]]:
    merged = {item["id"]: item for item in existing}
    new_ids = []
    for item in fresh:
        if item["id"] not in merged:
            new_ids.append(item["id"])
        merged[item["id"]] = item
    merged_items = sorted(
        merged.values(),
        key=lambda x: x.get("properties", {}).get("datetime", ""),
        reverse=True,
    )
    return merged_items, new_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve output paths.",
    )
    parser.add_argument(
        "--inventory-csv",
        default="/home/shin/Mineral_Gas_Locator/data/raw/pa_dep/pa_dep_unconventional_well_inventory.csv",
    )
    parser.add_argument("--padding-degrees", type=float, default=0.2)
    parser.add_argument("--bootstrap-days", type=int, default=180)
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--cloud-cover-max", type=float, default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = repo_root / "data/raw/satellite_catalog" / args.source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    items_path = out_dir / "catalog_items.json"
    existing_items = load_existing_items(items_path)
    last_dt = latest_datetime(existing_items)
    end_dt = utc_now()
    if last_dt is None:
        start_dt = end_dt - timedelta(days=args.bootstrap_days)
    else:
        start_dt = last_dt - timedelta(days=args.overlap_days)

    bbox = bbox_from_inventory(Path(args.inventory_csv), args.padding_degrees)
    first_url = search_url(
        collection=args.collection,
        bbox=bbox,
        start_dt=start_dt,
        end_dt=end_dt,
        cloud_cover_max=args.cloud_cover_max,
        limit=args.limit,
    )
    fresh_items = fetch_all_pages(first_url)
    merged_items, new_ids = merge_items(existing_items, fresh_items)

    payload = {
        "type": "FeatureCollection",
        "features": merged_items,
    }
    items_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = {
        "checked_at_utc": isoformat_z(end_dt),
        "source_id": args.source_id,
        "collection": args.collection,
        "bbox": bbox,
        "query_start": isoformat_z(start_dt),
        "query_end": isoformat_z(end_dt),
        "cloud_cover_max": args.cloud_cover_max,
        "limit": args.limit,
        "fetched_item_count": len(fresh_items),
        "new_item_count": len(new_ids),
        "total_catalog_count": len(merged_items),
        "new_item_ids": new_ids,
        "last_seen_datetime": isoformat_z(latest_datetime(merged_items)) if merged_items else None,
        "catalog_items_path": str(items_path),
        "action": "incremental_refresh",
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
