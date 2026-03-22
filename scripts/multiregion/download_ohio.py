#!/usr/bin/env python3
"""Download public Ohio ODNR oil and gas datasets (horizontal wells and production)."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"

WELLS_LAYER_URL = (
    "https://gis.ohiodnr.gov/arcgis/rest/services/DOG_Services/Oilgas_Wells_public/MapServer/4/query"
)
# Ohio production is not available via GIS API; wells-only for now
PRODUCTION_LAYER_URL = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=180) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=180) as response:
        return response.read()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"downloads": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def item_key(item: dict) -> tuple:
    dataset = item.get("dataset")
    return (dataset,)


def merge_downloads(existing: list[dict], updates: list[dict]) -> list[dict]:
    merged = {item_key(item): item for item in existing}
    for item in updates:
        merged[item_key(item)] = item
    return sorted(merged.values(), key=lambda item: item_key(item))


def persist_manifest(manifest_path: Path, existing: list[dict], updates: list[dict]) -> None:
    write_manifest(
        manifest_path,
        {
            "downloaded_at_utc": utc_now(),
            "downloads": merge_downloads(existing, updates),
        },
    )


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("._") or "file"


def fetch_arcgis_geojson(
    layer_url: str,
    chunk_size: int,
    where: str = "1=1",
) -> tuple[dict, int]:
    """Page through an ArcGIS REST endpoint and return a merged GeoJSON FeatureCollection."""
    all_features: list[dict] = []
    offset = 0
    while True:
        params = urlencode(
            {
                "where": where,
                "outFields": "*",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": chunk_size,
                "outSR": 4326,
                "f": "geojson",
            }
        )
        payload = json.loads(fetch_text(f"{layer_url}?{params}"))
        features = payload.get("features", [])
        if not features:
            break
        all_features.extend(features)
        if len(features) < chunk_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(features)

    return {"type": "FeatureCollection", "features": all_features}, len(all_features)


def download_wells(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
    skip_existing: bool,
) -> list[dict]:
    """Download ODNR horizontal well data via ArcGIS GeoJSON pagination."""
    outputs: list[dict] = []
    wells_path = out_dir / "oh_odnr_horizontal_wells.geojson"

    if skip_existing and wells_path.exists():
        existing_item = next(
            (
                item
                for item in existing_downloads
                if item.get("dataset") == "oh_odnr_horizontal_wells"
            ),
            None,
        )
        if existing_item:
            reused = dict(existing_item)
            reused["action"] = "reused"
            outputs.append(reused)
            persist_manifest(manifest_path, existing_downloads, outputs)
            return outputs

    wells_payload, wells_count = fetch_arcgis_geojson(
        WELLS_LAYER_URL,
        chunk_size=chunk_size,
        where="SLANT='H'",
    )
    wells_path.write_text(json.dumps(wells_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "oh_odnr_horizontal_wells",
            "source_url": WELLS_LAYER_URL,
            "output_path": str(wells_path),
            "feature_count": wells_count,
            "action": "downloaded",
        }
    )
    persist_manifest(manifest_path, existing_downloads, outputs)
    return outputs


def download_production(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
    skip_existing: bool,
) -> list[dict]:
    """Download ODNR production data for horizontal wells via ArcGIS GeoJSON pagination."""
    outputs: list[dict] = []
    production_path = out_dir / "oh_odnr_production.geojson"

    if skip_existing and production_path.exists():
        existing_item = next(
            (
                item
                for item in existing_downloads
                if item.get("dataset") == "oh_odnr_production"
            ),
            None,
        )
        if existing_item:
            reused = dict(existing_item)
            reused["action"] = "reused"
            outputs.append(reused)
            persist_manifest(manifest_path, existing_downloads, outputs)
            return outputs

    if PRODUCTION_LAYER_URL is None:
        print("Ohio production GIS API not available; skipping production download.")
        return outputs
    production_payload, production_count = fetch_arcgis_geojson(
        PRODUCTION_LAYER_URL,
        chunk_size=chunk_size,
        where="1=1",
    )
    production_path.write_text(json.dumps(production_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "oh_odnr_production",
            "source_url": PRODUCTION_LAYER_URL,
            "output_path": str(production_path),
            "feature_count": production_count,
            "action": "downloaded",
        }
    )
    persist_manifest(manifest_path, existing_downloads, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Ohio ODNR horizontal well and production data.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/oh_odnr",
        help="Directory for downloaded OH ODNR files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="ArcGIS page size for GeoJSON pagination.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing files already present in the manifest.",
    )
    parser.add_argument("--wells-only", action="store_true")
    parser.add_argument("--production-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "download_manifest.json"
    manifest = load_manifest(manifest_path)
    existing_downloads = manifest.get("downloads", [])

    downloads: list[dict] = []
    if not args.production_only:
        downloads.extend(
            download_wells(
                out_dir=out_dir,
                manifest_path=manifest_path,
                existing_downloads=existing_downloads,
                chunk_size=args.chunk_size,
                skip_existing=args.skip_existing,
            )
        )

    if not args.wells_only:
        downloads.extend(
            download_production(
                out_dir=out_dir,
                manifest_path=manifest_path,
                existing_downloads=existing_downloads,
                chunk_size=args.chunk_size,
                skip_existing=args.skip_existing,
            )
        )

    persist_manifest(manifest_path, existing_downloads, downloads)
    print(json.dumps({"downloaded_at_utc": utc_now(), "downloads": downloads}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
