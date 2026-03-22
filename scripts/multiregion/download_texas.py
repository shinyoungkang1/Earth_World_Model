#!/usr/bin/env python3
"""Download public Texas RRC oil and gas datasets.

One script serves all three TX basins (Permian, Eagle Ford, Haynesville).
Data source: Texas Railroad Commission (RRC) ArcGIS REST services.

Wells endpoint:
    https://gis.rrc.texas.gov/arcgis/rest/services/public/RRC_GISViewer/MapServer/0/query

Production endpoint:
    https://gis.rrc.texas.gov/arcgis/rest/services/public/RRC_GISViewer/MapServer/1/query
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; mineral-gas-locator/1.0)"

WELLS_LAYER_URL = (
    "https://gis.rrc.texas.gov/server/rest/services/rrc_public/RRC_Public_Viewer_Srvs/MapServer/9/query"
)
# TX production is not available via GIS API; use RRC bulk data downloads separately
PRODUCTION_LAYER_URL = None


# ---------------------------------------------------------------------------
# Utility helpers (mirror download_wv_dep.py patterns)
# ---------------------------------------------------------------------------

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
    if dataset == "tx_rrc_production":
        return (dataset, item.get("year"))
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


# ---------------------------------------------------------------------------
# ArcGIS GeoJSON pagination
# ---------------------------------------------------------------------------

def fetch_arcgis_geojson(
    layer_url: str,
    chunk_size: int,
    where: str = "1=1",
) -> tuple[dict, int]:
    """Page through an ArcGIS REST endpoint and return a GeoJSON FeatureCollection."""
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


# ---------------------------------------------------------------------------
# Wells download
# ---------------------------------------------------------------------------

def download_wells(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
) -> list[dict]:
    """Download horizontal well completion data from the RRC GIS API."""
    outputs: list[dict] = []
    wells_path = out_dir / "tx_rrc_horizontal_wells.geojson"

    wells_payload, wells_count = fetch_arcgis_geojson(
        WELLS_LAYER_URL,
        chunk_size=chunk_size,
        where="1=1",  # Layer 9 is already filtered to horizontal/directional wells
    )
    wells_path.write_text(json.dumps(wells_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "tx_rrc_horizontal_wells",
            "source_url": WELLS_LAYER_URL,
            "output_path": str(wells_path),
            "feature_count": wells_count,
            "action": "downloaded",
        }
    )
    persist_manifest(manifest_path, existing_downloads, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Production download
# ---------------------------------------------------------------------------

def _build_year_where(year: int) -> str:
    """Build an ArcGIS WHERE clause for a single production year."""
    return f"CYCLE_YEAR={year}"


def download_production(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
    start_year: int,
    skip_existing: bool,
) -> list[dict]:
    """Download monthly production records from the RRC GIS API, year by year."""
    current_year = datetime.now(timezone.utc).year
    production_dir = out_dir / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    existing_by_year: dict[int, dict] = {
        item.get("year"): item
        for item in existing_downloads
        if item.get("dataset") == "tx_rrc_production" and item.get("year")
    }

    outputs: list[dict] = []
    for year in range(start_year, current_year + 1):
        existing = existing_by_year.get(year)
        if skip_existing and existing and Path(existing["output_path"]).exists():
            reused = dict(existing)
            reused["action"] = "reused"
            outputs.append(reused)
            persist_manifest(manifest_path, existing_downloads, outputs)
            continue

        if PRODUCTION_LAYER_URL is None:
            print("TX production GIS API not available; skipping production download.")
            return outputs
        where = _build_year_where(year)
        prod_payload, feat_count = fetch_arcgis_geojson(
            PRODUCTION_LAYER_URL,
            chunk_size=chunk_size,
            where=where,
        )
        filename = f"tx_rrc_production_{year}.geojson"
        output_path = production_dir / filename
        output_path.write_text(json.dumps(prod_payload), encoding="utf-8")
        outputs.append(
            {
                "dataset": "tx_rrc_production",
                "year": year,
                "source_url": PRODUCTION_LAYER_URL,
                "output_path": str(output_path),
                "feature_count": feat_count,
                "action": "downloaded",
            }
        )
        persist_manifest(manifest_path, existing_downloads, outputs)

    return outputs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Texas RRC well and production data for all TX basins.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/tx_rrc",
        help="Directory for downloaded TX RRC files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="ArcGIS page size for GeoJSON pagination.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Earliest production year to download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing production files already present in the manifest.",
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
            )
        )

    if not args.wells_only:
        downloads.extend(
            download_production(
                out_dir=out_dir,
                manifest_path=manifest_path,
                existing_downloads=existing_downloads,
                chunk_size=args.chunk_size,
                start_year=args.start_year,
                skip_existing=args.skip_existing,
            )
        )

    persist_manifest(manifest_path, existing_downloads, downloads)
    print(json.dumps({"downloaded_at_utc": utc_now(), "downloads": downloads}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
