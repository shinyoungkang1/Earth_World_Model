#!/usr/bin/env python3
"""Download public West Virginia DEP oil and gas MVP datasets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"
HORIZONTAL_WELLS_LAYER_URL = (
    "https://tagis.dep.wv.gov/arcgis/rest/services/WVDEP_enterprise/oil_gas/MapServer/0/query"
)
LATERALS_LAYER_URL = (
    "https://tagis.dep.wv.gov/arcgis/rest/services/WVDEP_enterprise/oil_gas/MapServer/6/query"
)
PRODUCTION_PAGE = "https://dep.wv.gov/oil-and-gas/databaseinfo/Pages/default.aspx"
H6A_LINK_RE = re.compile(
    r'href="(https://apps\.dep\.wv\.gov/Documents/OOG/ProductionReports/H6A_Production_Data/[^"]+\.(?:xls|xlsx|zip))"',
    re.IGNORECASE,
)


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
    if dataset == "wv_dep_h6a_production":
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


def fetch_arcgis_geojson(layer_url: str, chunk_size: int) -> tuple[dict, int]:
    all_features: list[dict] = []
    offset = 0
    while True:
        params = urlencode(
            {
                "where": "1=1",
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


def download_wells_and_laterals(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
) -> list[dict]:
    outputs = []
    wells_path = out_dir / "wv_dep_horizontal_wells.geojson"
    laterals_path = out_dir / "wv_dep_horizontal_laterals.geojson"

    wells_payload, wells_count = fetch_arcgis_geojson(HORIZONTAL_WELLS_LAYER_URL, chunk_size=chunk_size)
    wells_path.write_text(json.dumps(wells_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "wv_dep_horizontal_wells",
            "source_url": HORIZONTAL_WELLS_LAYER_URL,
            "output_path": str(wells_path),
            "feature_count": wells_count,
            "action": "downloaded",
        }
    )
    persist_manifest(manifest_path, existing_downloads, outputs)

    laterals_payload, laterals_count = fetch_arcgis_geojson(LATERALS_LAYER_URL, chunk_size=chunk_size)
    laterals_path.write_text(json.dumps(laterals_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "wv_dep_horizontal_laterals",
            "source_url": LATERALS_LAYER_URL,
            "output_path": str(laterals_path),
            "feature_count": laterals_count,
            "action": "downloaded",
        }
    )
    persist_manifest(manifest_path, existing_downloads, outputs)
    return outputs


def parse_h6a_links(page_html: str) -> list[tuple[int, str]]:
    links = []
    seen = set()
    for url in H6A_LINK_RE.findall(page_html):
        if url in seen:
            continue
        seen.add(url)
        year_match = re.search(r"(20\d{2}|19\d{2})", url)
        if not year_match:
            continue
        links.append((int(year_match.group(1)), url))
    links.sort(key=lambda item: item[0])
    return links


def download_h6a_production(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    start_year: int,
    skip_existing: bool,
) -> list[dict]:
    page_html = fetch_text(PRODUCTION_PAGE)
    links = [(year, url) for year, url in parse_h6a_links(page_html) if year >= start_year]
    existing_by_year = {
        item.get("year"): item
        for item in existing_downloads
        if item.get("dataset") == "wv_dep_h6a_production" and item.get("year")
    }

    outputs = []
    production_dir = out_dir / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    for year, url in links:
        existing = existing_by_year.get(year)
        if skip_existing and existing and Path(existing["output_path"]).exists():
            reused = dict(existing)
            reused["action"] = "reused"
            outputs.append(reused)
            persist_manifest(manifest_path, existing_downloads, outputs)
            continue

        content = fetch_bytes(url)
        filename = sanitize_filename(Path(url).name)
        output_path = production_dir / filename
        output_path.write_bytes(content)
        outputs.append(
            {
                "dataset": "wv_dep_h6a_production",
                "year": year,
                "source_url": url,
                "output_path": str(output_path),
                "size_bytes": len(content),
                "action": "downloaded",
            }
        )
        persist_manifest(manifest_path, existing_downloads, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/wv_dep",
        help="Directory for downloaded WV DEP files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="ArcGIS page size for GeoJSON pagination.",
    )
    parser.add_argument(
        "--h6a-start-year",
        type=int,
        default=2018,
        help="Earliest WV H6A year to download.",
    )
    parser.add_argument(
        "--skip-existing-production",
        action="store_true",
        help="Reuse existing H6A files already present in the manifest.",
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
            download_wells_and_laterals(
                out_dir=out_dir,
                manifest_path=manifest_path,
                existing_downloads=existing_downloads,
                chunk_size=args.chunk_size,
            )
        )

    if not args.wells_only:
        downloads.extend(
            download_h6a_production(
                out_dir=out_dir,
                manifest_path=manifest_path,
                existing_downloads=existing_downloads,
                start_year=args.h6a_start_year,
                skip_existing=args.skip_existing_production,
            )
        )

    persist_manifest(manifest_path, existing_downloads, downloads)
    print(json.dumps({"downloaded_at_utc": utc_now(), "downloads": downloads}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
