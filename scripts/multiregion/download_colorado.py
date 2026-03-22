#!/usr/bin/env python3
"""Download public Colorado COGCC oil and gas datasets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"
# Colorado COGIS database is at ecmc.state.co.us/cogisdb/ (ASP.NET web app, not ArcGIS REST).
# Well facility search: https://ecmc.state.co.us/cogisdb/Facility/FacilitySearch.aspx
# Production search: https://ecmc.state.co.us/cogisdb/Production/ProdSearch.aspx
# TODO: Implement form-based session download from COGIS database.
# For now, use the ECMC data portal if available.
WELLS_LAYER_URL = None
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
    if dataset == "co_cogcc_production":
        return (dataset, item.get("chunk_offset"))
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


def download_wells(
    out_dir: Path,
    manifest_path: Path,
    existing_downloads: list[dict],
    chunk_size: int,
) -> list[dict]:
    """Download COGCC horizontal well data via ArcGIS REST."""
    outputs: list[dict] = []
    wells_path = out_dir / "co_cogcc_wells.geojson"

    wells_payload, wells_count = fetch_arcgis_geojson(
        WELLS_LAYER_URL,
        chunk_size=chunk_size,
        where="Facil_Type='WELL' AND Dir_Drill='Horizontal'",
    )
    wells_path.write_text(json.dumps(wells_payload), encoding="utf-8")
    outputs.append(
        {
            "dataset": "co_cogcc_wells",
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
    start_year: int,
    skip_existing: bool,
) -> list[dict]:
    """Download COGCC monthly production data via ArcGIS REST, page by page."""
    outputs: list[dict] = []
    production_dir = out_dir / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    where = f"Prod_Year>={start_year}"

    existing_by_offset = {
        item.get("chunk_offset"): item
        for item in existing_downloads
        if item.get("dataset") == "co_cogcc_production" and item.get("chunk_offset") is not None
    }

    offset = 0
    while True:
        if skip_existing and offset in existing_by_offset:
            existing = existing_by_offset[offset]
            if Path(existing["output_path"]).exists():
                reused = dict(existing)
                reused["action"] = "reused"
                outputs.append(reused)
                persist_manifest(manifest_path, existing_downloads, outputs)
                offset += existing.get("feature_count", chunk_size)
                continue

        params = urlencode(
            {
                "where": where,
                "outFields": "*",
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": chunk_size,
                "f": "json",
            }
        )
        raw = fetch_text(f"{PRODUCTION_LAYER_URL}?{params}")
        payload = json.loads(raw)
        features = payload.get("features", [])
        if not features:
            break

        chunk_file = production_dir / f"co_cogcc_production_{offset:08d}.json"
        chunk_file.write_text(json.dumps(payload), encoding="utf-8")
        outputs.append(
            {
                "dataset": "co_cogcc_production",
                "source_url": PRODUCTION_LAYER_URL,
                "chunk_offset": offset,
                "output_path": str(chunk_file),
                "feature_count": len(features),
                "action": "downloaded",
            }
        )
        persist_manifest(manifest_path, existing_downloads, outputs)

        if len(features) < chunk_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(features)

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Colorado COGCC oil & gas datasets.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/co_cogcc",
        help="Directory for downloaded COGCC files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="ArcGIS page size for pagination.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Earliest production year to download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse production chunks already present in the manifest.",
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
