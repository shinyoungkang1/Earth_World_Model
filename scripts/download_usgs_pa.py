#!/usr/bin/env python3
"""Download public USGS layers for the Pennsylvania MVP footprint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"

FAULT_GIS_ZIP = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip"
FAULT_KMZ = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/qfaults.kmz"
ELEVATION_BASE = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation"


def fetch_to_path(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response:
        path.write_bytes(response.read())


def tile_id(lat_floor: int, lon_floor: int) -> str:
    lat_prefix = "n" if lat_floor >= 0 else "s"
    lon_prefix = "e" if lon_floor >= 0 else "w"
    return f"{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}"


def pa_tiles_from_inventory(path: Path) -> list[str]:
    tiles = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = row.get("LATITUDE_DECIMAL")
            lon = row.get("LONGITUDE_DECIMAL")
            if not lat or not lon:
                continue
            lat_floor = math.floor(float(lat))
            lon_floor = math.floor(float(lon))
            tiles.add(tile_id(lat_floor, lon_floor))
    return sorted(tiles)


def elevation_urls(tile: str, arc: str) -> dict[str, str]:
    return {
        "tif": f"{ELEVATION_BASE}/{arc}/TIFF/current/{tile}/USGS_{arc}_{tile}.tif",
        "xml": f"{ELEVATION_BASE}/{arc}/TIFF/current/{tile}/USGS_{arc}_{tile}.xml",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-csv",
        default="/home/shin/Mineral_Gas_Locator/data/raw/pa_dep/pa_dep_unconventional_well_inventory.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/usgs",
    )
    parser.add_argument(
        "--arc",
        default="1",
        choices=["1", "13"],
        help="Elevation resolution directory: 1=1 arc-second, 13=1/3 arc-second.",
    )
    args = parser.parse_args()

    inventory_csv = Path(args.inventory_csv)
    out_dir = Path(args.out_dir)
    dem_dir = out_dir / f"elevation_{args.arc}"
    dem_dir.mkdir(parents=True, exist_ok=True)

    tiles = pa_tiles_from_inventory(inventory_csv)
    downloads: list[dict[str, str]] = []

    fault_zip_path = out_dir / "qfault_gis.zip"
    fault_zip_action = "reused"
    if not fault_zip_path.exists():
        fetch_to_path(FAULT_GIS_ZIP, fault_zip_path)
        fault_zip_action = "downloaded"
    downloads.append(
        {"dataset": "qfault_gis_zip", "output_path": str(fault_zip_path), "action": fault_zip_action}
    )

    fault_kmz_path = out_dir / "qfault.kmz"
    fault_kmz_action = "reused"
    if not fault_kmz_path.exists():
        fetch_to_path(FAULT_KMZ, fault_kmz_path)
        fault_kmz_action = "downloaded"
    downloads.append(
        {"dataset": "qfault_kmz", "output_path": str(fault_kmz_path), "action": fault_kmz_action}
    )

    for tile in tiles:
        urls = elevation_urls(tile, args.arc)
        tif_path = dem_dir / f"USGS_{args.arc}_{tile}.tif"
        xml_path = dem_dir / f"USGS_{args.arc}_{tile}.xml"
        tif_action = "reused"
        xml_action = "reused"
        if not tif_path.exists():
            fetch_to_path(urls["tif"], tif_path)
            tif_action = "downloaded"
        if not xml_path.exists():
            fetch_to_path(urls["xml"], xml_path)
            xml_action = "downloaded"
        downloads.append(
            {
                "dataset": f"usgs_elevation_{args.arc}",
                "tile": tile,
                "tif_path": str(tif_path),
                "xml_path": str(xml_path),
                "tif_action": tif_action,
                "xml_action": xml_action,
            }
        )

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_csv": str(inventory_csv),
        "arc": args.arc,
        "tiles": tiles,
        "downloads": downloads,
    }

    manifest_path = out_dir / f"download_manifest_arc{args.arc}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"download failed: {exc}", file=sys.stderr)
        raise
