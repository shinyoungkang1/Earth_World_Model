#!/usr/bin/env python3
"""Download paginated public ArcGIS layers for the Pennsylvania gas MVP."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"


@dataclass(frozen=True)
class LayerSpec:
    dataset: str
    query_url: str
    out_path: str
    out_fields: str
    page_size: int = 1000
    out_sr: int | None = 4326


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec(
        dataset="pa_dep_unconventional_permits",
        query_url=(
            "https://gis.dep.pa.gov/depgisprd/rest/services/OilGas/"
            "UconventionalPermits/MapServer/0/query"
        ),
        out_path="data/raw/pa_dep/pa_dep_unconventional_permits.geojson",
        out_fields="*",
        page_size=1000,
        out_sr=4326,
    ),
)


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def count_features(query_url: str) -> int:
    params = {
        "where": "1=1",
        "returnCountOnly": "true",
        "f": "json",
    }
    payload = fetch_json(f"{query_url}?{urlencode(params)}")
    return int(payload["count"])


def paged_feature_collection(spec: LayerSpec) -> tuple[dict, int]:
    total = count_features(spec.query_url)
    features: list[dict] = []
    for offset in range(0, total, spec.page_size):
        params = {
            "where": "1=1",
            "outFields": spec.out_fields,
            "returnGeometry": "true",
            "f": "geojson",
            "orderByFields": "OBJECTID",
            "resultOffset": str(offset),
            "resultRecordCount": str(spec.page_size),
        }
        if spec.out_sr is not None:
            params["outSR"] = str(spec.out_sr)
        page = fetch_json(f"{spec.query_url}?{urlencode(params)}")
        page_features = page.get("features", [])
        if not isinstance(page_features, list):
            raise RuntimeError(f"Unexpected GeoJSON payload for {spec.dataset}")
        features.extend(page_features)

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    return collection, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve output paths.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    downloads: list[dict[str, object]] = []

    for spec in LAYERS:
        output_path = repo_root / spec.out_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        collection, expected_count = paged_feature_collection(spec)
        output_path.write_text(json.dumps(collection), encoding="utf-8")
        actual_count = len(collection["features"])
        if actual_count != expected_count:
            raise RuntimeError(
                f"{spec.dataset} expected {expected_count} features but wrote {actual_count}"
            )
        downloads.append(
            {
                "dataset": spec.dataset,
                "query_url": spec.query_url,
                "output_path": str(output_path),
                "feature_count": actual_count,
                "out_fields": spec.out_fields,
                "out_sr": spec.out_sr,
                "action": "snapshot_refreshed",
            }
        )

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": downloads,
    }
    manifest_path = repo_root / "data/raw/arcgis_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"download failed: {exc}", file=sys.stderr)
        raise
