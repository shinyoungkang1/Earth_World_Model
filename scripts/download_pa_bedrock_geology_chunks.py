#!/usr/bin/env python3
"""Download the PA DCNR bedrock geology layer in chunked GeoJSON pages."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"
QUERY_URL = "https://mapservices.pasda.psu.edu/server/rest/services/pasda/DCNR2/MapServer/11/query"
OUT_FIELDS = "OBJECTID,MAP_SYMBOL,NAME,AGE,LITH1,LITH2,LITH3,L_DESC"


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def fetch_count() -> int:
    params = {
        "where": "1=1",
        "returnCountOnly": "true",
        "f": "json",
    }
    payload = fetch_json(f"{QUERY_URL}?{urlencode(params)}")
    return int(payload["count"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve output paths.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Number of features per GeoJSON chunk request.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = repo_root / "data/raw/dcnr/pa_dcnr_bedrock_geology_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_count = fetch_count()
    downloads = []
    for index, offset in enumerate(range(0, total_count, args.chunk_size), start=1):
        params = {
            "where": "1=1",
            "outFields": OUT_FIELDS,
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",
            "orderByFields": "OBJECTID_1",
            "resultOffset": str(offset),
            "resultRecordCount": str(args.chunk_size),
        }
        payload = fetch_json(f"{QUERY_URL}?{urlencode(params)}")
        chunk_path = out_dir / f"chunk_{index:04d}.geojson"
        chunk_path.write_text(json.dumps(payload), encoding="utf-8")
        feature_count = len(payload.get("features", []))
        downloads.append(
            {
                "chunk_index": index,
                "result_offset": offset,
                "requested_count": args.chunk_size,
                "feature_count": feature_count,
                "output_path": str(chunk_path),
            }
        )
        print(
            json.dumps(
                {
                    "chunk_index": index,
                    "result_offset": offset,
                    "requested_count": args.chunk_size,
                    "feature_count": feature_count,
                    "output_path": str(chunk_path),
                }
            ),
            flush=True,
        )

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "pa_dcnr_bedrock_geology",
        "query_url": QUERY_URL,
        "source_host": "PASDA/DCNR",
        "out_fields": OUT_FIELDS,
        "chunk_size": args.chunk_size,
        "feature_count": total_count,
        "chunk_count": len(downloads),
        "action": "snapshot_refreshed",
        "downloads": downloads,
    }
    manifest_path = repo_root / "data/raw/dcnr/download_manifest_bedrock_geology_chunks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"download failed: {exc}", file=sys.stderr)
        raise
