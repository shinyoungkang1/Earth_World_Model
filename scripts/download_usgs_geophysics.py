#!/usr/bin/env python3
"""Download direct USGS geophysics bundles for the Pennsylvania gas MVP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"

DOWNLOADS = (
    {
        "dataset": "usgs_ds9_geophysics_zip",
        "url": "https://pubs.usgs.gov/ds/009/ds9.zip",
        "out_path": "data/raw/usgs/ds9_geophysics.zip",
    },
    {
        "dataset": "usgs_ds9_readme",
        "url": "https://pubs.usgs.gov/ds/009/readme.txt",
        "out_path": "data/raw/usgs/ds9_readme.txt",
    },
)


def fetch_to_path(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=300) as response, path.open("wb") as dst:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve output paths.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    results = []
    for item in DOWNLOADS:
        output_path = repo_root / item["out_path"]
        action = "reused"
        if not output_path.exists():
            fetch_to_path(item["url"], output_path)
            action = "downloaded"
        results.append(
            {
                "dataset": item["dataset"],
                "url": item["url"],
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "action": action,
            }
        )

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": results,
    }
    manifest_path = repo_root / "data/raw/usgs/download_manifest_geophysics.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"download failed: {exc}", file=sys.stderr)
        raise
