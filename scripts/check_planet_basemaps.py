#!/usr/bin/env python3
"""Record Planet basemaps availability and a small mosaic sample."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path("/home/shin/Mineral_Gas_Locator")
PLANET_MOSAICS_URL = "https://api.planet.com/basemaps/v1/mosaics"


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


def candidate_requests(api_key: str, url: str) -> list[tuple[str, dict[str, str], str]]:
    basic = base64.b64encode(f"{api_key}:".encode()).decode()
    return [
        (
            "basic",
            {
                "Authorization": f"Basic {basic}",
                "User-Agent": "codex-cli/1.0",
            },
            url,
        ),
        (
            "api-key-header",
            {
                "Authorization": f"api-key {api_key}",
                "User-Agent": "codex-cli/1.0",
            },
            url,
        ),
        (
            "query-param",
            {"User-Agent": "codex-cli/1.0"},
            f"{url}?{urlencode({'api_key': api_key})}",
        ),
    ]


def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    output_dir = REPO_ROOT / "data/raw/satellite_catalog/planet_basemaps_catalog"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "catalog_items.json"
    manifest_path = output_dir / "download_manifest.json"

    api_key = os.getenv("PLANETLAB_API_KEY")
    manifest: dict = {
        "checked_at_utc": utc_now(),
        "configured": bool(api_key),
        "authorized": False,
        "source_id": "planet_basemaps_catalog",
        "sample_page_size": 10,
    }
    mosaics: list[dict] = []
    http_statuses: list[int] = []

    if api_key:
        for auth_mode, headers, candidate_url in candidate_requests(api_key, f"{PLANET_MOSAICS_URL}?_page_size=10"):
            try:
                request = Request(candidate_url, headers=headers)
                with urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                mosaics = payload.get("mosaics", [])
                manifest["authorized"] = True
                manifest["auth_mode"] = auth_mode
                break
            except HTTPError as exc:
                http_statuses.append(exc.code)
                manifest["last_error"] = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception as exc:  # pragma: no cover
                manifest["last_error"] = str(exc)
                break
    else:
        manifest["last_error"] = "PLANETLAB_API_KEY missing"

    manifest["http_statuses"] = http_statuses
    manifest["mosaic_count_sample"] = len(mosaics)
    manifest["has_basemaps"] = len(mosaics) > 0
    manifest["sample_mosaics"] = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "first_acquired": item.get("first_acquired"),
            "last_acquired": item.get("last_acquired"),
            "interval": item.get("interval"),
        }
        for item in mosaics
        if item.get("name")
    ]
    if manifest["authorized"] and not manifest["has_basemaps"]:
        manifest["message"] = "Planet authenticated, but no accessible basemap mosaics were returned."
    elif manifest["authorized"]:
        manifest["message"] = "Planet basemap mosaics available."

    catalog_path.write_text(json.dumps({"mosaics": manifest["sample_mosaics"]}, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
