"""Planet basemap helpers for the local API."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import settings


PLANET_MOSAICS_URL = "https://api.planet.com/basemaps/v1/mosaics"
PLANET_TILE_TEMPLATE = "https://tiles.planet.com/basemaps/v1/planet-tiles/{mosaic_name}/gmap/{z}/{x}/{y}.png"


def _auth_headers() -> dict[str, str]:
    api_key = settings()["planet_api_key"]
    if not api_key:
        raise RuntimeError("PLANETLAB_API_KEY is not configured")
    basic = base64.b64encode(f"{api_key}:".encode()).decode()
    return {
        "Authorization": f"Basic {basic}",
        "User-Agent": "codex-cli/1.0",
    }


def _fetch_json_with_fallback(url: str) -> dict:
    api_key = settings()["planet_api_key"]
    errors = []
    for headers, candidate_url in [
        (_auth_headers(), url),
        ({"Authorization": f"api-key {api_key}", "User-Agent": "codex-cli/1.0"}, url),
        ({"User-Agent": "codex-cli/1.0"}, f"{url}?{urlencode({'api_key': api_key})}"),
    ]:
        try:
            request = Request(candidate_url, headers=headers)
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            errors.append(exc.code)
    raise RuntimeError(f"Planet request failed with HTTP statuses {errors}")


class PlanetClient:
    def status(self) -> dict:
        api_key = settings()["planet_api_key"]
        if not api_key:
            return {"configured": False, "authorized": False, "message": "PLANETLAB_API_KEY missing"}
        try:
            payload = _fetch_json_with_fallback(f"{PLANET_MOSAICS_URL}?_page_size=5")
            mosaics = payload.get("mosaics", [])
            has_basemaps = len(mosaics) > 0
            return {
                "configured": True,
                "authorized": True,
                "has_basemaps": has_basemaps,
                "mosaic_count_sample": len(mosaics),
                "message": (
                    "Planet basemaps available"
                    if has_basemaps
                    else "Planet authenticated, but no accessible basemap mosaics were returned"
                ),
            }
        except Exception as exc:
            return {
                "configured": True,
                "authorized": False,
                "has_basemaps": False,
                "message": f"Planet basemaps unavailable: {exc}",
            }

    def list_mosaics(self, page_size: int = 25) -> dict:
        payload = _fetch_json_with_fallback(f"{PLANET_MOSAICS_URL}?_page_size={page_size}")
        mosaics = payload.get("mosaics", [])
        items = []
        for item in mosaics:
            name = item.get("name")
            if not name:
                continue
            items.append(
                {
                    "id": item.get("id"),
                    "name": name,
                    "first_acquired": item.get("first_acquired"),
                    "last_acquired": item.get("last_acquired"),
                    "interval": item.get("interval"),
                    "tile_url_template": f"/api/planet/tiles/{name}" + "/{z}/{x}/{y}.png",
                }
            )
        return {"mosaics": items}

    def fetch_tile(self, mosaic_name: str, z: int, x: int, y: int) -> tuple[bytes, str]:
        api_key = settings()["planet_api_key"]
        if not api_key:
            raise RuntimeError("PLANETLAB_API_KEY missing")
        url = PLANET_TILE_TEMPLATE.format(mosaic_name=mosaic_name, z=z, x=x, y=y)
        url = f"{url}?{urlencode({'api_key': api_key})}"
        request = Request(url, headers={"User-Agent": "codex-cli/1.0"})
        with urlopen(request, timeout=60) as response:
            return response.read(), response.headers.get_content_type()


@lru_cache(maxsize=1)
def planet_client() -> PlanetClient:
    return PlanetClient()
