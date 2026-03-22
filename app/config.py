"""Configuration helpers for the SubTerra PA MVP app."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path("/home/shin/Mineral_Gas_Locator")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def settings() -> dict[str, str | Path | None]:
    load_env_file(REPO_ROOT / ".env")
    return {
        "repo_root": REPO_ROOT,
        "planet_api_key": os.getenv("PLANETLAB_API_KEY"),
    }
