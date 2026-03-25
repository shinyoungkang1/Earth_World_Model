#!/usr/bin/env python3
"""Shared helpers for Earth Engine stage launch scripts."""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from typing import Any


EE_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def require_ee() -> Any:
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("earthengine-api is not installed. Install requirements first.") from exc
    return ee


def initialize_ee(ee: Any, *, project: str, authenticate: bool = False) -> None:
    """Initialize Earth Engine with a durable auth preference order.

    Prefer the VM compute service account first so fresh VMs do not depend on
    cached human ADC credentials. Fall back to standard EE initialization only
    if the service-account path is unavailable.
    """

    if authenticate:  # pragma: no cover - interactive auth
        ee.Authenticate()
    try:
        from google.auth import compute_engine  # type: ignore

        credentials = compute_engine.Credentials(scopes=[EE_SCOPE])
        ee.Initialize(credentials=credentials, project=project)
        return
    except Exception:
        try:
            ee.Initialize(project=project)
            return
        except Exception:
            import google.auth  # type: ignore

            credentials, _ = google.auth.default(scopes=[EE_SCOPE])
            ee.Initialize(credentials=credentials, project=project)


def read_csv_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise SystemExit(f"CSV not found: {path}") from exc


def detect_nul_bytes(data: bytes) -> int:
    return data.count(b"\x00")


def decode_csv_text(data: bytes, *, path: Path) -> str:
    nul_count = detect_nul_bytes(data)
    if nul_count:
        raise SystemExit(
            f"CSV {path} contains {nul_count} NUL bytes; replace it with a clean copy before launching."
        )
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"CSV {path} is not valid UTF-8 text: {exc}") from exc


def load_location_rows(path: Path, *, sample_offset: int, sample_limit: int) -> list[dict[str, Any]]:
    data = read_csv_bytes(path)
    text = decode_csv_text(data, path=path)
    rows = list(csv.DictReader(io.StringIO(text)))
    rows = rows[max(0, int(sample_offset)) :]
    if sample_limit > 0:
        rows = rows[:sample_limit]
    if not rows:
        raise SystemExit(f"No rows selected from {path}")
    return rows


def check_stage_storage(path: Path, *, min_free_gb: float) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    if free_gb < float(min_free_gb):
        raise SystemExit(
            f"Insufficient free space for stage root {path}: {free_gb:.2f} GiB free < {float(min_free_gb):.2f} GiB required."
        )
    return {
        "free_gb": round(free_gb, 2),
        "used_gb": round(used_gb, 2),
        "total_gb": round(total_gb, 2),
    }


def smoke_test_ee(project: str, *, authenticate: bool = False) -> int:
    ee = require_ee()
    initialize_ee(ee, project=project, authenticate=authenticate)
    return int(ee.Number(1).getInfo())
