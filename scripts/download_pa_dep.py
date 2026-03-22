#!/usr/bin/env python3
"""Download public Pennsylvania DEP oil and gas MVP datasets."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener


USER_AGENT = "Mozilla/5.0 (compatible; codex-cli/1.0; +https://openai.com)"

PRODUCTION_PAGE = "https://greenport.pa.gov/ReportExtracts/OG/OilGasWellProdReport"
INVENTORY_PAGE = "https://greenport.pa.gov/ReportExtracts/OG/OilGasWellInventoryReport"
DATA_DICTIONARY_PDF = (
    "https://files.dep.state.pa.us/OilGas/BOGM/BOGMPortalFiles/"
    "OilGasReports/HelpDocs/SSRS_Report_Data_Dictionary/"
    "DEP_Oil_and_GAS_Reports_Data_Dictionary.pdf"
)

TOKEN_RE = re.compile(
    r'name="__RequestVerificationToken"\s+type="hidden"\s+value="([^"]+)"'
)
OPTION_RE = re.compile(r'<option value="([^"]*)">(.*?)</option>', re.DOTALL)
FORM_ACTION_RE = re.compile(r'<form[^>]+action="([^"]+)"[^>]*>', re.DOTALL)


@dataclass
class PageInfo:
    url: str
    html_text: str
    token: str
    action: str


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"downloads": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def item_key(item: dict) -> tuple:
    if item.get("dataset") == "pa_dep_production":
        return (item.get("dataset"), item.get("period_id"))
    return (item.get("dataset"),)


def merge_downloads(existing: list[dict], updates: list[dict]) -> list[dict]:
    merged = {item_key(item): item for item in existing}
    for item in updates:
        merged[item_key(item)] = item
    return list(merged.values())


def persist_manifest_snapshot(manifest_path: Path, existing_downloads: list[dict], updates: list[dict]) -> None:
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": merge_downloads(existing_downloads, updates),
    }
    write_manifest(manifest_path, manifest)


def build_session():
    jar = CookieJar()
    return build_opener(HTTPCookieProcessor(jar))


def fetch_text(opener, url: str, data: bytes | None = None, referer: str | None = None) -> str:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = Request(url, data=data, headers=headers)
    with opener.open(request, timeout=120) as response:
        body = response.read()
    return body.decode("utf-8", errors="replace")


def fetch_bytes(opener, url: str, data: bytes | None = None, referer: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = Request(url, data=data, headers=headers)
    with opener.open(request, timeout=120) as response:
        return response.read()


def parse_page(url: str, html_text: str) -> PageInfo:
    token_match = TOKEN_RE.search(html_text)
    action_match = FORM_ACTION_RE.search(html_text)
    if not token_match or not action_match:
        raise RuntimeError(f"Could not parse CSRF token or form action from {url}")
    return PageInfo(
        url=url,
        html_text=html_text,
        token=token_match.group(1),
        action=urljoin(url, html.unescape(action_match.group(1))),
    )


def extract_options(html_text: str, select_id: str) -> list[tuple[str, str]]:
    marker = f'id="{select_id}"'
    start = html_text.find(marker)
    if start == -1:
        raise RuntimeError(f"Could not find select {select_id}")
    select_start = html_text.rfind("<select", 0, start)
    select_end = html_text.find("</select>", start)
    if select_start == -1 or select_end == -1:
        raise RuntimeError(f"Could not extract select block for {select_id}")
    block = html_text[select_start:select_end]
    options = []
    for value, label in OPTION_RE.findall(block):
        clean_label = html.unescape(re.sub(r"\s+", " ", label)).strip()
        options.append((value, clean_label))
    return options


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("._") or "file"


def write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def ensure_csv(content: bytes, context: str) -> None:
    sample = content[:200].lstrip()
    if sample.startswith(b"<") or b"<html" in sample.lower():
        raise RuntimeError(f"{context} returned HTML instead of CSV")


def download_inventory(opener, out_dir: Path, page: PageInfo) -> dict:
    payload = urlencode(
        [
            ("Operator", "-1"),
            ("Unconvid", "Y"),
            ("__RequestVerificationToken", page.token),
        ]
    ).encode()
    content = fetch_bytes(opener, page.action, data=payload, referer=page.url)
    ensure_csv(content, "Inventory export")
    output = out_dir / "pa_dep_unconventional_well_inventory.csv"
    write_file(output, content)
    return {
        "dataset": "pa_dep_unconventional_well_inventory",
        "source_page": page.url,
        "form_action": page.action,
        "output_path": str(output),
        "action": "downloaded",
    }


def select_recent_unconventional_periods(
    options: Iterable[tuple[str, str]], latest_months: int
) -> list[tuple[str, str]]:
    selected = []
    for value, label in options:
        if "PRODUCTION: Unconventional wells" in label:
            selected.append((value, label))
        if len(selected) >= latest_months:
            break
    return selected


def select_all_unconventional_periods(options: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(value, label) for value, label in options if "Unconventional wells" in label]


def select_recent_conventional_annual_periods(
    options: Iterable[tuple[str, str]], years: int
) -> list[tuple[str, str]]:
    selected = []
    for value, label in options:
        if "(Conventional wells)" in label and "Jan - Dec" in label:
            selected.append((value, label))
        if len(selected) >= years:
            break
    return selected


def download_production(
    opener,
    out_dir: Path,
    page: PageInfo,
    manifest_path: Path,
    latest_unconventional_months: int,
    all_unconventional_periods: bool,
    conventional_years: int,
    existing_downloads: list[dict],
    skip_existing: bool,
    max_retries: int,
    retry_sleep_seconds: float,
    refresh_page_every: int,
) -> list[dict]:
    options = extract_options(page.html_text, "RptPeriodsid")
    selected = []
    if all_unconventional_periods:
        selected.extend(select_all_unconventional_periods(options))
    else:
        selected.extend(select_recent_unconventional_periods(options, latest_unconventional_months))
    selected.extend(select_recent_conventional_annual_periods(options, conventional_years))
    existing_by_period = {
        item.get("period_id"): item
        for item in existing_downloads
        if item.get("dataset") == "pa_dep_production" and item.get("period_id")
    }

    downloads = []
    requests_since_refresh = 0
    for value, label in selected:
        existing = existing_by_period.get(value)
        if skip_existing and existing:
            existing_path = Path(existing["output_path"])
            if existing_path.exists():
                reused = dict(existing)
                reused["action"] = "reused"
                downloads.append(reused)
                persist_manifest_snapshot(manifest_path, existing_downloads, downloads)
                continue

        attempt = 0
        while True:
            attempt += 1
            try:
                if refresh_page_every > 0 and requests_since_refresh >= refresh_page_every:
                    page = parse_page(PRODUCTION_PAGE, fetch_text(opener, PRODUCTION_PAGE))
                    requests_since_refresh = 0
                payload = urlencode(
                    [
                        ("RptPeriodsid", value),
                        ("Operator", ""),
                        ("perNum", ""),
                        ("__RequestVerificationToken", page.token),
                    ]
                ).encode()
                content = fetch_bytes(opener, page.action, data=payload, referer=page.url)
                ensure_csv(content, f"Production export for {label}")
                requests_since_refresh += 1
                break
            except (HTTPError, URLError, RuntimeError) as exc:
                if attempt >= max_retries:
                    raise
                time.sleep(retry_sleep_seconds * attempt)
                opener = build_session()
                page = parse_page(PRODUCTION_PAGE, fetch_text(opener, PRODUCTION_PAGE))
                requests_since_refresh = 0

        filename = sanitize_filename(f"production_{value}_{label}.csv")
        output = out_dir / filename
        write_file(output, content)
        downloads.append(
            {
                "dataset": "pa_dep_production",
                "period_id": value,
                "period_label": label,
                "source_page": page.url,
                "form_action": page.action,
                "output_path": str(output),
                "action": "downloaded",
                "retry_count": attempt - 1,
            }
        )
        persist_manifest_snapshot(manifest_path, existing_downloads, downloads)
    return downloads


def download_data_dictionary(opener, out_dir: Path) -> dict:
    output = out_dir / "pa_dep_oil_and_gas_reports_data_dictionary.pdf"
    if output.exists():
        return {
            "dataset": "pa_dep_data_dictionary",
            "source_url": DATA_DICTIONARY_PDF,
            "output_path": str(output),
            "action": "reused",
        }
    content = fetch_bytes(opener, DATA_DICTIONARY_PDF)
    write_file(output, content)
    return {
        "dataset": "pa_dep_data_dictionary",
        "source_url": DATA_DICTIONARY_PDF,
        "output_path": str(output),
        "action": "downloaded",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="/home/shin/Mineral_Gas_Locator/data/raw/pa_dep",
        help="Directory for downloaded public PA DEP files.",
    )
    parser.add_argument(
        "--latest-unconventional-months",
        type=int,
        default=12,
        help="Number of most recent monthly unconventional production CSVs to download.",
    )
    parser.add_argument(
        "--conventional-years",
        type=int,
        default=2,
        help="Number of most recent annual conventional production CSVs to download.",
    )
    parser.add_argument(
        "--all-unconventional-periods",
        action="store_true",
        help="Download the full currently listed unconventional production history instead of only the most recent months.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Refresh only the unconventional well inventory and data dictionary.",
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Refresh only production files.",
    )
    parser.add_argument(
        "--skip-existing-production",
        action="store_true",
        help="Reuse production periods already recorded in the manifest when the file exists.",
    )
    parser.add_argument(
        "--production-max-retries",
        type=int,
        default=4,
        help="Maximum attempts per production export request.",
    )
    parser.add_argument(
        "--production-retry-sleep-seconds",
        type=float,
        default=3.0,
        help="Base sleep time used for backoff between production export retries.",
    )
    parser.add_argument(
        "--refresh-production-page-every",
        type=int,
        default=8,
        help="Refresh the production page/token after this many export requests. Set to 0 to disable.",
    )
    args = parser.parse_args()

    if args.inventory_only and args.production_only:
        raise SystemExit("--inventory-only and --production-only cannot be used together")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "download_manifest.json"
    existing_manifest = load_manifest(manifest_path)
    existing_downloads = existing_manifest.get("downloads", [])

    opener = build_session()

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": [],
    }

    if not args.production_only:
        inventory_html = fetch_text(opener, INVENTORY_PAGE)
        inventory_page = parse_page(INVENTORY_PAGE, inventory_html)
        manifest["downloads"].append(download_inventory(opener, out_dir, inventory_page))
        manifest["downloads"].append(download_data_dictionary(opener, out_dir))

    if not args.inventory_only:
        production_html = fetch_text(opener, PRODUCTION_PAGE)
        production_page = parse_page(PRODUCTION_PAGE, production_html)
        manifest["downloads"].extend(
            download_production(
                opener,
                out_dir,
                production_page,
                manifest_path=manifest_path,
                latest_unconventional_months=args.latest_unconventional_months,
                all_unconventional_periods=args.all_unconventional_periods,
                conventional_years=args.conventional_years,
                existing_downloads=existing_downloads,
                skip_existing=args.skip_existing_production,
                max_retries=args.production_max_retries,
                retry_sleep_seconds=args.production_retry_sleep_seconds,
                refresh_page_every=args.refresh_production_page_every,
            )
        )

    manifest["downloads"] = merge_downloads(existing_downloads, manifest["downloads"])
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"download failed: {exc}", file=sys.stderr)
        raise
