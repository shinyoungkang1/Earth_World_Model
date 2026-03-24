#!/usr/bin/env python3
"""Download FracFocus national well database (all US fracked wells).

Uses Playwright to handle the terms acceptance page.
Output: data/raw/fracfocus/FracFocusCSV.zip (~433 MB, ~200K+ wells)

Usage:
    python scripts/download_fracfocus.py
    python scripts/download_fracfocus.py --output-dir data/raw/fracfocus
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


async def download_fracfocus(output_dir: Path) -> None:
    from playwright.async_api import async_playwright

    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        print("Loading FracFocus download page...")
        await page.goto("https://www.fracfocus.org/data-download", timeout=30000)
        await page.wait_for_load_state("networkidle")

        # Find and click the Oil & Gas CSV download link
        csv_links = await page.query_selector_all("a[href*='FracFocusCSV']")
        if not csv_links:
            raise RuntimeError("FracFocusCSV download link not found on page")

        print(f"Found {len(csv_links)} CSV download link(s), downloading...")
        async with page.expect_download(timeout=600000) as download_info:
            await csv_links[0].click()
        download = await download_info.value
        save_path = output_dir / download.suggested_filename
        await download.save_as(str(save_path))
        print(f"Downloaded: {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")

        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FracFocus national well database.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/raw/fracfocus"))
    args = parser.parse_args()
    asyncio.run(download_fracfocus(Path(args.output_dir)))


if __name__ == "__main__":
    main()
