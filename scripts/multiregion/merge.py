#!/usr/bin/env python3
"""Merge per-state training tables into a unified multi-region dataset.

Reads per-state training tables, renames columns per the region registry,
validates against the unified schema, and concatenates into one unified table.

Usage:
    python scripts/multiregion/merge.py --repo-root /home/shin/Mineral_Gas_Locator
    python scripts/multiregion/merge.py --regions swpa_core wv_horizontal
    python scripts/multiregion/merge.py --output-version v2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multiregion.schema import (
    ALL_COLUMNS,
    PA_COLUMN_RENAMES,
    WV_COLUMN_RENAMES,
    NEW_STATE_COLUMN_RENAMES,
    add_region_metadata,
    align_columns,
    validate_training_table,
)

# ---------------------------------------------------------------------------
# Column rename lookup (registry string -> actual mapping)
# ---------------------------------------------------------------------------
_RENAME_MAPS: dict[str, dict[str, str]] = {
    "PA_COLUMN_RENAMES": PA_COLUMN_RENAMES,
    "WV_COLUMN_RENAMES": WV_COLUMN_RENAMES,
    # OH, CO, TX all use the canonical-from-the-start naming
    "OH_COLUMN_RENAMES": NEW_STATE_COLUMN_RENAMES,
    "CO_COLUMN_RENAMES": NEW_STATE_COLUMN_RENAMES,
    "TX_COLUMN_RENAMES": NEW_STATE_COLUMN_RENAMES,
    "NEW_STATE_COLUMN_RENAMES": NEW_STATE_COLUMN_RENAMES,
}

# ---------------------------------------------------------------------------
# Training table path resolution per state / region
# ---------------------------------------------------------------------------
_STATE_TABLE_TEMPLATES: dict[str, str] = {
    "PA": "data/features/{basin_id}/gas_training_table_v2.csv",
    "WV": "data/features/wv_horizontal_statewide/gas_training_table_wv_v0.csv",
    "OH": "data/features/oh_utica_statewide/gas_training_table_oh_v0.csv",
    "CO": "data/features/co_dj_basin_wattenberg/gas_training_table_co_v0.csv",
    "TX": "data/features/{basin_id}/gas_training_table_tx_v0.csv",
}


def load_region_registry(repo_root: Path) -> dict:
    """Load the multi-region registry from config/multiregion/region_registry.json."""
    registry_path = repo_root / "config/multiregion/region_registry.json"
    return json.loads(registry_path.read_text(encoding="utf-8"))


def resolve_training_table_path(region: dict, repo_root: Path) -> Path:
    """Return the expected training table CSV path for a region entry."""
    state = region["state"]
    basin_id = region["basin_config"]
    template = _STATE_TABLE_TEMPLATES.get(state)
    if template is None:
        raise ValueError(f"No training table template for state {state!r}")
    rel_path = template.format(basin_id=basin_id)
    return repo_root / rel_path


def get_rename_map(region: dict) -> dict[str, str]:
    """Look up the column rename mapping for a region."""
    key = region["column_renames"]
    rename_map = _RENAME_MAPS.get(key)
    if rename_map is None:
        raise ValueError(
            f"Unknown column_renames key {key!r} for region {region['region_id']!r}. "
            f"Expected one of: {sorted(_RENAME_MAPS)}"
        )
    return rename_map


def load_and_prepare_region(
    region: dict,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a single region's training table, rename, add metadata, and validate.

    Returns (prepared_df, warnings).
    """
    region_id = region["region_id"]
    csv_path = resolve_training_table_path(region, repo_root)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training table not found for region {region_id!r}: {csv_path}"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    rename_map = get_rename_map(region)

    # Rename columns to canonical names
    df = align_columns(df, rename_map)

    # Inject region metadata columns
    df = add_region_metadata(
        df,
        region_id=region_id,
        state=region["state"],
        basin=region["basin_config"],
    )

    # Validate against the unified schema
    warnings = validate_training_table(df, region_id)

    return df, warnings


def utc_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    """CLI entry point for the merge pipeline."""
    parser = argparse.ArgumentParser(
        description="Merge per-state training tables into a unified multi-region dataset.",
    )
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve configs and data paths.",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Specific region_id values to include. Omit for all regions.",
    )
    parser.add_argument(
        "--output-version",
        default="v1",
        help="Version suffix for the output filename (default: v1).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    registry = load_region_registry(repo_root)
    regions = registry["regions"]

    # Filter to requested regions if specified
    if args.regions is not None:
        requested = set(args.regions)
        regions = [r for r in regions if r["region_id"] in requested]
        found = {r["region_id"] for r in regions}
        missing = requested - found
        if missing:
            print(
                f"WARNING: requested regions not found in registry: {sorted(missing)}",
                file=sys.stderr,
            )

    if not regions:
        print("No regions to merge.", file=sys.stderr)
        return 1

    # Process each region
    frames: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    region_summaries: list[dict] = []

    for region in regions:
        region_id = region["region_id"]
        print(f"Loading region: {region_id} ({region['state']})")

        try:
            df, warnings = load_and_prepare_region(region, repo_root)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}", file=sys.stderr)
            region_summaries.append({
                "region_id": region_id,
                "state": region["state"],
                "status": "skipped",
                "reason": str(exc),
            })
            continue

        all_warnings.extend(warnings)
        frames.append(df)

        summary = {
            "region_id": region_id,
            "state": region["state"],
            "basin_config": region["basin_config"],
            "status": "merged",
            "row_count": len(df),
            "warning_count": len(warnings),
        }
        if warnings:
            summary["warnings"] = warnings
        region_summaries.append(summary)

        print(f"  rows={len(df):,}  warnings={len(warnings)}")

    if not frames:
        print("No training tables loaded. Nothing to merge.", file=sys.stderr)
        return 1

    # Concatenate all regions
    unified = pd.concat(frames, ignore_index=True)
    print(f"\nUnified table: {len(unified):,} rows x {len(unified.columns)} columns")

    # Print any warnings
    if all_warnings:
        print(f"\n{len(all_warnings)} validation warning(s):", file=sys.stderr)
        for w in all_warnings:
            print(f"  {w}", file=sys.stderr)

    # Write output
    output_dir = repo_root / "data/features/multiregion"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"unified_training_table_{args.output_version}.csv"
    unified.to_csv(output_csv, index=False)
    print(f"\nWrote: {output_csv}")

    # Write metadata sidecar
    metadata = {
        "merged_at_utc": utc_now(),
        "registry_version": registry.get("registry_version"),
        "output_version": args.output_version,
        "output_path": str(output_csv),
        "total_rows": len(unified),
        "total_warnings": len(all_warnings),
        "columns": list(unified.columns),
        "regions": region_summaries,
    }
    metadata_path = output_dir / f"unified_training_table_{args.output_version}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
