#!/usr/bin/env python3
"""Build a paper-scale regional center set with isolated centers plus 3x3 mesoscale clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-scale 10k regional center design from candidate regions.",
    )
    parser.add_argument("--regions-path", type=Path, required=True, help="Candidate-region parquet or CSV.")
    parser.add_argument("--output-path", type=Path, required=True, help="Detailed sampled centers table.")
    parser.add_argument(
        "--output-ee-points-csv",
        type=Path,
        default=None,
        help="Optional Earth Engine points CSV with cluster metadata passthrough.",
    )
    parser.add_argument("--target-total-centers", type=int, default=10000)
    parser.add_argument(
        "--target-cluster-count",
        type=int,
        default=222,
        help="Number of 3x3 cluster anchors. 222 anchors yield 1,998 clustered centers.",
    )
    parser.add_argument("--cluster-grid-radius", type=int, default=1, help="1 means 3x3 clusters.")
    parser.add_argument(
        "--cluster-anchor-min-separation-cells",
        type=int,
        default=3,
        help="Minimum Chebyshev separation between cluster anchors in candidate-grid cells.",
    )
    parser.add_argument(
        "--isolated-exclusion-radius-cells",
        type=int,
        default=2,
        help="Exclude isolated centers within this Chebyshev distance of any selected cluster anchor.",
    )
    parser.add_argument(
        "--equal-bucket-fraction",
        type=float,
        default=0.5,
        help="Fraction of quotas allocated evenly across occupied diversification buckets.",
    )
    parser.add_argument("--selection-seed", default="paper_scale_10k_v2")
    parser.add_argument("--sample-id-prefix", default="paper10k_")
    parser.add_argument("--sample-id-start", type=int, default=0)
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults next to output-path.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported input format for {path}. Expected .csv or .parquet")


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    raise SystemExit(f"Unsupported output format for {path}. Expected .csv or .parquet")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required candidate-region columns: {missing}")


def chebyshev_distance(ax: int, ay: int, bx: int, by: int) -> int:
    return int(max(abs(int(ax) - int(bx)), abs(int(ay) - int(by))))


def stable_rank_key(text: str, seed: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def distribute_quota(
    available_counts: pd.Series,
    *,
    target_total: int,
    equal_fraction: float,
) -> dict[str, int]:
    counts = available_counts[available_counts > 0].sort_index()
    if counts.empty:
        return {}
    if target_total <= 0 or target_total >= int(counts.sum()):
        return {str(bucket): int(value) for bucket, value in counts.items()}

    quotas = {str(bucket): 0 for bucket in counts.index}
    bucket_ids = list(counts.index)
    equal_total = min(target_total, int(round(target_total * equal_fraction)))
    if equal_total > 0:
        base = equal_total // len(bucket_ids)
        remainder = equal_total % len(bucket_ids)
        for index, bucket in enumerate(bucket_ids):
            want = base + (1 if index < remainder else 0)
            take = min(int(counts.loc[bucket]), want)
            quotas[str(bucket)] += take

    def remaining_capacity() -> pd.Series:
        return counts - pd.Series(quotas)

    remaining = target_total - sum(quotas.values())
    while remaining > 0:
        capacity = remaining_capacity()
        capacity = capacity[capacity > 0]
        if capacity.empty:
            break

        weights = capacity.astype(float) / float(capacity.sum())
        raw = weights * remaining
        floor = raw.map(int)
        assigned = int(floor.sum())
        for bucket, value in floor.items():
            if value > 0:
                quotas[str(bucket)] += int(value)

        leftovers = remaining - assigned
        if leftovers > 0:
            fractional = (raw - floor).sort_values(ascending=False)
            for bucket in fractional.index:
                key = str(bucket)
                if leftovers <= 0:
                    break
                if quotas[key] >= int(counts.loc[bucket]):
                    continue
                quotas[key] += 1
                leftovers -= 1

        new_remaining = target_total - sum(quotas.values())
        if new_remaining == remaining:
            greedy = capacity.sort_values(ascending=False)
            for bucket in greedy.index:
                key = str(bucket)
                if new_remaining <= 0:
                    break
                if quotas[key] >= int(counts.loc[bucket]):
                    continue
                quotas[key] += 1
                new_remaining -= 1
        remaining = target_total - sum(quotas.values())

    return quotas


def cluster_member_offsets(radius: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dy in range(radius, -radius - 1, -1):
        for dx in range(-radius, radius + 1):
            offsets.append((dx, dy))
    return offsets


def has_full_cluster(center_lookup: dict[tuple[int, int], dict], *, grid_x: int, grid_y: int, radius: int) -> bool:
    return all((int(grid_x) + dx, int(grid_y) + dy) in center_lookup for dx, dy in cluster_member_offsets(radius))


def anchor_respects_separation(
    *,
    anchor_x: int,
    anchor_y: int,
    chosen_anchors: list[tuple[int, int]],
    min_separation_cells: int,
) -> bool:
    return all(
        chebyshev_distance(anchor_x, anchor_y, other_x, other_y) >= int(min_separation_cells)
        for other_x, other_y in chosen_anchors
    )


def select_cluster_anchors(
    *,
    candidates: pd.DataFrame,
    target_cluster_count: int,
    cluster_grid_radius: int,
    min_separation_cells: int,
    equal_bucket_fraction: float,
    selection_seed: str,
) -> pd.DataFrame:
    center_lookup = {
        (int(row.grid_x), int(row.grid_y)): row._asdict()
        for row in candidates.itertuples(index=False)
    }
    anchor_candidates = candidates.copy()
    anchor_candidates["has_full_cluster"] = [
        has_full_cluster(
            center_lookup,
            grid_x=int(row.grid_x),
            grid_y=int(row.grid_y),
            radius=int(cluster_grid_radius),
        )
        for row in anchor_candidates.itertuples(index=False)
    ]
    anchor_candidates = anchor_candidates[anchor_candidates["has_full_cluster"]].copy()
    if anchor_candidates.empty:
        raise SystemExit("No valid cluster anchors found with full local 3x3 support.")

    anchor_candidates["anchor_rank"] = anchor_candidates["region_id"].map(
        lambda value: stable_rank_key(str(value), f"{selection_seed}:cluster_anchor")
    )
    anchor_candidates = anchor_candidates.sort_values(["bucket_id", "anchor_rank", "region_id"]).reset_index(drop=True)

    quotas = distribute_quota(
        anchor_candidates["bucket_id"].value_counts().sort_index(),
        target_total=int(target_cluster_count),
        equal_fraction=float(equal_bucket_fraction),
    )
    chosen_rows: list[pd.Series] = []
    chosen_anchors: list[tuple[int, int]] = []
    quota_counts = {bucket: 0 for bucket in quotas}

    for bucket_id in sorted(quotas):
        bucket_frame = anchor_candidates[anchor_candidates["bucket_id"] == bucket_id]
        quota = int(quotas[bucket_id])
        for row in bucket_frame.itertuples(index=False):
            if quota_counts[bucket_id] >= quota:
                break
            anchor_x = int(row.grid_x)
            anchor_y = int(row.grid_y)
            if not anchor_respects_separation(
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                chosen_anchors=chosen_anchors,
                min_separation_cells=int(min_separation_cells),
            ):
                continue
            chosen_rows.append(pd.Series(row._asdict()))
            chosen_anchors.append((anchor_x, anchor_y))
            quota_counts[bucket_id] += 1

    if len(chosen_rows) < int(target_cluster_count):
        remaining = anchor_candidates[
            ~anchor_candidates["region_id"].isin({str(row["region_id"]) for row in chosen_rows})
        ].sort_values(["anchor_rank", "region_id"])
        for row in remaining.itertuples(index=False):
            if len(chosen_rows) >= int(target_cluster_count):
                break
            anchor_x = int(row.grid_x)
            anchor_y = int(row.grid_y)
            if not anchor_respects_separation(
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                chosen_anchors=chosen_anchors,
                min_separation_cells=int(min_separation_cells),
            ):
                continue
            chosen_rows.append(pd.Series(row._asdict()))
            chosen_anchors.append((anchor_x, anchor_y))

    if len(chosen_rows) < int(target_cluster_count):
        raise SystemExit(
            f"Only found {len(chosen_rows)} non-overlapping cluster anchors; "
            f"requested {target_cluster_count}. Relax spacing or cluster count."
        )
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def expand_cluster_members(
    *,
    anchor_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    cluster_grid_radius: int,
) -> pd.DataFrame:
    center_lookup = {
        (int(row.grid_x), int(row.grid_y)): row._asdict()
        for row in candidates.itertuples(index=False)
    }
    rows: list[dict] = []
    cluster_side = int(2 * cluster_grid_radius + 1)
    for cluster_index, anchor_row in enumerate(anchor_frame.itertuples(index=False), start=1):
        anchor_x = int(anchor_row.grid_x)
        anchor_y = int(anchor_row.grid_y)
        cluster_id = f"cluster_{cluster_index:04d}"
        for dx, dy in cluster_member_offsets(cluster_grid_radius):
            member = center_lookup[(anchor_x + dx, anchor_y + dy)]
            rows.append(
                {
                    **member,
                    "sampling_design": "clustered_3x3",
                    "sampling_group_id": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_anchor_region_id": str(anchor_row.region_id),
                    "cluster_anchor_grid_x": anchor_x,
                    "cluster_anchor_grid_y": anchor_y,
                    "cluster_dx_idx": int(dx),
                    "cluster_dy_idx": int(dy),
                    "cluster_grid_radius": int(cluster_grid_radius),
                    "cluster_grid_side": int(cluster_side),
                }
            )
    return pd.DataFrame(rows)


def build_cluster_exclusion_mask(
    *,
    candidates: pd.DataFrame,
    anchor_frame: pd.DataFrame,
    exclusion_radius_cells: int,
) -> pd.Series:
    if exclusion_radius_cells < 0:
        raise SystemExit("isolated-exclusion-radius-cells must be >= 0.")
    if anchor_frame.empty:
        return pd.Series([False] * len(candidates), index=candidates.index)
    mask = pd.Series([False] * len(candidates), index=candidates.index)
    for anchor in anchor_frame.itertuples(index=False):
        anchor_x = int(anchor.grid_x)
        anchor_y = int(anchor.grid_y)
        distances = candidates.apply(
            lambda row: chebyshev_distance(int(row["grid_x"]), int(row["grid_y"]), anchor_x, anchor_y),
            axis=1,
        )
        mask |= distances <= int(exclusion_radius_cells)
    return mask


def select_isolated_centers(
    *,
    candidates: pd.DataFrame,
    target_count: int,
    equal_bucket_fraction: float,
    selection_seed: str,
) -> pd.DataFrame:
    if target_count <= 0:
        return candidates.head(0).copy()
    isolated_pool = candidates.copy()
    isolated_pool["isolated_rank"] = isolated_pool["region_id"].map(
        lambda value: stable_rank_key(str(value), f"{selection_seed}:isolated")
    )
    isolated_pool = isolated_pool.sort_values(["bucket_id", "isolated_rank", "region_id"]).reset_index(drop=True)
    quotas = distribute_quota(
        isolated_pool["bucket_id"].value_counts().sort_index(),
        target_total=int(target_count),
        equal_fraction=float(equal_bucket_fraction),
    )
    selected_frames: list[pd.DataFrame] = []
    for bucket_id in sorted(quotas):
        quota = int(quotas[bucket_id])
        bucket_frame = isolated_pool[isolated_pool["bucket_id"] == bucket_id].sort_values(
            ["isolated_rank", "region_id"]
        )
        selected_frames.append(bucket_frame.head(quota).copy())
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else isolated_pool.head(0).copy()
    if len(selected) < int(target_count):
        remaining = isolated_pool[~isolated_pool["region_id"].isin(set(selected["region_id"]))].sort_values(
            ["isolated_rank", "region_id"]
        )
        selected = pd.concat([selected, remaining.head(int(target_count) - len(selected)).copy()], ignore_index=True)
    selected = selected.head(int(target_count)).copy()
    selected["sampling_design"] = "isolated"
    selected["sampling_group_id"] = selected["region_id"].astype(str)
    selected["cluster_id"] = ""
    selected["cluster_anchor_region_id"] = ""
    selected["cluster_anchor_grid_x"] = pd.NA
    selected["cluster_anchor_grid_y"] = pd.NA
    selected["cluster_dx_idx"] = pd.NA
    selected["cluster_dy_idx"] = pd.NA
    selected["cluster_grid_radius"] = pd.NA
    selected["cluster_grid_side"] = pd.NA
    return selected


def build_output_table(sampled: pd.DataFrame, *, sample_id_prefix: str, sample_id_start: int) -> pd.DataFrame:
    out = sampled.copy().reset_index(drop=True)
    out["sample_id"] = [
        f"{sample_id_prefix}{int(sample_id_start) + idx:08d}"
        for idx in range(len(out))
    ]
    ordered_columns = [
        "sample_id",
        "region_id",
        "sampling_design",
        "sampling_group_id",
        "cluster_id",
        "cluster_anchor_region_id",
        "cluster_anchor_grid_x",
        "cluster_anchor_grid_y",
        "cluster_dx_idx",
        "cluster_dy_idx",
        "cluster_grid_radius",
        "cluster_grid_side",
        "center_latitude",
        "center_longitude",
        "continent_bucket",
        "country_bucket",
        "latitude_band",
        "bucket_id",
        "grid_x",
        "grid_y",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "region_side_km",
        "region_spacing_km",
        "projected_crs",
        "selection_seed",
    ]
    passthrough_columns = [column for column in out.columns if column not in ordered_columns]
    out = out[ordered_columns + passthrough_columns]
    return out


def build_ee_points_table(frame: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "sample_id",
        "center_latitude",
        "center_longitude",
        "region_id",
        "sampling_design",
        "sampling_group_id",
        "cluster_id",
        "cluster_anchor_region_id",
        "cluster_dx_idx",
        "cluster_dy_idx",
        "continent_bucket",
        "country_bucket",
        "latitude_band",
        "bucket_id",
    ]
    columns = [column for column in preferred_columns if column in frame.columns]
    out = frame[columns].copy()
    out = out.rename(
        columns={
            "center_latitude": "latitude",
            "center_longitude": "longitude",
        }
    )
    return out


def main() -> int:
    args = parse_args()
    cluster_grid_side = int(2 * int(args.cluster_grid_radius) + 1)
    cluster_cell_count = int(cluster_grid_side ** 2)
    clustered_capacity = int(args.target_cluster_count) * cluster_cell_count
    if clustered_capacity > int(args.target_total_centers):
        raise SystemExit(
            "Requested clustered-center capacity exceeds target-total-centers: "
            f"{clustered_capacity} > {args.target_total_centers}"
        )

    regions = read_table(args.regions_path)
    require_columns(
        regions,
        [
            "region_id",
            "center_latitude",
            "center_longitude",
            "grid_x",
            "grid_y",
            "bucket_id",
            "continent_bucket",
            "country_bucket",
            "latitude_band",
        ],
    )
    regions = regions.copy()
    regions["region_id"] = regions["region_id"].astype(str)
    regions["bucket_id"] = regions["bucket_id"].astype(str)
    regions["grid_x"] = regions["grid_x"].astype(int)
    regions["grid_y"] = regions["grid_y"].astype(int)
    regions = regions.sort_values(["bucket_id", "region_id"]).reset_index(drop=True)

    cluster_anchors = select_cluster_anchors(
        candidates=regions,
        target_cluster_count=int(args.target_cluster_count),
        cluster_grid_radius=int(args.cluster_grid_radius),
        min_separation_cells=int(args.cluster_anchor_min_separation_cells),
        equal_bucket_fraction=float(args.equal_bucket_fraction),
        selection_seed=str(args.selection_seed),
    )
    clustered_centers = expand_cluster_members(
        anchor_frame=cluster_anchors,
        candidates=regions,
        cluster_grid_radius=int(args.cluster_grid_radius),
    )
    cluster_region_ids = set(clustered_centers["region_id"].astype(str))
    cluster_exclusion_mask = build_cluster_exclusion_mask(
        candidates=regions,
        anchor_frame=cluster_anchors,
        exclusion_radius_cells=int(args.isolated_exclusion_radius_cells),
    )
    isolated_pool = regions[
        ~regions["region_id"].isin(cluster_region_ids) & ~cluster_exclusion_mask
    ].copy()

    clustered_center_count = int(len(clustered_centers))
    isolated_target_count = int(args.target_total_centers) - clustered_center_count
    if isolated_target_count < 0:
        raise SystemExit(
            f"Clustered centers already exceed target total: {clustered_center_count} > {args.target_total_centers}"
        )

    isolated_centers = select_isolated_centers(
        candidates=isolated_pool,
        target_count=isolated_target_count,
        equal_bucket_fraction=float(args.equal_bucket_fraction),
        selection_seed=str(args.selection_seed),
    )

    combined = pd.concat([isolated_centers, clustered_centers], ignore_index=True)
    combined["design_rank"] = combined["sampling_design"].map({"isolated": 0, "clustered_3x3": 1}).fillna(9).astype(int)
    combined["stable_rank"] = combined["region_id"].map(lambda value: stable_rank_key(str(value), str(args.selection_seed)))
    combined = combined.sort_values(
        [
            "design_rank",
            "sampling_group_id",
            "cluster_dy_idx",
            "cluster_dx_idx",
            "stable_rank",
            "region_id",
        ],
        na_position="last",
    ).reset_index(drop=True)
    out = build_output_table(
        combined,
        sample_id_prefix=str(args.sample_id_prefix),
        sample_id_start=int(args.sample_id_start),
    )
    write_table(args.output_path, out)
    if args.output_ee_points_csv is not None:
        ee_points = build_ee_points_table(out)
        args.output_ee_points_csv.parent.mkdir(parents=True, exist_ok=True)
        ee_points.to_csv(args.output_ee_points_csv, index=False)

    summary_path = args.output_summary_json or args.output_path.with_name(f"{args.output_path.stem}_summary.json")
    region_spacing_km = (
        float(out["region_spacing_km"].iloc[0]) if "region_spacing_km" in out.columns and not out.empty else None
    )
    summary = {
        "regions_path": str(args.regions_path),
        "output_path": str(args.output_path),
        "output_ee_points_csv": str(args.output_ee_points_csv) if args.output_ee_points_csv else None,
        "target_total_centers": int(args.target_total_centers),
        "selected_total_centers": int(len(out)),
        "cluster_anchor_count": int(len(cluster_anchors)),
        "clustered_center_count": int(clustered_center_count),
        "isolated_center_count": int(len(isolated_centers)),
        "cluster_grid_radius": int(args.cluster_grid_radius),
        "cluster_grid_side": int(cluster_grid_side),
        "cluster_cell_count": int(cluster_cell_count),
        "cluster_layout": f"{cluster_grid_side}x{cluster_grid_side}",
        "cluster_anchor_min_separation_cells": int(args.cluster_anchor_min_separation_cells),
        "isolated_exclusion_radius_cells": int(args.isolated_exclusion_radius_cells),
        "region_spacing_km": region_spacing_km,
        "cluster_union_side_km": float(cluster_grid_side * region_spacing_km) if region_spacing_km is not None else None,
        "design_counts": {
            key: int(value)
            for key, value in out["sampling_design"].value_counts().sort_index().items()
        },
        "continent_counts": {
            key: int(value)
            for key, value in out["continent_bucket"].value_counts().sort_index().items()
        },
        "cluster_anchor_bucket_counts": {
            key: int(value)
            for key, value in cluster_anchors["bucket_id"].value_counts().sort_index().items()
        },
        "selection_seed": str(args.selection_seed),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
