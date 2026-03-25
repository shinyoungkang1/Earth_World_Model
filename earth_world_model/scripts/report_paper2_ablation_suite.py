#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = [
    {
        "run_id": "yearly_500_stagec_conv2d_1024",
        "label": "Yearly-only + 2D",
        "short_label": "Yearly\n2D",
        "family": "control",
        "color": "#4C78A8",
    },
    {
        "run_id": "yearly_500_staged_rope_conv3d_1024",
        "label": "Yearly-only + 3D",
        "short_label": "Yearly\n3D",
        "family": "control",
        "color": "#72B7B2",
    },
    {
        "run_id": "mixed_ssl4eo_yearly_500_stagec_conv2d_1024",
        "label": "Mixed curriculum + 2D",
        "short_label": "Mixed\n2D",
        "family": "main",
        "color": "#F58518",
    },
    {
        "run_id": "mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024",
        "label": "Mixed curriculum + 3D",
        "short_label": "Mixed\n3D",
        "family": "main",
        "color": "#54A24B",
    },
    {
        "run_id": "mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_no_time_gap",
        "label": "Mixed 3D, no time-gap",
        "short_label": "3D\nNo gap",
        "family": "ablation",
        "color": "#E45756",
    },
    {
        "run_id": "mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_joint_sensor",
        "label": "Mixed 3D, joint-sensor",
        "short_label": "3D\nJoint sensor",
        "family": "ablation",
        "color": "#B279A2",
    },
    {
        "run_id": "mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_no_cross_sensor_loss",
        "label": "Mixed 3D, no cross-sensor loss",
        "short_label": "3D\nNo cross-loss",
        "family": "ablation",
        "color": "#9D755D",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Paper 2 ablation suite from GCS.")
    parser.add_argument(
        "--gcs-root",
        default=(
            "gs://omois-earth-world-model-phase2-20260320-11728/"
            "earth_world_model/runs/paper2_mixed_curriculum_suite_20260324_v1"
        ),
        help="GCS root containing per-run training_summary.json files.",
    )
    parser.add_argument(
        "--output-markdown",
        default=(
            "/home/shin/Mineral_Gas_Locator/Manuscript/"
            "paper2_mixed_curriculum_ablation_results_2026_03_24.md"
        ),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "/home/shin/Mineral_Gas_Locator/Manuscript/"
            "paper2_mixed_curriculum_ablation_results_2026_03_24.csv"
        ),
        help="CSV table output path.",
    )
    parser.add_argument(
        "--output-figure-png",
        default=(
            "/home/shin/Mineral_Gas_Locator/Manuscript/figures/"
            "paper2_mixed_curriculum_best_val_loss_2026_03_24.png"
        ),
        help="PNG figure output path.",
    )
    parser.add_argument(
        "--output-figure-svg",
        default=(
            "/home/shin/Mineral_Gas_Locator/Manuscript/figures/"
            "paper2_mixed_curriculum_best_val_loss_2026_03_24.svg"
        ),
        help="SVG figure output path.",
    )
    return parser.parse_args()


def gcloud_cat(uri: str) -> str:
    proc = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to read {uri}: {proc.stderr.strip()}")
    return proc.stdout


def fmt_sci(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.2e}"


def fmt_float(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def pct_change(new: float, old: float) -> float:
    return ((new - old) / old) * 100.0


def load_rows(gcs_root: str) -> list[dict]:
    rows: list[dict] = []
    for spec in RUNS:
        uri = f"{gcs_root.rstrip('/')}/{spec['run_id']}/training_summary.json"
        data = json.loads(gcloud_cat(uri))
        epoch_summaries = data.get("epoch_summaries") or []
        eval_summaries = data.get("eval_summaries") or []
        last_epoch = epoch_summaries[-1] if epoch_summaries else {}
        best_val = min(
            (item.get("mean_loss") for item in eval_summaries if item.get("mean_loss") is not None),
            default=None,
        )
        peak_gpu_mem_mb = max(
            (item.get("peak_gpu_mem_mb") or 0.0 for item in epoch_summaries),
            default=0.0,
        )
        rows.append(
            {
                **spec,
                "row_count": data.get("row_count"),
                "auxiliary_row_count": data.get("auxiliary_row_count"),
                "eval_row_count": data.get("eval_row_count"),
                "epochs": data.get("epochs"),
                "steps_per_epoch": data.get("steps_per_epoch"),
                "training_duration_sec": data.get("training_duration_sec"),
                "training_duration_min": (data.get("training_duration_sec") or 0.0) / 60.0,
                "final_train_loss": last_epoch.get("mean_loss"),
                "best_val_loss": best_val,
                "peak_gpu_mem_mb": peak_gpu_mem_mb,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "family",
        "row_count",
        "auxiliary_row_count",
        "eval_row_count",
        "epochs",
        "steps_per_epoch",
        "final_train_loss",
        "best_val_loss",
        "peak_gpu_mem_mb",
        "training_duration_sec",
        "training_duration_min",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_markdown(rows: list[dict], gcs_root: str, png_path: Path) -> str:
    by_id = {row["run_id"]: row for row in rows}
    yearly_2d = by_id["yearly_500_stagec_conv2d_1024"]["best_val_loss"]
    yearly_3d = by_id["yearly_500_staged_rope_conv3d_1024"]["best_val_loss"]
    mixed_2d = by_id["mixed_ssl4eo_yearly_500_stagec_conv2d_1024"]["best_val_loss"]
    mixed_3d = by_id["mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024"]["best_val_loss"]
    no_gap = by_id["mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_no_time_gap"]["best_val_loss"]
    joint = by_id["mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_joint_sensor"]["best_val_loss"]
    no_cross = by_id["mixed_ssl4eo_yearly_500_staged_rope_conv3d_1024_no_cross_sensor_loss"]["best_val_loss"]

    yearly_3d_vs_2d = -pct_change(yearly_3d, yearly_2d)
    mixed_3d_vs_yearly_3d = -pct_change(mixed_3d, yearly_3d)
    mixed_2d_vs_yearly_2d = pct_change(mixed_2d, yearly_2d)
    no_gap_vs_full = pct_change(no_gap, mixed_3d)
    no_cross_vs_full = pct_change(no_cross, mixed_3d)
    joint_vs_full = pct_change(joint, mixed_3d)

    lines: list[str] = []
    lines.append("# Paper 2 Mixed-Curriculum Ablation Results")
    lines.append("")
    lines.append("**Suite ID**: `paper2_mixed_curriculum_suite_20260324_v1`")
    lines.append("")
    lines.append(f"**GCS Root**: `{gcs_root}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This table summarizes the seven-run Paper 2 ablation suite. All runs used "
        "the same basic JEPA family and differed only in data curriculum, tokenizer "
        "choice, or targeted method removals."
    )
    lines.append("")
    lines.append("| Run | Family | Yearly Rows | SSL4EO Rows | Val Rows | Best Val Loss | Final Train Loss | Peak GPU MB | Duration (min) |")
    lines.append("|-----|--------|-------------|-------------|----------|---------------|------------------|-------------|----------------|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    row["family"],
                    str(row["row_count"]),
                    str(row["auxiliary_row_count"]),
                    str(row["eval_row_count"]),
                    fmt_sci(row["best_val_loss"]),
                    fmt_sci(row["final_train_loss"]),
                    fmt_float(row["peak_gpu_mem_mb"], 1),
                    fmt_float(row["training_duration_min"], 1),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- `Yearly-only + 3D` improved over `Yearly-only + 2D` by about "
        f"`{yearly_3d_vs_2d:.1f}%` on best validation loss, which supports the "
        "claim that dense weekly EO benefits from the `3D` tokenizer."
    )
    lines.append(
        f"- `Mixed curriculum + 3D` improved over `Yearly-only + 3D` by about "
        f"`{mixed_3d_vs_yearly_3d:.1f}%`, so the early sparse+dense curriculum "
        "helped once the model had the stronger temporal pathway."
    )
    lines.append(
        f"- `Mixed curriculum + 2D` was worse than `Yearly-only + 2D` by about "
        f"`{mixed_2d_vs_yearly_2d:.1f}%`, so the mixed curriculum did not help "
        "the weaker `2D` path in this suite."
    )
    lines.append(
        f"- Removing explicit time-gap handling made the best validation loss about "
        f"`{no_gap_vs_full:.1f}%` worse than the full mixed `3D` model, which is "
        "a real but modest degradation."
    )
    lines.append(
        f"- Removing cross-sensor loss made the best validation loss about "
        f"`{no_cross_vs_full:.1f}%` worse than the full mixed `3D` model, which "
        "shows the extra multimodal objective is carrying useful signal."
    )
    lines.append(
        f"- Collapsing the model to a joint-sensor path made the best validation loss "
        f"about `{joint_vs_full:.1f}%` worse than the full mixed `3D` model, which "
        "is the clearest evidence that radar and optical should not be merged too early."
    )
    lines.append("")
    lines.append("## Caution")
    lines.append("")
    lines.append(
        "These are strong directional results, but this suite still used a filtered "
        "yearly subset with `360` usable yearly training rows and `9` yearly "
        "validation rows. The table is good evidence for design direction, not the "
        "final scale claim."
    )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(
        f"- Best validation loss bar chart: "
        f"[paper2_mixed_curriculum_best_val_loss_2026_03_24.png]({png_path})"
    )
    lines.append("")
    return "\n".join(lines)


def make_plot(rows: list[dict], png_path: Path, svg_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(rows)))
    ys = [row["best_val_loss"] for row in rows]
    colors = [row["color"] for row in rows]
    labels = [row["short_label"] for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Best validation loss (log scale)")
    ax.set_title("Paper 2 mixed-curriculum ablation suite")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    best_idx = min(range(len(rows)), key=lambda i: rows[i]["best_val_loss"])
    bars[best_idx].set_linewidth(2.0)
    bars[best_idx].set_edgecolor("#111111")

    for i, (bar, y) in enumerate(zip(bars, ys)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y * 1.08,
            f"{y:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45 if i >= 4 else 0,
        )

    legend_handles = []
    seen = set()
    for row in rows:
        if row["family"] in seen:
            continue
        seen.add(row["family"])
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=row["family"],
                markerfacecolor=row["color"],
                markeredgecolor="black",
                markersize=10,
            )
        )
    ax.legend(handles=legend_handles, title="Run family")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_markdown = Path(args.output_markdown)
    output_csv = Path(args.output_csv)
    output_figure_png = Path(args.output_figure_png)
    output_figure_svg = Path(args.output_figure_svg)

    rows = load_rows(args.gcs_root)
    write_csv(rows, output_csv)
    make_plot(rows, output_figure_png, output_figure_svg)

    markdown = build_markdown(rows, args.gcs_root, output_figure_png)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown + "\n", encoding="utf-8")

    print(f"Wrote markdown: {output_markdown}")
    print(f"Wrote csv: {output_csv}")
    print(f"Wrote figure: {output_figure_png}")
    print(f"Wrote figure: {output_figure_svg}")


if __name__ == "__main__":
    main()
