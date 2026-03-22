#!/usr/bin/env python3
"""Run the first WV horizontal gas pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_step(argv: list[str], repo_root: Path) -> None:
    subprocess.run(argv, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="wv_horizontal_statewide")
    parser.add_argument("--label-column", default="label_f12_ge_2000000")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    model_path = repo_root / "models" / args.basin_id / f"gas_baseline_wv_v0_{args.label_column}.joblib"
    steps = [
        ["python", "scripts/build_wv_training_table_v0.py", "--basin-id", args.basin_id],
        ["python", "scripts/train_wv_baseline_v0.py", "--basin-id", args.basin_id, "--label-column", args.label_column],
        ["python", "scripts/score_wv_prospect_cells_v0.py", "--basin-id", args.basin_id, "--model-path", str(model_path)],
        ["python", "scripts/publish_results_snapshot.py"],
    ]
    for step in steps:
        run_step(step, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
