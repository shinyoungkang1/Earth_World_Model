#!/usr/bin/env python3
"""Run the first gas training and scoring pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_step(argv: list[str], repo_root: Path) -> None:
    subprocess.run(argv, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    steps = [
        ["python", "scripts/build_gas_training_table_v0.py"],
        ["python", "scripts/train_gas_baseline_v0.py"],
        ["python", "scripts/score_gas_prospect_cells_v0.py"],
        ["python", "scripts/publish_results_snapshot.py"],
    ]
    for step in steps:
        run_step(step, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
