#!/usr/bin/env python3
"""Inspect and render commands for registered Earth World Model experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "earth_world_model" / "experiments" / "registry.yaml"
TRAIN_SCRIPT_PATH = REPO_ROOT / "earth_world_model" / "train_tpu.py"
PROBE_SCRIPT_PATH = REPO_ROOT / "earth_world_model" / "scripts" / "run_phase3_probe_gpu_vm.sh"
COMPARE_IGNORE_KEYS = {
    "runtime.checkpoint_dir",
    "data.root_dir",
    "data.index_path",
    "eval.data.index_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and render registered EWM experiments.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    list_parser = subparsers.add_parser("list", help="List experiment ids and readiness.")
    list_parser.add_argument("--status", default=None, help="Filter by registry status.")

    show_parser = subparsers.add_parser("show", help="Show one experiment with paper fields and config summary.")
    show_parser.add_argument("experiment_id")
    show_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    compare_parser = subparsers.add_parser("compare", help="Compare two registered experiments.")
    compare_parser.add_argument("left_experiment_id")
    compare_parser.add_argument("right_experiment_id")

    command_parser = subparsers.add_parser("command", help="Print a shell snippet for launching an experiment.")
    command_parser.add_argument("experiment_id")
    command_parser.add_argument("--gcs-data-uri", default=None)
    command_parser.add_argument("--gcs-run-uri", default=None)
    command_parser.add_argument("--local-run-root", default=None)
    command_parser.add_argument("--local-data-root", default=None)
    command_parser.add_argument("--local-checkpoint-dir", default=None)
    command_parser.add_argument("--run-log-path", default=None)
    command_parser.add_argument("--resume-from-basename", default=None)
    command_parser.add_argument("--resume-from", default=None)
    command_parser.add_argument("--dense-index-path", default=None)
    command_parser.add_argument("--dense-eval-index-path", default=None)
    command_parser.add_argument("--checkpoint-dir", default=None)
    command_parser.add_argument("--backend", default="cuda")
    command_parser.add_argument(
        "--data-access-mode",
        default="auto",
        choices=("auto", "localdisk", "localsync", "gcsfuse", "gcs_direct"),
    )
    command_parser.add_argument("--skip-data-sync", action="store_true")
    command_parser.add_argument("--skip-setup-gpu", action="store_true")
    command_parser.add_argument("--use-gcs-data-direct", action="store_true")
    command_parser.add_argument("--disable-continuous-gcs-sync", action="store_true")

    probe_parser = subparsers.add_parser("probe-command", help="Print a shell snippet for the standard probe step.")
    probe_parser.add_argument("experiment_id")
    probe_parser.add_argument("--gcs-run-uri", default=None)
    probe_parser.add_argument("--gcs-probe-uri", default=None)
    probe_parser.add_argument("--checkpoint-basename", default=None)
    probe_parser.add_argument("--probe-task", default="landcover_forest__regr")
    probe_parser.add_argument("--probe-network", default="teacher")
    probe_parser.add_argument("--local-run-root", default=None)
    probe_parser.add_argument("--local-checkpoint-dir", default=None)

    return parser.parse_args()


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Registry is malformed: {REGISTRY_PATH}")
    return payload


def experiments_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    experiments = registry.get("experiments", [])
    indexed: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        experiment_id = experiment["id"]
        indexed[experiment_id] = experiment
    return indexed


def repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_mapping(child, child_prefix))
        return flattened
    flattened[prefix] = value
    return flattened


def shell_quote(value: str, *, raw: bool = False) -> str:
    if raw:
        return value
    return shlex.quote(value)


def shell_placeholder(name: str) -> str:
    return "${%s:?set %s}" % (name, name)


def load_experiment_config(experiment: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    config_path_value = experiment.get("config_path")
    if not config_path_value:
        return None, {}
    config_path = repo_path(config_path_value)
    if not config_path.exists():
        return config_path, {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config is malformed: {config_path}")
    return config_path, config


def important_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    flattened = flatten_mapping(config)
    summary: dict[str, Any] = {}
    for key in sorted(flattened):
        if key in COMPARE_IGNORE_KEYS:
            continue
        summary[key] = flattened[key]
    return summary


def print_list(registry: dict[str, Any], status_filter: str | None) -> int:
    rows = registry.get("experiments", [])
    header = f"{'experiment_id':42} {'status':32} {'launcher':18} description"
    print(header)
    print("-" * len(header))
    for row in rows:
        status = row.get("status", "")
        if status_filter and status != status_filter:
            continue
        launcher_kind = row.get("launcher", {}).get("kind", "")
        print(
            f"{row['id']:42} {status:32} {launcher_kind:18} {row.get('description', '')}"
        )
    return 0


def show_experiment(experiment: dict[str, Any], *, as_json: bool) -> int:
    config_path, config = load_experiment_config(experiment)
    payload = {
        "id": experiment["id"],
        "status": experiment.get("status"),
        "kind": experiment.get("kind"),
        "description": experiment.get("description"),
        "resume_from_experiment": experiment.get("resume_from_experiment"),
        "launcher": experiment.get("launcher", {}),
        "config_path": str(config_path) if config_path else None,
        "paper": experiment.get("paper", {}),
        "config_summary": important_config_summary(config),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"id: {payload['id']}")
    print(f"status: {payload['status']}")
    print(f"kind: {payload['kind']}")
    print(f"description: {payload['description']}")
    if payload["resume_from_experiment"]:
        print(f"resume_from_experiment: {payload['resume_from_experiment']}")
    if payload["config_path"]:
        print(f"config_path: {payload['config_path']}")
    print("paper:")
    for key, value in sorted(payload["paper"].items()):
        print(f"  {key}: {json.dumps(value, sort_keys=True)}")
    if payload["config_summary"]:
        print("config_summary:")
        for key, value in payload["config_summary"].items():
            print(f"  {key}: {json.dumps(value, sort_keys=True)}")
    return 0


def compare_experiments(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_config_path, left_config = load_experiment_config(left)
    right_config_path, right_config = load_experiment_config(right)
    left_paper = flatten_mapping(left.get("paper", {}), prefix="paper")
    right_paper = flatten_mapping(right.get("paper", {}), prefix="paper")
    left_flat = important_config_summary(left_config)
    right_flat = important_config_summary(right_config)

    print(f"left:  {left['id']}")
    if left_config_path:
        print(f"  config: {left_config_path}")
    print(f"right: {right['id']}")
    if right_config_path:
        print(f"  config: {right_config_path}")

    print("\npaper_differences:")
    paper_keys = sorted(set(left_paper) | set(right_paper))
    paper_diff_count = 0
    for key in paper_keys:
        left_value = left_paper.get(key)
        right_value = right_paper.get(key)
        if left_value == right_value:
            continue
        paper_diff_count += 1
        print(f"  {key}: {json.dumps(left_value, sort_keys=True)} -> {json.dumps(right_value, sort_keys=True)}")
    if paper_diff_count == 0:
        print("  none")

    print("\nconfig_differences:")
    config_keys = sorted(set(left_flat) | set(right_flat))
    config_diff_count = 0
    for key in config_keys:
        left_value = left_flat.get(key)
        right_value = right_flat.get(key)
        if left_value == right_value:
            continue
        config_diff_count += 1
        print(f"  {key}: {json.dumps(left_value, sort_keys=True)} -> {json.dumps(right_value, sort_keys=True)}")
    if config_diff_count == 0:
        print("  none")
    return 0


def render_export(name: str, value: str, *, raw: bool = False) -> str:
    return f"export {name}={shell_quote(value, raw=raw)}"


def render_phase2_command(experiment: dict[str, Any], args: argparse.Namespace) -> str:
    config_path = repo_path(experiment["config_path"])
    script_path = repo_path(experiment["launcher"]["script_path"])
    data_access_mode = args.data_access_mode
    if args.use_gcs_data_direct:
        data_access_mode = "gcs_direct"
    lines = [
        render_export("CONFIG_PATH", str(config_path)),
        render_export("GCS_DATA_URI", args.gcs_data_uri or shell_placeholder("GCS_DATA_URI"), raw=args.gcs_data_uri is None),
        render_export("GCS_RUN_URI", args.gcs_run_uri or shell_placeholder("GCS_RUN_URI"), raw=args.gcs_run_uri is None),
        render_export("DATA_ACCESS_MODE", data_access_mode),
    ]
    if args.local_run_root:
        lines.append(render_export("LOCAL_RUN_ROOT", args.local_run_root))
    if args.local_data_root:
        lines.append(render_export("LOCAL_DATA_ROOT", args.local_data_root))
    if args.local_checkpoint_dir:
        lines.append(render_export("LOCAL_CHECKPOINT_DIR", args.local_checkpoint_dir))
    if args.run_log_path:
        lines.append(render_export("RUN_LOG_PATH", args.run_log_path))
    if args.resume_from_basename:
        lines.append(render_export("RESUME_FROM_BASENAME", args.resume_from_basename))
    if args.skip_data_sync:
        lines.append(render_export("SKIP_DATA_SYNC", "1"))
    if args.skip_setup_gpu:
        lines.append(render_export("SKIP_SETUP_GPU", "1"))
    if args.disable_continuous_gcs_sync:
        lines.append(render_export("CONTINUOUS_GCS_SYNC", "0"))
    lines.append(f"bash {shell_quote(str(script_path))}")
    return "\n".join(lines)


def resolve_resume_path(experiment: dict[str, Any], explicit_resume_from: str | None) -> tuple[str, bool]:
    if explicit_resume_from:
        return explicit_resume_from, False
    parent_id = experiment.get("resume_from_experiment")
    if not parent_id:
        return "", False
    return shell_placeholder(f"RESUME_FROM_{parent_id.upper()}".replace("-", "_")), True


def render_direct_train_command(experiment: dict[str, Any], args: argparse.Namespace) -> str:
    config_path = repo_path(experiment["config_path"])
    resume_from_value, resume_is_placeholder = resolve_resume_path(experiment, args.resume_from)
    lines = [
        render_export("PYTHONPATH", f"{REPO_ROOT / 'earth_world_model' / 'src'}:{REPO_ROOT}"),
        render_export("PYTHONUNBUFFERED", "1"),
        render_export("EWM_BACKEND", args.backend),
        render_export(
            "EWM_DENSE_INDEX_PATH",
            args.dense_index_path or shell_placeholder("EWM_DENSE_INDEX_PATH"),
            raw=args.dense_index_path is None,
        ),
        render_export(
            "EWM_DENSE_EVAL_INDEX_PATH",
            args.dense_eval_index_path or shell_placeholder("EWM_DENSE_EVAL_INDEX_PATH"),
            raw=args.dense_eval_index_path is None,
        ),
        render_export(
            "EWM_CHECKPOINT_DIR",
            args.checkpoint_dir or shell_placeholder("EWM_CHECKPOINT_DIR"),
            raw=args.checkpoint_dir is None,
        ),
    ]
    command_parts = [
        "python",
        shlex.quote(str(TRAIN_SCRIPT_PATH)),
        "--config",
        shlex.quote(str(config_path)),
    ]
    if resume_from_value:
        command_parts.extend(
            [
                "--resume-from",
                shell_quote(resume_from_value, raw=resume_is_placeholder),
            ]
        )
    lines.append(" ".join(command_parts))
    return "\n".join(lines)


def render_command(experiment: dict[str, Any], args: argparse.Namespace) -> str:
    launcher_kind = experiment.get("launcher", {}).get("kind")
    if launcher_kind == "phase2_gpu_vm":
        return render_phase2_command(experiment, args)
    if launcher_kind == "direct_train":
        return render_direct_train_command(experiment, args)
    if launcher_kind == "planned":
        return "# Planned experiment. The registry entry exists, but no launcher has been implemented yet."
    raise SystemExit(f"Unsupported launcher kind: {launcher_kind}")


def render_probe_command(args: argparse.Namespace) -> str:
    lines = [
        render_export("GCS_RUN_URI", args.gcs_run_uri or shell_placeholder("GCS_RUN_URI"), raw=args.gcs_run_uri is None),
        render_export("PROBE_TASK", args.probe_task),
        render_export("PROBE_NETWORK", args.probe_network),
    ]
    if args.gcs_probe_uri:
        lines.append(render_export("GCS_PROBE_URI", args.gcs_probe_uri))
    if args.checkpoint_basename:
        lines.append(render_export("CHECKPOINT_BASENAME", args.checkpoint_basename))
    if args.local_run_root:
        lines.append(render_export("LOCAL_RUN_ROOT", args.local_run_root))
    if args.local_checkpoint_dir:
        lines.append(render_export("LOCAL_CHECKPOINT_DIR", args.local_checkpoint_dir))
    lines.append(f"bash {shell_quote(str(PROBE_SCRIPT_PATH))}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    registry = load_registry()
    experiment_index = experiments_by_id(registry)

    if args.command_name == "list":
        return print_list(registry, args.status)

    if args.command_name == "show":
        experiment = experiment_index.get(args.experiment_id)
        if experiment is None:
            raise SystemExit(f"Unknown experiment id: {args.experiment_id}")
        return show_experiment(experiment, as_json=args.json)

    if args.command_name == "compare":
        left = experiment_index.get(args.left_experiment_id)
        right = experiment_index.get(args.right_experiment_id)
        if left is None:
            raise SystemExit(f"Unknown experiment id: {args.left_experiment_id}")
        if right is None:
            raise SystemExit(f"Unknown experiment id: {args.right_experiment_id}")
        return compare_experiments(left, right)

    if args.command_name == "command":
        experiment = experiment_index.get(args.experiment_id)
        if experiment is None:
            raise SystemExit(f"Unknown experiment id: {args.experiment_id}")
        print(render_command(experiment, args))
        return 0

    if args.command_name == "probe-command":
        if args.experiment_id not in experiment_index:
            raise SystemExit(f"Unknown experiment id: {args.experiment_id}")
        print(render_probe_command(args))
        return 0

    raise SystemExit(f"Unsupported command: {args.command_name}")


if __name__ == "__main__":
    raise SystemExit(main())
