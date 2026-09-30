"""Shared command-line and logging helpers for staged training scripts."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add the common TensorBoard/W&B command-line options."""
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--logger",
        choices=("tensorboard", "wandb"),
        default="tensorboard",
        help="Metric logger to use.",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        default="unitree_dsc_lab",
        help="Weights & Biases project name.",
    )
    group.add_argument("--run_name", type=str, default="", help="Optional run-name suffix.")


def validate_logger_dependency(logger: str) -> None:
    """Fail before Isaac Sim starts when the selected logger is unavailable."""
    if logger == "wandb" and importlib.util.find_spec("wandb") is None:
        raise ModuleNotFoundError(
            "W&B logging requires the 'wandb' package. Re-run ./unitree_dsc_lab.sh -i "
            "or install wandb in the active environment."
        )


def configure_runner_logging(runner_cfg, args: argparse.Namespace, log_dir: str) -> None:
    """Apply logging CLI overrides to an rsl-rl runner config."""
    runner_cfg.logger = args.logger
    runner_cfg.wandb_project = args.wandb_project
    runner_cfg.run_name = Path(log_dir).name


def create_log_dir(log_root: str, task: str, stage: str, run_name: str) -> str:
    """Create a unique timestamped run directory and return its absolute path."""
    if run_name and (Path(run_name).name != run_name or run_name in {".", ".."}):
        raise ValueError("--run_name must be a single path-safe name, not a path.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{run_name}" if run_name else ""
    run_stem = f"{timestamp}_{stage}{suffix}"
    task_root = Path(log_root).expanduser().resolve() / task

    log_dir = task_root / run_stem
    collision = 2
    while log_dir.exists():
        log_dir = task_root / f"{run_stem}_{collision:02d}"
        collision += 1
    log_dir.mkdir(parents=True)
    print(f"[INFO] Logging run to: {log_dir}")
    return str(log_dir)
