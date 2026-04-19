#!/usr/bin/env python3
"""Monitor RFD3 training progress and compare experiments.

Reads experiment logs and CSV metrics from local_runs/ to show:
- Loss curves across epochs
- Comparison between different training configs
- Checkpoint availability

Usage:
  python monitor_training.py                     # Show all experiments
  python monitor_training.py --name high_quality  # Filter by name
  python monitor_training.py --compare            # Side-by-side comparison
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_DIR / "foundry" / "models" / "rfd3" / "local_runs" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--name", help="Filter experiments by name substring.")
    parser.add_argument("--compare", action="store_true", help="Compare all experiments side by side.")
    parser.add_argument("--tail", type=int, default=20, help="Lines from end of log to show.")
    return parser.parse_args()


def find_experiments(runs_dir: Path, name_filter: str | None = None) -> list[dict]:
    """Find all experiment directories with their metadata."""
    experiments = []
    if not runs_dir.exists():
        return experiments

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if name_filter and name_filter not in exp_dir.name:
            continue

        # Find the latest run within the experiment
        run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        if not run_dirs:
            continue

        latest_run = run_dirs[0]
        log_file = latest_run / "experiment.log"
        ckpt_dir = latest_run / "ckpt"

        # Parse checkpoints
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []

        # Parse log for latest epoch info
        epoch_info = parse_log_epochs(log_file) if log_file.exists() else []

        # CSV metrics
        csv_dir = latest_run / "csv"
        metrics = parse_csv_metrics(csv_dir) if csv_dir.exists() else {}

        experiments.append({
            "name": exp_dir.name,
            "run_dir": str(latest_run),
            "log_file": str(log_file) if log_file.exists() else None,
            "checkpoints": [str(c) for c in ckpts],
            "n_checkpoints": len(ckpts),
            "epochs": epoch_info,
            "metrics": metrics,
        })

    return experiments


def parse_log_epochs(log_file: Path) -> list[dict]:
    """Extract epoch summaries from the training log."""
    epochs = []
    current_epoch = {}

    with log_file.open("r", errors="replace") as f:
        for line in f:
            # Match epoch summary lines
            match = re.search(r"Epoch (\d+) Summary", line)
            if match:
                if current_epoch:
                    epochs.append(current_epoch)
                current_epoch = {"epoch": int(match.group(1))}

            # Match loss values in epoch summaries
            for metric in ["total_loss", "mse_loss_mean", "seq_recovery", "mean_lddt"]:
                match = re.search(rf"<Train> Mean {metric}\s+│\s+([\d.]+|nan)", line)
                if match and current_epoch:
                    val = match.group(1)
                    current_epoch[metric] = float(val) if val != "nan" else None

            # Match batch progress
            match = re.search(r"Epoch (\d+) Batch (\d+).*?([\d.]+)%", line)
            if match:
                current_epoch = {
                    "epoch": int(match.group(1)),
                    "batch": int(match.group(2)),
                    "progress": float(match.group(3)),
                }

    if current_epoch:
        epochs.append(current_epoch)

    return epochs


def parse_csv_metrics(csv_dir: Path) -> dict:
    """Parse CSV metric files from the logger."""
    metrics = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        try:
            with csv_file.open("r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    metrics[csv_file.stem] = rows
        except Exception:
            pass
    return metrics


def display_experiment(exp: dict) -> None:
    """Display a single experiment's status."""
    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp['name']}")
    print(f"  Run dir: {exp['run_dir']}")
    print(f"  Checkpoints: {exp['n_checkpoints']}")
    if exp["checkpoints"]:
        for ckpt in exp["checkpoints"][-3:]:
            print(f"    {Path(ckpt).name}")

    if exp["epochs"]:
        latest = exp["epochs"][-1]
        print(f"\n  Latest progress:")
        if "progress" in latest:
            print(f"    Epoch {latest['epoch']}, Batch {latest.get('batch', '?')}, Progress: {latest['progress']:.1f}%")

        # Show epoch history
        completed = [e for e in exp["epochs"] if "total_loss" in e]
        if completed:
            print(f"\n  {'Epoch':>6} {'Total Loss':>11} {'MSE Loss':>10} {'Seq Recov':>10} {'LDDT':>8}")
            print(f"  {'-' * 50}")
            for e in completed:
                total = f"{e['total_loss']:.4f}" if e.get('total_loss') is not None else "nan"
                mse = f"{e['mse_loss_mean']:.4f}" if e.get('mse_loss_mean') is not None else "nan"
                sr = f"{e['seq_recovery']:.4f}" if e.get('seq_recovery') is not None else "nan"
                lddt = f"{e['mean_lddt']:.4f}" if e.get('mean_lddt') is not None else "nan"
                print(f"  {e['epoch']:>6} {total:>11} {mse:>10} {sr:>10} {lddt:>8}")


def display_comparison(experiments: list[dict]) -> None:
    """Side-by-side comparison of experiments."""
    print(f"\n{'Experiment':<45} {'Epochs':>7} {'Ckpts':>6} {'Last Loss':>10} {'Last Seq Rec':>12}")
    print("-" * 85)
    for exp in experiments:
        completed = [e for e in exp["epochs"] if "total_loss" in e]
        n_epochs = len(completed)
        last_loss = f"{completed[-1]['total_loss']:.4f}" if completed and completed[-1].get('total_loss') is not None else "-"
        last_sr = f"{completed[-1]['seq_recovery']:.4f}" if completed and completed[-1].get('seq_recovery') is not None else "-"
        print(f"  {exp['name']:<43} {n_epochs:>7} {exp['n_checkpoints']:>6} {last_loss:>10} {last_sr:>12}")


def run() -> int:
    args = parse_args()

    experiments = find_experiments(args.runs_dir, args.name)

    if not experiments:
        print(f"No experiments found in {args.runs_dir}")
        if args.name:
            print(f"  (filter: '{args.name}')")
        return 0

    print(f"Found {len(experiments)} experiments.", flush=True)

    if args.compare:
        display_comparison(experiments)
    else:
        for exp in experiments:
            display_experiment(exp)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
