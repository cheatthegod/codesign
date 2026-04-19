from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from COT_enzyme_design.cot_agent.training.prior_train import PriorTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder entrypoint for training the biological prior head.")
    parser.add_argument("--dataset-root", type=Path, required=False)
    args = parser.parse_args()

    trainer = PriorTrainer()
    metrics = trainer.step([])
    print(f"Prior trainer placeholder completed. dataset_root={args.dataset_root} metrics={metrics}")


if __name__ == "__main__":
    main()
