from __future__ import annotations

import argparse
from pathlib import Path
from datetime import UTC, datetime
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from COT_enzyme_design.cot_agent.actions.registry import build_default_registry
from COT_enzyme_design.cot_agent.backends import GoldStateBackend
from COT_enzyme_design.cot_agent.inference.runner import InferenceRunner
from COT_enzyme_design.cot_agent.models.policy import DesignAgent
from COT_enzyme_design.cot_agent.schemas.task import TaskSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal COT design agent scaffold.")
    repo_root = REPO_ROOT
    default_task = repo_root / "COT_enzyme_design" / "datasets" / "cot_example_v3" / "tasks" / "task_000001" / "task.json"
    default_run_dir = repo_root / "COT_enzyme_design" / "runs" / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--task-json", type=Path, default=default_task)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    task_spec = TaskSpec.from_file(args.task_json)
    runner = InferenceRunner(
        agent=DesignAgent(),
        registry=build_default_registry(),
        backends={"gold_state": GoldStateBackend()},
        run_dir=args.run_dir,
        max_steps=args.max_steps,
    )
    result = runner.run(task_spec)
    print(f"Run directory: {result.run_dir}")
    print(f"Trace path: {result.trace_path}")
    print(f"CoT path: {result.cot_path}")
    print(f"Final object: {result.final_object_id}")
    print(f"Final PDB: {result.final_pdb_path}")


if __name__ == "__main__":
    main()
