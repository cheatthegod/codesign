#!/usr/bin/env python3
"""Run RFD3 inference with enzyme-conditioned checkpoints.

Wraps the standard RFD3 inference engine to handle checkpoints trained
with val=null (which the base engine doesn't support out of the box).

Usage:
  python run_enzyme_inference.py \
    --ckpt epoch-0018.ckpt \
    --inputs inference_inputs.json \
    --out-dir output/conditioned \
    --n-batches 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FOUNDRY_DIR = PROJECT_DIR / "foundry"

# Add to PYTHONPATH
for p in [
    FOUNDRY_DIR / "src",
    FOUNDRY_DIR / "models" / "rfd3" / "src",
    FOUNDRY_DIR / "models" / "rf3" / "src",
    FOUNDRY_DIR / "models" / "mpnn" / "src",
]:
    sys.path.insert(0, str(p))

import os
os.environ.setdefault("PDB_MIRROR_PATH", str(FOUNDRY_DIR / "models/rfd3/local_data/pdb_mirror"))
os.environ.setdefault("CCD_MIRROR_PATH", str(FOUNDRY_DIR / "models/rfd3/local_data/ccd_mirror"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--inputs", type=Path, required=True, help="JSON file with design specifications.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--n-batches", type=int, default=2, help="Number of design batches per task.")
    parser.add_argument("--diffusion-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--low-memory", action="store_true", help="Enable low memory mode.")
    return parser.parse_args()


def run() -> int:
    args = parse_args()

    import torch
    from toolz import merge

    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from foundry.inference_engines.base import BaseInferenceEngine

    # Monkey-patch _construct_pipeline to handle val=null
    _original_construct_pipeline = BaseInferenceEngine._construct_pipeline

    def _patched_construct_pipeline(self, cfg):
        # If val is null, use the first train dataset's transform instead
        if cfg.get("datasets", {}).get("val") is None:
            train = cfg.get("datasets", {}).get("train", {})
            if train:
                first_key = next(iter(train))
                train_transform = dict(train[first_key].get("dataset", {}).get("transform", {}))
                # Override for inference mode
                train_transform["is_inference"] = True
                train_transform["return_atom_array"] = True
                val_entry = {
                    first_key: {
                        "dataset": {
                            "transform": train_transform,
                        },
                    }
                }
                if "datasets" not in cfg:
                    cfg["datasets"] = {}
                cfg["datasets"]["val"] = val_entry
                print(f"  Injected val config from train dataset '{first_key}' (is_inference=True)", flush=True)

        return _original_construct_pipeline(self, cfg)

    BaseInferenceEngine._construct_pipeline = _patched_construct_pipeline

    print(f"Checkpoint: {args.ckpt}", flush=True)
    print(f"Inputs: {args.inputs}", flush=True)
    print(f"Output: {args.out_dir}", flush=True)
    print(f"Batches: {args.n_batches}, Diffusion batch size: {args.diffusion_batch_size}", flush=True)

    config = RFD3InferenceConfig(
        ckpt_path=str(args.ckpt),
        diffusion_batch_size=args.diffusion_batch_size,
        seed=args.seed,
        skip_existing=True,
        devices_per_node=args.devices,
        inference_sampler={"num_timesteps": args.num_timesteps},
        low_memory_mode=args.low_memory,
    )

    engine = RFD3InferenceEngine(**config)

    # Pass file path (engine expects path, not dict)
    inputs_path = str(args.inputs.resolve())
    with args.inputs.open("r") as f:
        n_tasks = len(json.load(f))

    print(f"Running {n_tasks} design tasks x {args.n_batches} batches ...", flush=True)
    engine.run(
        inputs=inputs_path,
        n_batches=args.n_batches,
        out_dir=str(args.out_dir),
    )

    print(f"\nDone. Outputs in {args.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
