"""Preprocess a chunk of PDB complexes into .pt cache files.

Usage: python scripts/preprocess_chunk.py --chunk_id 0 --timeout 300
"""
import argparse
import logging
import os
import signal
import sys
import time

import torch


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Sample processing timed out")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco.data.pdb_complex_adapter import PDBComplexDataset
from disco.data.task_manager import TaskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunk_file", type=str, default=None)
    parser.add_argument("--pdb_dir", default="local_data/pdb_complexes")
    parser.add_argument("--output_dir", default="local_data/pdb_cache")
    parser.add_argument("--crop_size", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=300, help="Per-sample timeout in seconds (default 300s)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load chunk
    chunk_file = args.chunk_file or f"local_data/pdb_ids_chunk_{args.chunk_id}.txt"
    with open(chunk_file) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Chunk {args.chunk_id}: {len(pdb_ids)} samples")

    # Skip already cached
    existing = set(fn.replace(".pt", "") for fn in os.listdir(args.output_dir) if fn.endswith(".pt"))
    to_process = [pid for pid in pdb_ids if pid not in existing]
    logger.info(f"Already cached: {len(pdb_ids) - len(to_process)}, remaining: {len(to_process)}")

    if not to_process:
        logger.info("All done!")
        return

    # Create dataset once
    tm = TaskManager(transform_masked_ref_pos=True, ref_pos_augment=True)
    ds = PDBComplexDataset(
        pdb_dir=args.pdb_dir, task_manager=tm, crop_size=args.crop_size,
        require_ligand=False, min_protein_length=10, max_protein_length=99999,
    )
    id_to_idx = {s["pdb_id"]: i for i, s in enumerate(ds.samples)}

    ok, skip, error = 0, 0, 0
    total_kb = 0
    start = time.time()

    signal.signal(signal.SIGALRM, _timeout_handler)

    for i, pdb_id in enumerate(to_process):
        out_path = os.path.join(args.output_dir, f"{pdb_id}.pt")
        try:
            idx = id_to_idx.get(pdb_id)
            if idx is None:
                skip += 1
                continue

            signal.alarm(args.timeout)
            sample = ds[idx]
            signal.alarm(0)

            if sample is None:
                skip += 1
                continue

            cache = {k: v for k, v in sample["input_feature_dict"].items() if torch.is_tensor(v)}
            cache["_sample_info"] = sample.get("sample_info", {})
            cache["_sample_name"] = sample.get("sample_name", pdb_id)
            torch.save(cache, out_path)
            total_kb += os.path.getsize(out_path) / 1024
            ok += 1
        except TimeoutError:
            signal.alarm(0)
            skip += 1
            logger.warning(f"{pdb_id}: timed out after {args.timeout}s, skipping")
        except Exception as e:
            signal.alarm(0)
            error += 1

        if (i + 1) % 200 == 0 or i + 1 == len(to_process):
            elapsed = time.time() - start
            rate = (ok + skip + error) / max(elapsed, 1)
            eta = (len(to_process) - i - 1) / max(rate, 0.01) / 60
            logger.info(
                f"Chunk {args.chunk_id} [{i+1}/{len(to_process)}] "
                f"ok={ok} skip={skip} err={error} "
                f"({rate:.1f}/s, ~{eta:.0f}min, {total_kb/1024/1024:.1f}GB)"
            )

    logger.info(f"Chunk {args.chunk_id} done: ok={ok}, skip={skip}, err={error}, {total_kb/1024/1024:.1f}GB")


if __name__ == "__main__":
    main()
