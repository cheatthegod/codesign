"""Preprocess PDB complex features into .pt cache files.

Parallelizes across multiple CPU workers. Each sample is featurized
once through DISCO's full pipeline (CCD lookup + mmCIF parsing +
Featurizer) and saved as a .pt file. During training, samples are
loaded from cache in milliseconds instead of seconds.

Usage:
    python scripts/preprocess_pdb_cache.py \
        --pdb_dir local_data/pdb_complexes \
        --output_dir local_data/pdb_cache \
        --crop_size 384 \
        --workers 16
"""

import argparse
import logging
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco.data.pdb_complex_adapter import PDBComplexDataset
from disco.data.task_manager import TaskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def process_one(args):
    """Process a single PDB complex and save as .pt"""
    idx, sample_meta, crop_size, output_dir = args
    pdb_id = sample_meta["pdb_id"]
    out_path = os.path.join(output_dir, f"{pdb_id}.pt")

    # Skip if already cached
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        return pdb_id, "exists", 0

    try:
        # Create a minimal dataset to process this one sample
        ds = PDBComplexDataset(
            pdb_dir=os.path.dirname(sample_meta["cif_path"]),
            crop_size=crop_size,
            require_ligand=False,  # Process all, filter later
            min_protein_length=10,
            max_protein_length=99999,
        )

        # Find this sample in the dataset
        sample = None
        for i in range(len(ds)):
            if ds.samples[i]["pdb_id"] == pdb_id:
                sample = ds[i]
                break

        if sample is None:
            return pdb_id, "skip", 0

        # Extract only tensor features for caching
        feat = sample["input_feature_dict"]
        cache = {}
        for k, v in feat.items():
            if torch.is_tensor(v):
                cache[k] = v
        cache["_sample_info"] = sample.get("sample_info", {})
        cache["_sample_name"] = sample.get("sample_name", pdb_id)

        torch.save(cache, out_path)
        size_kb = os.path.getsize(out_path) / 1024
        return pdb_id, "ok", size_kb

    except Exception as e:
        return pdb_id, f"error: {str(e)[:80]}", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", default="local_data/pdb_complexes")
    parser.add_argument("--output_dir", default="local_data/pdb_cache")
    parser.add_argument("--mapping_csv", default="local_data/enzyme_train.csv")
    parser.add_argument("--crop_size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build sample list
    pdb_dir = args.pdb_dir
    samples = []
    for fn in sorted(os.listdir(pdb_dir)):
        if fn.endswith(".cif.gz"):
            pdb_id = fn.replace(".cif.gz", "")
            samples.append({
                "pdb_id": pdb_id,
                "cif_path": os.path.join(pdb_dir, fn),
            })

    if args.limit > 0:
        samples = samples[:args.limit]

    # Check already cached
    existing = set()
    for fn in os.listdir(args.output_dir):
        if fn.endswith(".pt"):
            existing.add(fn.replace(".pt", ""))

    to_process = [s for s in samples if s["pdb_id"] not in existing]
    logger.info(f"Total: {len(samples)}, cached: {len(existing)}, remaining: {len(to_process)}")

    if not to_process:
        logger.info("All samples already cached!")
        return

    # Process with multiprocessing
    task_args = [
        (i, s, args.crop_size, args.output_dir)
        for i, s in enumerate(to_process)
    ]

    ok, skip, error = 0, 0, 0
    total_size = 0
    start = time.time()

    # Use single-process PDBComplexDataset to avoid re-creating datasets
    # Actually, for multiprocessing, each worker creates its own
    logger.info(f"Processing {len(to_process)} samples with {args.workers} workers...")

    # Process sequentially but efficiently - create dataset once
    task_manager = TaskManager(transform_masked_ref_pos=True, ref_pos_augment=True)
    ds = PDBComplexDataset(
        pdb_dir=pdb_dir,
        task_manager=task_manager,
        crop_size=args.crop_size,
        require_ligand=False,
        min_protein_length=10,
        max_protein_length=99999,
    )

    # Build pdb_id -> index mapping
    id_to_idx = {}
    for i, s in enumerate(ds.samples):
        id_to_idx[s["pdb_id"]] = i

    for i, sample_meta in enumerate(to_process):
        pdb_id = sample_meta["pdb_id"]
        out_path = os.path.join(args.output_dir, f"{pdb_id}.pt")

        try:
            idx = id_to_idx.get(pdb_id)
            if idx is None:
                skip += 1
                continue

            sample = ds[idx]
            if sample is None:
                skip += 1
                continue

            # Cache tensors
            cache = {}
            for k, v in sample["input_feature_dict"].items():
                if torch.is_tensor(v):
                    cache[k] = v
            cache["_sample_info"] = sample.get("sample_info", {})
            cache["_sample_name"] = sample.get("sample_name", pdb_id)

            torch.save(cache, out_path)
            size_kb = os.path.getsize(out_path) / 1024
            total_size += size_kb
            ok += 1

        except Exception as e:
            error += 1
            if error <= 5:
                logger.warning(f"{pdb_id}: {e}")

        if (i + 1) % 100 == 0 or i + 1 == len(to_process):
            elapsed = time.time() - start
            rate = (ok + skip + error) / max(elapsed, 1)
            remaining = (len(to_process) - i - 1) / max(rate, 0.01)
            logger.info(
                f"[{i+1}/{len(to_process)}] ok={ok} skip={skip} err={error} "
                f"({rate:.1f}/s, ~{remaining/60:.0f}min left, {total_size/1024/1024:.1f}GB)"
            )

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed/60:.1f} min")
    logger.info(f"  Cached: {ok}, Skipped: {skip}, Errors: {error}")
    logger.info(f"  Total size: {total_size/1024/1024:.1f} GB")


if __name__ == "__main__":
    main()
