#!/usr/bin/env python3
"""Patch cached .pt files to add correct gt_seq from PDB structures.

The original cache generation masked all protein sequences (mask_sequence=True),
causing gt_seq to be all-UNK. This script reads the real amino acid sequences
from the PDB mmCIF files and injects correct gt_seq_full (token-level) and
gt_seq (protein-only) into each cache file.

Usage:
    python scripts/patch_cache_gt_seq.py \
        --cache-dir local_data/pdb_cache \
        --pdb-dir local_data/pdb_complexes \
        --workers 32
"""

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco.data.constants import PRO_STD_RESIDUES
from disco.data.pdb_complex_adapter import AA1_TO_IDX, UNK_IDX, parse_complex_from_mmcif

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def patch_one(pt_path: str, pdb_dir: str) -> tuple[str, bool, str]:
    """Patch a single .pt cache file with correct gt_seq.

    Returns (filename, success, message).
    """
    fname = os.path.basename(pt_path)
    pdb_id = fname.replace(".pt", "")

    try:
        # Load cache
        cache = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Check if already patched
        if "gt_seq_full" in cache:
            gt = cache["gt_seq_full"]
            if gt is not None and (gt != UNK_IDX).any():
                return (fname, True, "already patched")

        # Find PDB file
        cif_path = os.path.join(pdb_dir, f"{pdb_id}.cif.gz")
        if not os.path.exists(cif_path):
            cif_path = os.path.join(pdb_dir, f"{pdb_id}.cif")
            if not os.path.exists(cif_path):
                return (fname, False, "PDB file not found")

        # Parse real sequences
        info = parse_complex_from_mmcif(cif_path)
        if info is None:
            return (fname, False, "PDB parse failed")
        chains = info["protein_chains"]
        if not chains:
            return (fname, False, "no protein chains")

        # Get cache tensors
        prot_mask = cache.get("prot_residue_mask")
        asym_ids = cache.get("asym_id")
        res_idx = cache.get("residue_index")

        if prot_mask is None or asym_ids is None or res_idx is None:
            return (fname, False, "missing required tensors")

        N_token = prot_mask.shape[0]

        # Map unique protein asym_ids → chain index
        prot_asym_unique = asym_ids[prot_mask].unique(sorted=True)
        asym_to_chain = {a.item(): i for i, a in enumerate(prot_asym_unique)}

        # Build token-level gt_seq
        gt_seq_full = torch.full((N_token,), UNK_IDX, dtype=torch.long)
        n_mapped = 0
        for tok_i in range(N_token):
            if not prot_mask[tok_i]:
                continue
            chain_idx = asym_to_chain.get(asym_ids[tok_i].item())
            if chain_idx is not None and chain_idx < len(chains):
                seq = chains[chain_idx]["sequence"]
                pos = res_idx[tok_i].item() - 1  # 1-based → 0-based
                if 0 <= pos < len(seq):
                    aa_idx = AA1_TO_IDX.get(seq[pos], UNK_IDX)
                    gt_seq_full[tok_i] = aa_idx
                    if aa_idx != UNK_IDX:
                        n_mapped += 1

        # Derive protein-only gt_seq
        gt_seq = gt_seq_full[prot_mask]
        n_prot = int(prot_mask.sum().item())

        # Save back
        cache["gt_seq_full"] = gt_seq_full
        cache["gt_seq"] = gt_seq
        torch.save(cache, pt_path)

        n_real = int((gt_seq != UNK_IDX).sum().item())
        return (fname, True, f"patched {n_real}/{n_prot} residues")

    except Exception as e:
        return (fname, False, f"error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Patch cache gt_seq")
    parser.add_argument("--cache-dir", default="local_data/pdb_cache")
    parser.add_argument("--pdb-dir", default="local_data/pdb_complexes")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pt_files = sorted([
        f for f in os.listdir(args.cache_dir) if f.endswith(".pt")
    ])
    logger.info(f"Found {len(pt_files)} cache files to patch")

    if args.dry_run:
        # Test on first 5
        pt_files = pt_files[:5]
        logger.info("Dry run: processing first 5 files only")

    success = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for fname in pt_files:
            pt_path = os.path.join(args.cache_dir, fname)
            fut = executor.submit(patch_one, pt_path, args.pdb_dir)
            futures[fut] = fname

        for i, fut in enumerate(as_completed(futures)):
            fname, ok, msg = fut.result()
            if ok:
                if "already" in msg:
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                if failed <= 20:
                    logger.warning(f"FAILED {fname}: {msg}")

            if (i + 1) % 2000 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(pt_files)} "
                    f"(ok={success}, skip={skipped}, fail={failed})"
                )

    logger.info(
        f"Done: {success} patched, {skipped} already done, {failed} failed "
        f"out of {len(pt_files)} total"
    )


if __name__ == "__main__":
    main()
