#!/usr/bin/env python3
"""Layer 2: Compute geometric features from AlphaFold PDB structures.

Extracts per-protein structural statistics that can be used for conditioning,
filtering, and analysis without requiring real complex structures:

- pLDDT statistics (mean, min, fraction below thresholds)
- Secondary structure fractions (helix, sheet, loop)
- is_non_loopy flag (loop fraction < 0.30)
- SASA / burial statistics (mean per-residue SASA, buried/exposed fractions)
- Chain count and residue count from structure

Outputs a standalone parquet keyed by accession, designed to be joined with
the Layer 1 enriched base via a simple merge.

Requires: biotite (tested with 1.6.0), numpy, pandas, pyarrow.
Run with the polaris-env conda environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports — biotite is only available in polaris-env
# ---------------------------------------------------------------------------


def _import_biotite():
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    from biotite.structure import annotate_sse, sasa

    return struc, pdb_io, annotate_sse, sasa


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_ENRICHED_PATH = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_base.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"

# SASA thresholds (Å²) for per-residue burial classification
BURIED_THRESHOLD = 10.0
EXPOSED_THRESHOLD = 40.0

# pLDDT thresholds
PLDDT_LOW = 50.0
PLDDT_MEDIUM = 70.0

# Non-loopy threshold (consistent with RFD3)
LOOP_FRACTION_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute structural features from AlphaFold PDB files."
    )
    parser.add_argument(
        "--enriched-parquet",
        type=Path,
        default=DEFAULT_ENRICHED_PATH,
        help="Path to prot2text_enriched_base.parquet (provides accession → path mapping).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for structure_features.parquet.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Only process these splits (e.g. --splits test). Default: all splits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers. 0 = auto (cpu_count - 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only this many proteins (for debugging). 0 = all.",
    )
    parser.add_argument(
        "--sasa-points",
        type=int,
        default=100,
        help="Number of sample points for Shrake-Rupley SASA calculation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip accessions already present in the output file.",
    )
    return parser.parse_args()


def compute_one_protein(args_tuple: tuple[str, str, int]) -> dict[str, Any] | None:
    """Compute features for a single protein structure. Runs in worker process."""
    accession, pdb_path, sasa_points = args_tuple

    try:
        struc_mod, pdb_io, annotate_sse, sasa_fn = _import_biotite()

        pdb_file = pdb_io.PDBFile.read(pdb_path)
        atoms = pdb_file.get_structure(model=1, extra_fields=["b_factor", "occupancy"])

        # --- Basic stats ---
        chain_ids = np.unique(atoms.chain_id)
        num_chains = len(chain_ids)

        ca_mask = atoms.atom_name == "CA"
        ca_atoms = atoms[ca_mask]
        num_residues = len(ca_atoms)

        if num_residues == 0:
            return None

        # --- pLDDT from B-factor ---
        ca_plddt = atoms.b_factor[ca_mask]
        mean_plddt = float(np.mean(ca_plddt))
        min_plddt = float(np.min(ca_plddt))
        fraction_low_plddt = float(np.mean(ca_plddt < PLDDT_LOW))
        fraction_medium_plddt = float(np.mean(ca_plddt < PLDDT_MEDIUM))

        # --- Secondary structure ---
        sse = annotate_sse(atoms)
        n_sse = len(sse)
        if n_sse > 0:
            ss_helix = float(np.sum(sse == "a") / n_sse)
            ss_sheet = float(np.sum(sse == "b") / n_sse)
            ss_loop = float(np.sum(sse == "c") / n_sse)
        else:
            ss_helix = ss_sheet = ss_loop = 0.0

        is_non_loopy = ss_loop < LOOP_FRACTION_THRESHOLD

        # --- SASA / burial ---
        sasa_vals = sasa_fn(atoms, point_number=sasa_points)

        # Per-residue SASA (sum atom SASA within each residue)
        res_ids = atoms.res_id
        unique_res = np.unique(res_ids)
        per_res_sasa = np.array([
            np.nansum(sasa_vals[res_ids == rid]) for rid in unique_res
        ])

        mean_rasa = float(np.mean(per_res_sasa))
        buried_fraction = float(np.mean(per_res_sasa < BURIED_THRESHOLD))
        exposed_fraction = float(np.mean(per_res_sasa > EXPOSED_THRESHOLD))

        return {
            "accession": accession,
            "struct_num_residues": num_residues,
            "struct_num_chains": num_chains,
            "mean_plddt": round(mean_plddt, 2),
            "min_plddt": round(min_plddt, 2),
            "fraction_low_plddt": round(fraction_low_plddt, 4),
            "fraction_medium_plddt": round(fraction_medium_plddt, 4),
            "ss_fraction_helix": round(ss_helix, 4),
            "ss_fraction_sheet": round(ss_sheet, 4),
            "ss_fraction_loop": round(ss_loop, 4),
            "is_non_loopy": is_non_loopy,
            "mean_residue_sasa": round(mean_rasa, 2),
            "buried_fraction": round(buried_fraction, 4),
            "exposed_fraction": round(exposed_fraction, 4),
        }

    except Exception:
        print(f"  ERROR processing {accession}: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return None


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading protein list from {args.enriched_parquet} ...", flush=True)
    base_df = pd.read_parquet(args.enriched_parquet, columns=["accession", "path", "source_split"])

    if args.splits:
        base_df = base_df[base_df["source_split"].isin(args.splits)].copy()
        print(f"  Filtered to splits: {args.splits}", flush=True)

    # Deduplicate by accession (same protein could appear only once, but be safe)
    work_df = base_df.drop_duplicates(subset=["accession"])[["accession", "path"]].copy()

    # Resume support
    output_path = args.output_dir / "structure_features.parquet"
    done_accessions: set[str] = set()
    if args.resume and output_path.exists():
        existing = pd.read_parquet(output_path, columns=["accession"])
        done_accessions = set(existing["accession"].tolist())
        work_df = work_df[~work_df["accession"].isin(done_accessions)].copy()
        print(f"  Resuming: {len(done_accessions):,} already computed, {len(work_df):,} remaining.", flush=True)

    if args.limit > 0:
        work_df = work_df.head(args.limit).copy()

    total = len(work_df)
    print(f"  Processing {total:,} proteins.", flush=True)

    if total == 0:
        print("  Nothing to compute.", flush=True)
        return 0

    # Build task list
    tasks = [
        (row["accession"], row["path"], args.sasa_points)
        for _, row in work_df.iterrows()
    ]

    # --- Parallel computation ---
    n_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, total)
    print(f"[2/4] Computing features with {n_workers} workers ...", flush=True)

    results: list[dict[str, Any]] = []
    errors = 0
    t0 = time.time()

    if n_workers == 1:
        for i, task in enumerate(tasks, 1):
            result = compute_one_protein(task)
            if result is not None:
                results.append(result)
            else:
                errors += 1
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  {i:,}/{total:,} ({i/total:.1%}) | {rate:.1f} prot/s | ETA {eta:.0f}s | errors={errors}", flush=True)
    else:
        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(compute_one_protein, tasks, chunksize=32), 1):
                if result is not None:
                    results.append(result)
                else:
                    errors += 1
                if i % 2000 == 0 or i == total:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate if rate > 0 else 0
                    print(f"  {i:,}/{total:,} ({i/total:.1%}) | {rate:.1f} prot/s | ETA {eta:.0f}s | errors={errors}", flush=True)

    elapsed = time.time() - t0
    print(f"  Computed {len(results):,} features in {elapsed:.1f}s ({errors} errors).", flush=True)

    # --- Save ---
    print("[3/4] Building output dataframe ...", flush=True)
    feat_df = pd.DataFrame(results)

    # Merge with any existing results if resuming
    if done_accessions and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        feat_df = pd.concat([existing_df, feat_df], ignore_index=True)
        feat_df = feat_df.drop_duplicates(subset=["accession"], keep="last")

    feat_df = feat_df.sort_values("accession").reset_index(drop=True)

    print(f"[4/4] Writing {output_path} ({len(feat_df):,} rows) ...", flush=True)
    feat_df.to_parquet(output_path, index=False)

    # Summary statistics
    summary = {
        "total_computed": len(feat_df),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "plddt": {
            "mean": round(float(feat_df["mean_plddt"].mean()), 2),
            "median": round(float(feat_df["mean_plddt"].median()), 2),
            "fraction_high_confidence": round(float((feat_df["mean_plddt"] >= 70).mean()), 4),
        },
        "secondary_structure": {
            "mean_helix": round(float(feat_df["ss_fraction_helix"].mean()), 4),
            "mean_sheet": round(float(feat_df["ss_fraction_sheet"].mean()), 4),
            "mean_loop": round(float(feat_df["ss_fraction_loop"].mean()), 4),
            "fraction_non_loopy": round(float(feat_df["is_non_loopy"].mean()), 4),
        },
        "burial": {
            "mean_buried_fraction": round(float(feat_df["buried_fraction"].mean()), 4),
            "mean_exposed_fraction": round(float(feat_df["exposed_fraction"].mean()), 4),
        },
    }

    summary_path = args.output_dir / "structure_features_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
