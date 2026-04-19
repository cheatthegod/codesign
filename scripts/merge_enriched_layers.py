#!/usr/bin/env python3
"""Merge Layer 1 (annotations) and Layer 2 (structure features) into final enriched parquet files.

Reads:
- prot2text_enriched_base.parquet    (Layer 1: identity + enzyme + PDB)
- structure_features.parquet         (Layer 2: pLDDT + SS + SASA)

Outputs:
- prot2text_enriched_full.parquet    All splits, all features merged
- prot2text_enriched_enzyme.parquet  Enzyme subset (gold + silver), all features
- Per-split CSVs for inspection
- merge_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Layer 1 and Layer 2 enrichment outputs."
    )
    parser.add_argument(
        "--enriched-dir",
        type=Path,
        default=DEFAULT_ENRICHED_DIR,
        help="Directory containing Layer 1 and Layer 2 parquet files.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    enriched_dir = args.enriched_dir

    print("[1/4] Loading Layer 1 (annotations) ...", flush=True)
    base_path = enriched_dir / "prot2text_enriched_base.parquet"
    base_df = pd.read_parquet(base_path)
    print(f"  Loaded {len(base_df):,} rows, {len(base_df.columns)} columns.", flush=True)

    print("[2/4] Loading Layer 2 (structure features) ...", flush=True)
    feat_path = enriched_dir / "structure_features.parquet"
    feat_df = pd.read_parquet(feat_path)
    print(f"  Loaded {len(feat_df):,} rows, {len(feat_df.columns)} columns.", flush=True)

    print("[3/4] Merging ...", flush=True)
    merged = base_df.merge(feat_df, on="accession", how="left")

    # Integrity check
    assert len(merged) == len(base_df), (
        f"Row count mismatch: {len(merged)} vs {len(base_df)}"
    )

    feat_matched = merged["mean_plddt"].notna().sum()
    print(f"  Merged: {len(merged):,} rows, features matched: {feat_matched:,}/{len(merged):,}", flush=True)

    # Sort for deterministic output
    merged = merged.sort_values(["source_split", "accession"], kind="stable").reset_index(drop=True)

    print("[4/4] Writing outputs ...", flush=True)

    # Full merged parquet
    full_path = enriched_dir / "prot2text_enriched_full.parquet"
    merged.to_parquet(full_path, index=False)
    print(f"  Wrote {full_path} ({len(merged):,} rows, {len(merged.columns)} cols)", flush=True)

    # Enzyme subset
    enzyme_df = merged[merged["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()
    enzyme_path = enriched_dir / "prot2text_enriched_enzyme.parquet"
    enzyme_df.to_parquet(enzyme_path, index=False)
    print(f"  Wrote {enzyme_path} ({len(enzyme_df):,} rows)", flush=True)

    # Per-split CSVs (exclude sequence for size)
    csv_exclude = {"sequence"}
    csv_cols = [c for c in merged.columns if c not in csv_exclude]
    for split in merged["source_split"].unique():
        split_df = merged[merged["source_split"] == split]
        split_path = enriched_dir / f"{split}_enriched_full.csv"
        split_df[csv_cols].to_csv(split_path, index=False)

    # Summary
    summary = {
        "total_rows": int(len(merged)),
        "columns": list(merged.columns),
        "n_columns": len(merged.columns),
        "feature_coverage": {
            "layer1_enzyme_class": int(merged["enzyme_class"].ne("").sum()),
            "layer2_structure_features": int(feat_matched),
        },
        "splits": {
            split: int(count)
            for split, count in merged["source_split"].value_counts().items()
        },
        "enzyme_subset_size": int(len(enzyme_df)),
        "enzyme_class_distribution": merged["enzyme_class"].value_counts().to_dict(),
        "structure_feature_stats": {
            "mean_plddt": round(float(merged["mean_plddt"].mean()), 2),
            "fraction_non_loopy": round(float(merged["is_non_loopy"].mean()), 4),
            "mean_helix_fraction": round(float(merged["ss_fraction_helix"].mean()), 4),
            "mean_buried_fraction": round(float(merged["buried_fraction"].mean()), 4),
        },
    }

    summary_path = enriched_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
