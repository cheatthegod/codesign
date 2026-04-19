#!/usr/bin/env python3
"""Generate RFD3-ready CSVs with per-sample conditioning columns based on enriched metadata.

Reads the enriched parquet and adds conditioning override columns:
  cond_rasa        "always" for deeply_buried enzymes, "random" otherwise
  cond_ss          "always" for helical/sheet_rich enzymes, "random" otherwise
  cond_non_loopy   "always" for non-loopy enzymes, "random" otherwise
  cond_plddt       "always" for all (pLDDT is always useful)

These columns are consumed by OverrideConditioningFromMetadata transform
during training to make conditioning metadata-driven instead of purely random.

Outputs:
  enriched/conditioned/train.csv       Full enzyme set with conditioning columns
  enriched/conditioned/metadata.json   Statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_full.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "conditioned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched-parquet", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--enzyme-only", action="store_true", default=True)
    return parser.parse_args()


def assign_conditioning(row: pd.Series) -> dict:
    """Determine per-sample conditioning overrides based on metadata."""
    cond = {
        "cond_rasa": "random",
        "cond_ss": "random",
        "cond_non_loopy": "random",
        "cond_plddt": "always",  # pLDDT is always useful for AF structures
    }

    buried = row.get("buried_fraction", 0.0) or 0.0
    helix = row.get("ss_fraction_helix", 0.0) or 0.0
    sheet = row.get("ss_fraction_sheet", 0.0) or 0.0
    loop = row.get("ss_fraction_loop", 0.0) or 0.0
    non_loopy = row.get("is_non_loopy", False)

    # Pocket-driven RASA conditioning
    # deeply_buried → always compute RASA (the model should learn burial patterns)
    # surface_exposed → always compute RASA (learn the contrast)
    if buried >= 0.35 or buried < 0.15:
        cond["cond_rasa"] = "always"

    # Fold-driven SS conditioning
    # strong fold bias → always provide SS features (model learns fold preferences)
    if helix > 0.50 or sheet > 0.30:
        cond["cond_ss"] = "always"

    # Non-loopy conditioning (deterministic from structure)
    if non_loopy:
        cond["cond_non_loopy"] = "always"

    return cond


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.enriched_parquet} ...", flush=True)
    df = pd.read_parquet(args.enriched_parquet)

    if args.enzyme_only:
        df = df[df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()
        print(f"  Filtered to enzymes: {len(df):,}", flush=True)

    # Assign conditioning columns
    print("Assigning per-sample conditioning ...", flush=True)
    cond_records = df.apply(assign_conditioning, axis=1, result_type="expand")
    df = pd.concat([df, cond_records], axis=1)

    # Stats
    stats = {}
    for col in ["cond_rasa", "cond_ss", "cond_non_loopy", "cond_plddt"]:
        counts = df[col].value_counts().to_dict()
        stats[col] = counts
        always_pct = counts.get("always", 0) / len(df) * 100
        print(f"  {col}: always={counts.get('always', 0):,} ({always_pct:.1f}%), random={counts.get('random', 0):,}", flush=True)

    # Write per-split CSVs with conditioning columns
    # RFD3 needs: example_id, path + conditioning columns
    output_cols = ["example_id", "path", "cond_rasa", "cond_ss", "cond_non_loopy", "cond_plddt"]

    for split in ["train", "test", "validation"]:
        split_df = df[df["source_split"] == split][output_cols].copy()
        out_path = args.output_dir / f"{split}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  {split}: {len(split_df):,} rows -> {out_path}", flush=True)

    # Save metadata
    meta = {
        "total_rows": len(df),
        "conditioning_stats": stats,
        "columns": output_cols,
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
