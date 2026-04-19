#!/usr/bin/env python3
"""Build focused training subsets from enriched Prot2Text data.

Not all 133K enzymes are equally useful for every training objective.
This script generates targeted subsets with corresponding RFD3 dataset configs.

Available strategies:
  high_quality    High pLDDT (>85) + non-loopy + well-buried structures
  by_reaction     One subset per reaction type (oxidoreductase, hydrolase, etc.)
  by_pocket       Split by pocket profile (deeply_buried, semi_buried, surface_exposed)
  by_fold         Split by fold bias (helical, sheet_rich, mixed)
  balanced_ec     Balanced sampling across EC first-digit classes
  custom          Filter by arbitrary parquet column conditions

Outputs per subset:
  - {name}/train.csv          RFD3-ready CSV (example_id, path)
  - {name}/metadata.json      Subset statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_full.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "training_subsets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched-parquet", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["high_quality", "by_reaction", "by_pocket", "balanced_ec"],
        help="Which subset strategies to generate.",
    )
    parser.add_argument("--min-plddt", type=float, default=85.0, help="pLDDT threshold for high_quality.")
    parser.add_argument("--balance-n", type=int, default=5000, help="Samples per class for balanced strategies.")
    return parser.parse_args()


def write_subset(df: pd.DataFrame, name: str, output_dir: Path, description: str) -> dict:
    """Write a training subset and return its metadata."""
    subset_dir = output_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)

    # RFD3 only needs example_id + path
    train_df = df[df["source_split"] == "train"][["example_id", "path"]].copy()
    test_df = df[df["source_split"] == "test"][["example_id", "path"]].copy()
    val_df = df[df["source_split"] == "validation"][["example_id", "path"]].copy()

    train_df.to_csv(subset_dir / "train.csv", index=False)
    if len(test_df) > 0:
        test_df.to_csv(subset_dir / "test.csv", index=False)
    if len(val_df) > 0:
        val_df.to_csv(subset_dir / "validation.csv", index=False)

    metadata = {
        "name": name,
        "description": description,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "val_size": len(val_df),
        "total": len(df),
    }

    # Add distribution stats if enzyme columns exist
    if "enzyme_class" in df.columns:
        metadata["enzyme_class_dist"] = df["enzyme_class"].value_counts().to_dict()
    if "mean_plddt" in df.columns:
        metadata["plddt_stats"] = {
            "mean": round(float(df["mean_plddt"].mean()), 2),
            "median": round(float(df["mean_plddt"].median()), 2),
            "min": round(float(df["mean_plddt"].min()), 2),
        }
    if "ss_fraction_helix" in df.columns:
        metadata["ss_means"] = {
            "helix": round(float(df["ss_fraction_helix"].mean()), 3),
            "sheet": round(float(df["ss_fraction_sheet"].mean()), 3),
            "loop": round(float(df["ss_fraction_loop"].mean()), 3),
        }

    (subset_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def strategy_high_quality(df: pd.DataFrame, min_plddt: float, output_dir: Path) -> list[dict]:
    """High-confidence structures with good packing."""
    subset = df[
        (df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"]))
        & (df["mean_plddt"] >= min_plddt)
        & (df["is_non_loopy"] == True)
        & (df["buried_fraction"] >= 0.20)
    ].copy()
    meta = write_subset(subset, "high_quality", output_dir,
                        f"Enzymes with pLDDT>={min_plddt}, non-loopy, buried>=0.20")
    print(f"  high_quality: {meta['train_size']:,} train", flush=True)
    return [meta]


def strategy_by_reaction(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """One subset per reaction type."""
    enzyme_df = df[df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()

    # Need to compute reaction_type from EC
    ec_map = {"1": "oxidoreductase", "2": "transferase", "3": "hydrolase",
              "4": "lyase", "5": "isomerase", "6": "ligase", "7": "translocase"}
    enzyme_df["reaction_type"] = enzyme_df["ec_number"].str[:1].map(ec_map)

    results = []
    for rt in sorted(enzyme_df["reaction_type"].dropna().unique()):
        subset = enzyme_df[enzyme_df["reaction_type"] == rt]
        meta = write_subset(subset, f"reaction_{rt}", output_dir, f"Enzymes with reaction_type={rt}")
        print(f"  reaction_{rt}: {meta['train_size']:,} train", flush=True)
        results.append(meta)
    return results


def strategy_by_pocket(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Split by pocket burial profile."""
    enzyme_df = df[df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()

    results = []
    for bf_label, bf_min, bf_max in [
        ("deeply_buried", 0.35, 1.0),
        ("semi_buried", 0.20, 0.35),
        ("surface_exposed", 0.0, 0.20),
    ]:
        subset = enzyme_df[
            (enzyme_df["buried_fraction"] >= bf_min) & (enzyme_df["buried_fraction"] < bf_max)
        ]
        meta = write_subset(subset, f"pocket_{bf_label}", output_dir,
                            f"Enzymes with buried_fraction in [{bf_min}, {bf_max})")
        print(f"  pocket_{bf_label}: {meta['train_size']:,} train", flush=True)
        results.append(meta)
    return results


def strategy_balanced_ec(df: pd.DataFrame, n_per_class: int, output_dir: Path) -> list[dict]:
    """Balanced sampling across EC first-digit classes."""
    enzyme_df = df[df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()
    enzyme_df["ec_first"] = enzyme_df["ec_number"].str[:1]

    # Sample min(n_per_class, class_size) from each EC class
    frames = []
    for ec1 in sorted(enzyme_df["ec_first"].dropna().unique()):
        class_df = enzyme_df[enzyme_df["ec_first"] == ec1]
        n = min(n_per_class, len(class_df))
        sampled = class_df.sample(n=n, random_state=42)
        frames.append(sampled)

    balanced = pd.concat(frames, ignore_index=True)
    meta = write_subset(balanced, "balanced_ec", output_dir,
                        f"Balanced EC sampling, up to {n_per_class} per EC class")
    print(f"  balanced_ec: {meta['train_size']:,} train", flush=True)

    # Show per-class counts
    counts = balanced["ec_first"].value_counts().sort_index()
    ec_names = {"1": "oxidored.", "2": "transfer.", "3": "hydrol.", "4": "lyase",
                "5": "isomer.", "6": "ligase", "7": "transloc."}
    for ec1, cnt in counts.items():
        print(f"    EC {ec1} ({ec_names.get(ec1, '?')}): {cnt:,}", flush=True)

    return [meta]


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading enriched data from {args.enriched_parquet} ...", flush=True)
    df = pd.read_parquet(args.enriched_parquet)
    print(f"  {len(df):,} total rows.", flush=True)

    all_meta: list[dict] = []

    for strategy in args.strategies:
        print(f"\n--- Strategy: {strategy} ---", flush=True)
        if strategy == "high_quality":
            all_meta.extend(strategy_high_quality(df, args.min_plddt, args.output_dir))
        elif strategy == "by_reaction":
            all_meta.extend(strategy_by_reaction(df, args.output_dir))
        elif strategy == "by_pocket":
            all_meta.extend(strategy_by_pocket(df, args.output_dir))
        elif strategy == "balanced_ec":
            all_meta.extend(strategy_balanced_ec(df, args.balance_n, args.output_dir))
        else:
            print(f"  Unknown strategy: {strategy}", flush=True)

    # Write global summary
    summary = {"subsets": {m["name"]: {"train": m["train_size"], "total": m["total"]} for m in all_meta}}
    summary_path = args.output_dir / "subsets_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_meta)} subsets to {args.output_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
