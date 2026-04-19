#!/usr/bin/env python3
"""Interactive query tool for the enriched Prot2Text enzyme database.

Supports filtering by EC, reaction type, metals, cofactors, pocket profile,
fold bias, pLDDT, and free-text search in function descriptions.

Usage:
  # Find all zinc-dependent hydrolases with deeply buried pockets
  python query_enriched_db.py --reaction hydrolase --metals zinc --pocket deeply_buried

  # Find high-pLDDT oxidoreductases with heme
  python query_enriched_db.py --ec 1.14 --cofactors heme --min-plddt 90

  # Free-text search in function descriptions
  python query_enriched_db.py --search "cholesterol hydroxylation"

  # Show statistics for a subset
  python query_enriched_db.py --ec 3.4 --stats

  # Export results
  python query_enriched_db.py --reaction oxidoreductase --metals iron --export /tmp/iron_oxidoreductases.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_full.parquet"
DEFAULT_CONSTRAINTS = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_route_a_constraints.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched-parquet", type=Path, default=DEFAULT_ENRICHED)

    # Filters
    parser.add_argument("--ec", help="EC number prefix (e.g. 1.14 or 3.4.21).")
    parser.add_argument("--reaction", help="Reaction type (oxidoreductase, hydrolase, ...).")
    parser.add_argument("--metals", nargs="+", help="Require these metals.")
    parser.add_argument("--cofactors", nargs="+", help="Require these cofactors.")
    parser.add_argument("--pocket", choices=["deeply_buried", "semi_buried", "surface_exposed"])
    parser.add_argument("--fold", choices=["helical", "sheet_rich", "mixed", "loop_dominated"])
    parser.add_argument("--min-plddt", type=float, default=0)
    parser.add_argument("--max-length", type=int, default=99999)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--enzyme-only", action="store_true", default=True)
    parser.add_argument("--include-non-enzyme", action="store_true")
    parser.add_argument("--search", help="Free-text search in function descriptions.")
    parser.add_argument("--accession", help="Look up a specific accession.")

    # Output options
    parser.add_argument("--limit", type=int, default=20, help="Max rows to display.")
    parser.add_argument("--stats", action="store_true", help="Show distribution statistics.")
    parser.add_argument("--export", type=Path, help="Export filtered results to CSV.")
    parser.add_argument("--columns", nargs="+", default=None, help="Columns to display.")

    return parser.parse_args()


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply all filters to the dataframe."""
    mask = pd.Series(True, index=df.index)

    if args.enzyme_only and not args.include_non_enzyme:
        mask &= df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])

    if args.accession:
        mask &= df["accession"] == args.accession

    if args.ec:
        mask &= df["ec_number"].str.startswith(args.ec, na=False)

    if args.reaction:
        ec_map = {"oxidoreductase": "1", "transferase": "2", "hydrolase": "3",
                  "lyase": "4", "isomerase": "5", "ligase": "6", "translocase": "7"}
        ec_prefix = ec_map.get(args.reaction)
        if ec_prefix:
            mask &= df["ec_number"].str.startswith(ec_prefix, na=False)

    if args.metals:
        # Check GO terms and protein_names for metal mentions
        for metal in args.metals:
            go_metal_map = {"zinc": "GO:0008270", "iron": "GO:0005506",
                           "magnesium": "GO:0000287", "manganese": "GO:0030145",
                           "copper": "GO:0005507", "calcium": "GO:0005509"}
            go_id = go_metal_map.get(metal, "")
            text_match = df["function"].str.contains(metal, case=False, na=False)
            go_match = df["go_ids"].str.contains(go_id, na=False) if go_id else pd.Series(False, index=df.index)
            mask &= (text_match | go_match)

    if args.cofactors:
        for cof in args.cofactors:
            go_cof_map = {"PLP": "GO:0030170", "FAD": "GO:0050660", "FMN": "GO:0010181",
                         "heme": "GO:0020037", "NAD": "GO:0070403", "NADP": "GO:0070401"}
            go_id = go_cof_map.get(cof, "")
            text_match = df["function"].str.contains(cof, case=False, na=False)
            go_match = df["go_ids"].str.contains(go_id, na=False) if go_id else pd.Series(False, index=df.index)
            mask &= (text_match | go_match)

    if args.pocket:
        thresholds = {"deeply_buried": (0.35, 1.0), "semi_buried": (0.20, 0.35), "surface_exposed": (0.0, 0.20)}
        lo, hi = thresholds[args.pocket]
        mask &= (df["buried_fraction"] >= lo) & (df["buried_fraction"] < hi)

    if args.fold:
        fold_map = {"helical": ("ss_fraction_helix", 0.50),
                    "sheet_rich": ("ss_fraction_sheet", 0.30),
                    "loop_dominated": ("ss_fraction_loop", 0.60)}
        if args.fold == "mixed":
            mask &= (df["ss_fraction_helix"] <= 0.50) & (df["ss_fraction_sheet"] <= 0.30) & (df["ss_fraction_loop"] <= 0.60)
        else:
            col, thresh = fold_map[args.fold]
            mask &= df[col] > thresh

    if args.min_plddt > 0:
        mask &= df["mean_plddt"] >= args.min_plddt

    if args.min_length > 0:
        mask &= df["sequence_length"].astype(int) >= args.min_length

    if args.max_length < 99999:
        mask &= df["sequence_length"].astype(int) <= args.max_length

    if args.search:
        pattern = re.escape(args.search)
        mask &= df["function"].str.contains(pattern, case=False, na=False)

    return df[mask]


def show_stats(df: pd.DataFrame) -> None:
    """Show distribution statistics for the filtered subset."""
    print(f"\n--- Statistics for {len(df):,} proteins ---", flush=True)

    if "ec_number" in df.columns:
        ec_map = {"1": "oxidored.", "2": "transfer.", "3": "hydrol.", "4": "lyase",
                  "5": "isomer.", "6": "ligase", "7": "transloc."}
        ec1 = df["ec_number"].str[:1]
        print("\nEC class distribution:")
        for ec, cnt in ec1.value_counts().sort_index().items():
            print(f"  EC {ec} ({ec_map.get(ec, '?'):>10s}): {cnt:,}")

    for col, label in [("mean_plddt", "Mean pLDDT"), ("buried_fraction", "Buried fraction"),
                        ("ss_fraction_helix", "Helix fraction"), ("sequence_length", "Sequence length")]:
        if col in df.columns:
            vals = df[col].astype(float)
            print(f"\n{label}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
                  f"std={vals.std():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

    if "is_non_loopy" in df.columns:
        print(f"\nNon-loopy: {df['is_non_loopy'].sum():,}/{len(df):,} ({df['is_non_loopy'].mean():.1%})")

    if "has_pdb_mapping" in df.columns:
        print(f"PDB mapped: {df['has_pdb_mapping'].sum():,}/{len(df):,} ({df['has_pdb_mapping'].mean():.1%})")


def run() -> int:
    args = parse_args()

    print(f"Loading {args.enriched_parquet} ...", flush=True)
    df = pd.read_parquet(args.enriched_parquet)
    print(f"  {len(df):,} total rows.", flush=True)

    filtered = apply_filters(df, args)
    print(f"  {len(filtered):,} rows after filtering.", flush=True)

    if args.stats:
        show_stats(filtered)
        return 0

    if args.export:
        filtered.to_csv(args.export, index=False)
        print(f"\nExported {len(filtered):,} rows to {args.export}", flush=True)
        return 0

    # Display results
    display_cols = args.columns or [
        "accession", "ec_number", "enzyme_class", "protein_names",
        "mean_plddt", "ss_fraction_helix", "buried_fraction", "sequence_length",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    if len(filtered) == 0:
        print("\nNo matching proteins found.", flush=True)
        return 0

    show_df = filtered[display_cols].head(args.limit)

    # Truncate protein_names for display
    if "protein_names" in show_df.columns:
        show_df = show_df.copy()
        show_df["protein_names"] = show_df["protein_names"].str[:60]

    print(f"\nTop {min(args.limit, len(filtered))} results:")
    print(show_df.to_string(index=False))

    if len(filtered) > args.limit:
        print(f"\n  ... and {len(filtered) - args.limit:,} more. Use --limit or --export to see all.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
