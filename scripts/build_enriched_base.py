#!/usr/bin/env python3
"""Layer 1: Join RFD3-filtered Prot2Text data with UniProt/enzyme annotations and PDB mappings.

Produces enriched parquet files that augment the minimal RFD3 training CSVs with:
- enzyme classification (gold/silver/bronze/non_enzyme)
- UniProt annotations (protein_names, ec_number, rhea_id, go_ids)
- functional text (sequence, function description)
- PDB mapping status (pdb_ids, has_pdb_mapping)

Outputs:
- prot2text_enriched_base.parquet      All splits combined
- prot2text_enriched_enzyme.parquet    Enzyme subset (gold + silver)
- enrichment_summary.json             Join statistics and coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_RFD3_DIR = PROJECT_DIR / "Prot2Text-Data" / "rfd3_monomer_64_1024"
DEFAULT_ENZYME_DIR = PROJECT_DIR / "Prot2Text-Data" / "enzyme_filter_uniprot"
DEFAULT_DISCO_DIR = PROJECT_DIR / "DISCO" / "local_data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join RFD3 training data with enzyme/UniProt annotations and PDB mappings."
    )
    parser.add_argument(
        "--rfd3-dir",
        type=Path,
        default=DEFAULT_RFD3_DIR,
        help="Directory containing RFD3-filtered train/test/validation CSVs.",
    )
    parser.add_argument(
        "--enzyme-dir",
        type=Path,
        default=DEFAULT_ENZYME_DIR,
        help="Directory containing prot2text_with_enzyme_class.csv and uniprot_annotations.csv.",
    )
    parser.add_argument(
        "--disco-dir",
        type=Path,
        default=DEFAULT_DISCO_DIR,
        help="Directory containing enzyme_pdb_mapping.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for enriched parquet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Which data splits to process.",
    )
    return parser.parse_args()


def load_rfd3_splits(rfd3_dir: Path, splits: list[str]) -> pd.DataFrame:
    """Load and concatenate RFD3-filtered CSVs with a source_split column."""
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = rfd3_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing RFD3 split file: {path}")
        df = pd.read_csv(path, dtype=str)
        df["source_split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["accession"] = combined["accession"].astype(str).str.strip()
    combined["sequence_length"] = pd.to_numeric(combined["sequence_length"], errors="coerce").astype("Int64")
    return combined


def load_enzyme_annotations(enzyme_dir: Path) -> pd.DataFrame:
    """Load the merged prot2text_with_enzyme_class.csv for annotation fields."""
    path = enzyme_dir / "prot2text_with_enzyme_class.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing enzyme annotation file: {path}")

    wanted_cols = [
        "accession",
        "sequence",
        "function",
        "protein_names",
        "ec_number",
        "rhea_id",
        "go_ids",
        "enzyme_class",
    ]
    df = pd.read_csv(path, dtype=str, usecols=wanted_cols)
    df["accession"] = df["accession"].astype(str).str.strip()
    # deduplicate on accession (shouldn't happen, but defensive)
    df = df.drop_duplicates(subset=["accession"], keep="first")

    for col in ["sequence", "function", "protein_names", "ec_number", "rhea_id", "go_ids", "enzyme_class"]:
        df[col] = df[col].fillna("")

    return df


def load_pdb_mapping(disco_dir: Path) -> pd.DataFrame:
    """Load enzyme_pdb_mapping.tsv for PDB structure linkage."""
    path = disco_dir / "enzyme_pdb_mapping.tsv"
    if not path.exists():
        print(f"  Warning: PDB mapping file not found at {path}, skipping PDB join.")
        return pd.DataFrame(columns=["accession", "pdb_ids", "n_pdbs"])

    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"uniprot": "accession"})
    df["accession"] = df["accession"].astype(str).str.strip()
    df["n_pdbs"] = pd.to_numeric(df["n_pdbs"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["accession"], keep="first")

    # keep only accession, pdb_ids, n_pdbs (drop ec_number from this source to avoid conflicts)
    return df[["accession", "pdb_ids", "n_pdbs"]]


def build_enriched(
    rfd3_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    pdb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join all annotation sources onto the RFD3 base."""
    # Join enzyme annotations
    merged = rfd3_df.merge(enzyme_df, on="accession", how="left", suffixes=("", "_enzyme"))

    # Join PDB mapping
    merged = merged.merge(pdb_df, on="accession", how="left", suffixes=("", "_pdb"))

    # Derive boolean flags
    merged["has_pdb_mapping"] = merged["pdb_ids"].notna() & merged["pdb_ids"].ne("")
    merged["is_enzyme"] = merged["enzyme_class"].isin(["enzyme_gold", "enzyme_silver", "possible_enzyme_bronze"])

    # Fill NaN for string columns
    str_cols = ["sequence", "function", "protein_names", "ec_number", "rhea_id", "go_ids", "enzyme_class", "pdb_ids"]
    for col in str_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    # Fill NaN for numeric columns
    if "n_pdbs" in merged.columns:
        merged["n_pdbs"] = merged["n_pdbs"].fillna(0).astype(int)

    return merged


def build_summary(enriched: pd.DataFrame) -> dict:
    """Compute join coverage statistics."""
    total = len(enriched)

    enzyme_coverage = enriched["enzyme_class"].ne("").sum()
    pdb_coverage = enriched["has_pdb_mapping"].sum()
    enzyme_gold = (enriched["enzyme_class"] == "enzyme_gold").sum()
    enzyme_silver = (enriched["enzyme_class"] == "enzyme_silver").sum()
    enzyme_bronze = (enriched["enzyme_class"] == "possible_enzyme_bronze").sum()
    non_enzyme = (enriched["enzyme_class"] == "non_enzyme").sum()
    has_function = enriched["function"].ne("").sum()
    has_ec = enriched["ec_number"].ne("").sum()
    has_rhea = enriched["rhea_id"].ne("").sum()
    has_go = enriched["go_ids"].ne("").sum()

    return {
        "total_rows": int(total),
        "splits": {
            split: int(count)
            for split, count in enriched["source_split"].value_counts().items()
        },
        "enzyme_annotation_coverage": {
            "annotated": int(enzyme_coverage),
            "ratio": round(enzyme_coverage / total, 4) if total else 0,
        },
        "enzyme_class_distribution": {
            "enzyme_gold": int(enzyme_gold),
            "enzyme_silver": int(enzyme_silver),
            "possible_enzyme_bronze": int(enzyme_bronze),
            "non_enzyme": int(non_enzyme),
            "unresolved_or_missing": int(total - enzyme_gold - enzyme_silver - enzyme_bronze - non_enzyme),
        },
        "field_coverage": {
            "has_function_text": int(has_function),
            "has_ec_number": int(has_ec),
            "has_rhea_id": int(has_rhea),
            "has_go_ids": int(has_go),
            "has_pdb_mapping": int(pdb_coverage),
        },
        "enzyme_subset_size": int(enzyme_gold + enzyme_silver),
    }


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading RFD3 splits from {args.rfd3_dir} ...", flush=True)
    rfd3_df = load_rfd3_splits(args.rfd3_dir, args.splits)
    print(f"  Loaded {len(rfd3_df):,} rows across {rfd3_df['source_split'].nunique()} splits.", flush=True)

    print(f"[2/5] Loading enzyme annotations from {args.enzyme_dir} ...", flush=True)
    enzyme_df = load_enzyme_annotations(args.enzyme_dir)
    print(f"  Loaded {len(enzyme_df):,} unique accessions with enzyme class.", flush=True)

    print(f"[3/5] Loading PDB mappings from {args.disco_dir} ...", flush=True)
    pdb_df = load_pdb_mapping(args.disco_dir)
    print(f"  Loaded {len(pdb_df):,} accessions with PDB mappings.", flush=True)

    print("[4/5] Joining all sources ...", flush=True)
    enriched = build_enriched(rfd3_df, enzyme_df, pdb_df)

    # Integrity check: row count must be preserved
    assert len(enriched) == len(rfd3_df), (
        f"Row count mismatch after join: {len(enriched)} vs {len(rfd3_df)}. "
        f"Check for duplicate accessions in annotation sources."
    )

    summary = build_summary(enriched)
    print(f"  Join complete: {summary['total_rows']:,} rows", flush=True)
    print(f"  Enzyme annotated: {summary['enzyme_annotation_coverage']['annotated']:,} "
          f"({summary['enzyme_annotation_coverage']['ratio']:.1%})", flush=True)
    print(f"  PDB mapped: {summary['field_coverage']['has_pdb_mapping']:,}", flush=True)

    print(f"[5/5] Writing outputs to {args.output_dir} ...", flush=True)

    # Define column order for output
    output_cols = [
        # identity
        "example_id", "path", "source_split", "accession", "alphafold_id", "sequence_length",
        # original RFD3 metadata
        "name", "full_name", "taxon",
        # Layer 1: enzyme annotations
        "enzyme_class", "is_enzyme", "protein_names", "ec_number", "rhea_id", "go_ids",
        # Layer 1: functional text
        "sequence", "function",
        # Layer 1: PDB mapping
        "pdb_ids", "n_pdbs", "has_pdb_mapping",
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in enriched.columns]

    enriched_out = enriched[output_cols].copy()
    enriched_out = enriched_out.sort_values(["source_split", "accession"], kind="stable").reset_index(drop=True)

    # Write base parquet
    base_path = args.output_dir / "prot2text_enriched_base.parquet"
    enriched_out.to_parquet(base_path, index=False)
    print(f"  Wrote {base_path} ({len(enriched_out):,} rows)", flush=True)

    # Write per-split CSVs (without sequence column for size)
    csv_cols = [c for c in output_cols if c != "sequence"]
    for split in args.splits:
        split_df = enriched_out[enriched_out["source_split"] == split]
        split_path = args.output_dir / f"{split}_enriched.csv"
        split_df[csv_cols].to_csv(split_path, index=False)

    # Write enzyme subset
    enzyme_subset = enriched_out[enriched_out["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()
    enzyme_path = args.output_dir / "prot2text_enriched_enzyme.parquet"
    enzyme_subset.to_parquet(enzyme_path, index=False)
    print(f"  Wrote {enzyme_path} ({len(enzyme_subset):,} rows)", flush=True)

    # Write summary
    summary_path = args.output_dir / "enrichment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
