#!/usr/bin/env python3
"""Export a ligand-free Prot2Text subset into Foundry/RFD3-friendly tables.

The export targets the simple custom-dataset format used by Foundry/RFD3:
at minimum, each row contains an ``example_id`` and a structure ``path``.

Default filtering is intentionally conservative for monomer training:
- require a local AlphaFold structure file
- require canonical 20-AA sequences
- keep proteins with length in [64, 1024]

Outputs:
- train.csv / validation.csv / test.csv
- all_filtered.csv
- rejected_rows.csv
- filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

CANONICAL_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
AF_STRUCTURE_RE = re.compile(
    r"^AF-(?P<af_id>.+)-F1-model_v(?P<version>\d+)\.(?P<ext>pdb|cif)(?:\.gz)?$"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "Prot2Text-Data" / "data"
DEFAULT_STRUCTURES_DIR = PROJECT_DIR / "Prot2Text-Data" / "alphafold_structures" / "pdb"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "rfd3_monomer_64_1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a filtered Prot2Text subset as Foundry/RFD3-ready CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing Prot2Text parquet split files.",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=DEFAULT_STRUCTURES_DIR,
        help="Directory containing AlphaFold structure files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where filtered CSVs and summary files will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="prot2text",
        help="Dataset tag embedded into Foundry-style example_id values.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=64,
        help="Minimum sequence length to retain.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length to retain.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Optional debug limit applied independently to each split. 0 means no limit.",
    )
    parser.add_argument(
        "--require-canonical-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require sequences to contain only the 20 standard amino acids.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write parquet copies for each exported table.",
    )
    return parser.parse_args()


def infer_split_from_filename(path: Path) -> str:
    return path.name.split("-")[0]


def build_structure_index(structures_dir: Path) -> Dict[str, Path]:
    if not structures_dir.exists():
        raise FileNotFoundError(f"Structure directory not found: {structures_dir}")

    index: Dict[str, tuple[int, Path]] = {}
    for path in sorted(structures_dir.iterdir()):
        if not path.is_file():
            continue
        match = AF_STRUCTURE_RE.match(path.name)
        if not match:
            continue

        af_id = match.group("af_id")
        version = int(match.group("version"))
        current = index.get(af_id)
        resolved_path = path.resolve()
        if current is None or version > current[0]:
            index[af_id] = (version, resolved_path)

    return {af_id: path for af_id, (_, path) in index.items()}


def load_all_splits(data_dir: Path, limit_per_split: int = 0) -> pd.DataFrame:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    wanted_cols = [
        "accession",
        "name",
        "Full Name",
        "taxon",
        "sequence",
        "function",
        "AlphaFoldDB",
    ]
    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        split = infer_split_from_filename(path)
        df = pd.read_parquet(path, columns=wanted_cols).copy()
        if limit_per_split > 0:
            df = df.head(limit_per_split).copy()
        df["source_split"] = split
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["accession"] = merged["accession"].astype(str)
    merged["AlphaFoldDB"] = merged["AlphaFoldDB"].fillna("").astype(str).str.strip()
    merged["sequence"] = merged["sequence"].fillna("").astype(str).str.strip()
    merged["name"] = merged["name"].fillna("").astype(str)
    merged["Full Name"] = merged["Full Name"].fillna("").astype(str)
    merged["taxon"] = merged["taxon"].fillna("").astype(str)
    return merged


def build_example_id(dataset_name: str, accession: str) -> str:
    return f"{{['{dataset_name}']}}{{{accession}}}{{1}}{{[]}}"


def apply_filters(
    df: pd.DataFrame,
    *,
    structure_index: Dict[str, Path],
    dataset_name: str,
    min_length: int,
    max_length: int,
    require_canonical_sequence: bool,
) -> pd.DataFrame:
    out = df.copy()

    out["sequence_length"] = out["sequence"].str.len()
    out["canonical_sequence"] = out["sequence"].str.fullmatch(CANONICAL_AA_RE).fillna(False)
    out["path"] = out["AlphaFoldDB"].map(structure_index)
    out["has_structure"] = out["path"].notna()

    reasons: list[str] = []
    for _, row in out.iterrows():
        row_reasons: list[str] = []
        if not row["has_structure"]:
            row_reasons.append("missing_structure")
        if require_canonical_sequence and not row["canonical_sequence"]:
            row_reasons.append("noncanonical_sequence")
        if row["sequence_length"] < min_length:
            row_reasons.append("too_short")
        if row["sequence_length"] > max_length:
            row_reasons.append("too_long")
        reasons.append(";".join(row_reasons) if row_reasons else "ok")

    out["filter_status"] = reasons
    out["keep"] = out["filter_status"].eq("ok")
    out["example_id"] = out["accession"].map(lambda x: build_example_id(dataset_name, x))
    out["alphafold_id"] = out["AlphaFoldDB"]
    out["full_name"] = out["Full Name"]

    return out


def write_table(df: pd.DataFrame, path: Path, write_parquet: bool) -> None:
    df.to_csv(path, index=False)
    if write_parquet:
        df.to_parquet(path.with_suffix(".parquet"), index=False)


def build_summary(
    filtered_df: pd.DataFrame,
    *,
    min_length: int,
    max_length: int,
    dataset_name: str,
    structures_dir: Path,
    require_canonical_sequence: bool,
) -> dict:
    total_rows = len(filtered_df)
    kept_df = filtered_df[filtered_df["keep"]]
    rejected_df = filtered_df[~filtered_df["keep"]]

    summary = {
        "dataset_name": dataset_name,
        "structures_dir": str(structures_dir.resolve()),
        "filters": {
            "min_length": min_length,
            "max_length": max_length,
            "require_canonical_sequence": require_canonical_sequence,
            "require_local_structure": True,
        },
        "rows_total": total_rows,
        "rows_kept": int(len(kept_df)),
        "rows_rejected": int(len(rejected_df)),
        "splits": {},
        "rejection_reasons": dict(
            sorted(
                Counter(
                    reason
                    for status in rejected_df["filter_status"]
                    for reason in status.split(";")
                    if reason
                ).items()
            )
        ),
    }

    for split, split_df in filtered_df.groupby("source_split", sort=True):
        split_kept = split_df[split_df["keep"]]
        split_rejected = split_df[~split_df["keep"]]
        summary["splits"][split] = {
            "rows_total": int(len(split_df)),
            "rows_kept": int(len(split_kept)),
            "rows_rejected": int(len(split_rejected)),
            "length_min_kept": int(split_kept["sequence_length"].min()) if len(split_kept) else None,
            "length_median_kept": int(split_kept["sequence_length"].median()) if len(split_kept) else None,
            "length_max_kept": int(split_kept["sequence_length"].max()) if len(split_kept) else None,
        }

    return summary


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Indexing local structures from {args.structures_dir} ...", flush=True)
    structure_index = build_structure_index(args.structures_dir)
    print(f"Indexed {len(structure_index):,} AlphaFold structures.", flush=True)

    print(f"[2/4] Loading Prot2Text parquet splits from {args.data_dir} ...", flush=True)
    raw_df = load_all_splits(args.data_dir, limit_per_split=args.limit_per_split)
    print(f"Loaded {len(raw_df):,} rows across {raw_df['source_split'].nunique()} splits.", flush=True)

    print("[3/4] Applying RFD3-oriented monomer filters ...", flush=True)
    filtered_df = apply_filters(
        raw_df,
        structure_index=structure_index,
        dataset_name=args.dataset_name,
        min_length=args.min_length,
        max_length=args.max_length,
        require_canonical_sequence=args.require_canonical_sequence,
    )

    keep_cols = [
        "example_id",
        "path",
        "source_split",
        "accession",
        "alphafold_id",
        "sequence_length",
        "name",
        "full_name",
        "taxon",
    ]
    kept_df = filtered_df.loc[filtered_df["keep"], keep_cols].copy()
    kept_df["path"] = kept_df["path"].map(str)
    kept_df = kept_df.sort_values(["source_split", "accession"], kind="stable").reset_index(drop=True)

    rejected_cols = [
        "source_split",
        "accession",
        "alphafold_id",
        "sequence_length",
        "canonical_sequence",
        "has_structure",
        "filter_status",
    ]
    rejected_df = filtered_df.loc[~filtered_df["keep"], rejected_cols].copy()
    rejected_df = rejected_df.sort_values(["source_split", "accession"], kind="stable").reset_index(drop=True)

    print(f"[4/4] Writing filtered tables into {args.output_dir} ...", flush=True)
    write_table(kept_df, args.output_dir / "all_filtered.csv", args.write_parquet)
    write_table(rejected_df, args.output_dir / "rejected_rows.csv", args.write_parquet)

    for split, split_df in kept_df.groupby("source_split", sort=True):
        write_table(
            split_df.drop(columns=["source_split"]),
            args.output_dir / f"{split}.csv",
            args.write_parquet,
        )

    summary = build_summary(
        filtered_df,
        min_length=args.min_length,
        max_length=args.max_length,
        dataset_name=args.dataset_name,
        structures_dir=args.structures_dir,
        require_canonical_sequence=args.require_canonical_sequence,
    )
    summary_path = args.output_dir / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
