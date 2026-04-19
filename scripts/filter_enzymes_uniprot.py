#!/usr/bin/env python3
"""Filter enzyme proteins from Prot2Text-Data via UniProt annotations.

Classification (deterministic):
- enzyme_gold: has valid EC number.
- enzyme_silver: no EC but has Rhea catalytic reaction id.
- possible_enzyme_bronze: no EC/Rhea but contains GO:0003824 (catalytic activity).
- non_enzyme: found in UniProt but none of the above.
- unresolved: accession not returned by UniProt query.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
EC_TOKEN = re.compile(r"^\d+\.\d+\.\d+\.(\d+|-)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter enzyme data from Prot2Text-Data using UniProt EC/Rhea.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Prot2Text-Data/data"),
        help="Directory containing train/validation/test parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Prot2Text-Data/enzyme_filter"),
        help="Output directory for intermediate cache and final csv files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of accessions per UniProt request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="Sleep between successful requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries per failed request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit on number of accessions. 0 means all.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cache and request all accessions again.",
    )
    return parser.parse_args()


def batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def load_dataset(data_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        if "accession" not in df.columns:
            raise ValueError(f"Missing 'accession' in {f}")
        split = f.name.split("-")[0]
        df = df.copy()
        df["split"] = split
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["accession"] = merged["accession"].astype(str)
    return merged


def read_existing_cache(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame(columns=["accession", "protein_names", "ec_number", "rhea_id", "go_ids"])
    df = pd.read_csv(cache_file, dtype=str).fillna("")
    expected = {"accession", "protein_names", "ec_number", "rhea_id", "go_ids"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Cache file missing columns: {sorted(missing)}")
    return df


def normalize_ec(ec_raw: str) -> str:
    ec_raw = (ec_raw or "").strip()
    if not ec_raw:
        return ""

    tokens = [tok.strip() for tok in re.split(r"[;,\s]+", ec_raw) if tok.strip()]
    valid = [tok for tok in tokens if EC_TOKEN.match(tok)]
    if valid:
        return ";".join(sorted(set(valid)))

    return ""


def classify_row(ec: str, rhea: str, go_ids: str, unresolved: bool) -> str:
    if unresolved:
        return "unresolved"
    if ec:
        return "enzyme_gold"
    if rhea.strip():
        return "enzyme_silver"
    if "GO:0003824" in (go_ids or ""):
        return "possible_enzyme_bronze"
    return "non_enzyme"


def fetch_uniprot_rows(batch: List[str], timeout: int, max_retries: int) -> List[Dict[str, str]]:
    query = " OR ".join(f"accession:{acc}" for acc in batch)
    params = {
        "query": query,
        "fields": "accession,protein_name,ec,rhea,go_id",
        "format": "tsv",
        "size": str(len(batch)),
    }
    url = UNIPROT_SEARCH_URL + "?" + urllib.parse.urlencode(params)

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            rows: List[Dict[str, str]] = []
            for r in reader:
                rows.append(
                    {
                        "accession": (r.get("Entry") or "").strip(),
                        "protein_names": (r.get("Protein names") or "").strip(),
                        "ec_number": normalize_ec(r.get("EC number") or ""),
                        "rhea_id": (r.get("Rhea ID") or "").strip(),
                        "go_ids": (r.get("Gene Ontology IDs") or "").strip(),
                    }
                )
            return rows
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise RuntimeError(f"UniProt request failed after {max_retries} attempts for batch size={len(batch)}") from exc
            time.sleep(delay)
            delay *= 2

    return []


def append_rows(cache_file: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    header_needed = not cache_file.exists()
    with cache_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "protein_names", "ec_number", "rhea_id", "go_ids"])
        if header_needed:
            writer.writeheader()
        writer.writerows(rows)


def run() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading dataset from {args.data_dir} ...", flush=True)
    df = load_dataset(args.data_dir)
    all_accessions = df["accession"].dropna().astype(str).unique().tolist()
    if args.limit and args.limit > 0:
        all_accessions = all_accessions[: args.limit]
        df = df[df["accession"].isin(set(all_accessions))].copy()
    print(f"Total rows={len(df):,}, unique accessions={len(all_accessions):,}", flush=True)

    cache_file = args.out_dir / "uniprot_annotations.csv"
    existing = pd.DataFrame(columns=["accession", "protein_names", "ec_number", "rhea_id", "go_ids"])
    fetched_set = set()
    if not args.refresh_cache:
        existing = read_existing_cache(cache_file)
        fetched_set = set(existing["accession"].astype(str).tolist())

    todo = [acc for acc in all_accessions if acc not in fetched_set]
    print(f"[2/4] Existing cached accessions={len(fetched_set):,}, to fetch={len(todo):,}", flush=True)

    if todo:
        total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
        for idx, batch in enumerate(batched(todo, args.batch_size), start=1):
            rows = fetch_uniprot_rows(batch, timeout=args.timeout, max_retries=args.max_retries)
            # preserve unresolved accessions by writing empty placeholders
            found = {r["accession"] for r in rows if r["accession"]}
            missing = [acc for acc in batch if acc not in found]
            if missing:
                rows.extend(
                    {
                        "accession": acc,
                        "protein_names": "",
                        "ec_number": "",
                        "rhea_id": "",
                        "go_ids": "",
                    }
                    for acc in missing
                )
            append_rows(cache_file, rows)

            if idx % 10 == 0 or idx == total_batches:
                print(f"Fetched batch {idx}/{total_batches} ({idx / total_batches:.1%})", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"[3/4] Building classifications from {cache_file} ...", flush=True)
    ann = read_existing_cache(cache_file)
    ann = ann.drop_duplicates(subset=["accession"], keep="last")

    merged = df.merge(ann, on="accession", how="left")
    for col in ["protein_names", "ec_number", "rhea_id", "go_ids"]:
        merged[col] = merged[col].fillna("")

    unresolved = merged["protein_names"].eq("") & merged["ec_number"].eq("") & merged["rhea_id"].eq("") & merged["go_ids"].eq("")
    merged["enzyme_class"] = [
        classify_row(ec, rhea, go, unres)
        for ec, rhea, go, unres in zip(merged["ec_number"], merged["rhea_id"], merged["go_ids"], unresolved)
    ]

    print("[4/4] Writing outputs ...", flush=True)
    merged_out = args.out_dir / "prot2text_with_enzyme_class.csv"
    merged.to_csv(merged_out, index=False)

    for cls in ["enzyme_gold", "enzyme_silver", "possible_enzyme_bronze", "non_enzyme", "unresolved"]:
        part = merged[merged["enzyme_class"] == cls]
        part.to_csv(args.out_dir / f"{cls}.csv", index=False)

    summary = merged["enzyme_class"].value_counts(dropna=False).rename_axis("enzyme_class").reset_index(name="count")
    summary["ratio"] = (summary["count"] / len(merged)).round(6)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    print("Summary:", flush=True)
    for _, row in summary.iterrows():
        print(f"  {row['enzyme_class']}: {int(row['count']):,} ({row['ratio']:.2%})", flush=True)
    print(f"Done. Outputs in: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
