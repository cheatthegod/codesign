#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a Prot2Chat-style dataset into train/val JSON files."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input Prot2Chat JSON containing items with pdb/primaryAccession/conversations.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.json, val.json, and split_summary.json will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio at the protein-group level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--group-by",
        choices=["primary_accession", "pdb_stem"],
        default="primary_accession",
        help="How to group items before splitting.",
    )
    return parser.parse_args()


def conversation_count(item: dict) -> int:
    conversations = item.get("conversations", [])
    return sum(1 for pair in conversations if isinstance(pair, list) and len(pair) == 2)


def protein_group_key(item: dict, mode: str) -> str:
    if mode == "primary_accession":
        key = item.get("primaryAccession")
        if key:
            return str(key)
    pdb_name = item.get("pdb", "")
    if pdb_name:
        return Path(str(pdb_name)).stem
    protein_path = item.get("protein_path") or item.get("protein")
    if protein_path:
        return Path(str(protein_path)).stem
    raise ValueError("Could not infer protein group key from dataset item.")


def build_summary(items: list[dict]) -> dict:
    return {
        "items": len(items),
        "conversation_pairs": sum(conversation_count(item) for item in items),
        "unique_pdb": len({item.get("pdb") for item in items if item.get("pdb")}),
        "unique_primary_accession": len(
            {item.get("primaryAccession") for item in items if item.get("primaryAccession")}
        ),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Input JSON must be a non-empty list.")

    grouped_items: dict[str, list[dict]] = defaultdict(list)
    for item in raw:
        grouped_items[protein_group_key(item, args.group_by)].append(item)

    groups = list(grouped_items.items())
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    total_groups = len(groups)
    val_groups = max(1, int(round(total_groups * args.val_ratio)))
    if val_groups >= total_groups:
        val_groups = max(1, total_groups - 1)

    val_group_keys = {group_key for group_key, _ in groups[:val_groups]}

    train_items: list[dict] = []
    val_items: list[dict] = []
    for group_key, items in groups:
        if group_key in val_group_keys:
            val_items.extend(items)
        else:
            train_items.extend(items)

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    summary_path = output_dir / "split_summary.json"

    train_path.write_text(json.dumps(train_items, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_items, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_json": str(input_path),
        "group_by": args.group_by,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_groups": total_groups,
        "val_groups": val_groups,
        "train_groups": total_groups - val_groups,
        "train": build_summary(train_items),
        "val": build_summary(val_items),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
