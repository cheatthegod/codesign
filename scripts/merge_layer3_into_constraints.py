#!/usr/bin/env python3
"""Merge Layer 3 LLM enrichment results into Route A constraints.

Reads:
- prot2text_route_a_constraints.jsonl    (Layer 1+2 rule-based)
- layer3_llm_enrichment.jsonl            (Layer 3 LLM-extracted)

Outputs:
- prot2text_route_a_constraints_full.jsonl  (merged, LLM fields override where available)
- merge_layer3_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enriched-dir",
        type=Path,
        default=DEFAULT_ENRICHED_DIR,
    )
    return parser.parse_args()


def merge_record(base: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    """Merge LLM enrichment into a base constraint record."""
    merged = base.copy()

    # LLM-refined fields override rule-based where they provide real info
    if llm.get("llm_substrate_family") and llm["llm_substrate_family"] != "unknown":
        merged["substrate_family_llm"] = llm["llm_substrate_family"]

    # New fields only from LLM
    merged["required_roles"] = llm.get("llm_required_roles") or None
    merged["active_site_style"] = llm.get("llm_active_site_style")
    merged["reaction_mechanism"] = llm.get("llm_reaction_mechanism")
    merged["evidence_spans"] = llm.get("llm_evidence_spans") or []
    merged["llm_confidence"] = llm.get("llm_confidence", "low")

    # Update overall confidence: take the higher of rule-based and LLM
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    rule_conf = confidence_order.get(merged.get("confidence", "low"), 1)
    llm_conf = confidence_order.get(llm.get("llm_confidence", "low"), 1)
    if llm_conf >= rule_conf:
        merged["confidence"] = llm.get("llm_confidence", merged.get("confidence"))

    return merged


def run() -> int:
    args = parse_args()
    enriched_dir = args.enriched_dir

    base_path = enriched_dir / "prot2text_route_a_constraints.jsonl"
    llm_path = enriched_dir / "layer3_llm_enrichment.jsonl"
    output_path = enriched_dir / "prot2text_route_a_constraints_full.jsonl"

    print(f"[1/3] Loading base constraints from {base_path} ...", flush=True)
    base_records: dict[str, dict[str, Any]] = {}
    with base_path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            base_records[rec["accession"]] = rec
    print(f"  Loaded {len(base_records):,} base records.", flush=True)

    print(f"[2/3] Loading LLM enrichment from {llm_path} ...", flush=True)
    llm_records: dict[str, dict[str, Any]] = {}
    if llm_path.exists():
        with llm_path.open("r") as f:
            for line in f:
                rec = json.loads(line)
                llm_records[rec["accession"]] = rec
    print(f"  Loaded {len(llm_records):,} LLM records.", flush=True)

    print("[3/3] Merging and writing ...", flush=True)
    merged_count = 0
    base_only_count = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for acc, base in sorted(base_records.items()):
            if acc in llm_records:
                merged = merge_record(base, llm_records[acc])
                merged_count += 1
            else:
                merged = base.copy()
                merged["llm_confidence"] = None
                base_only_count += 1
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    summary = {
        "total_output": merged_count + base_only_count,
        "with_llm_enrichment": merged_count,
        "base_only": base_only_count,
    }
    summary_path = enriched_dir / "merge_layer3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
