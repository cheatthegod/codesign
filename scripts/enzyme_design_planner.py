#!/usr/bin/env python3
"""End-to-end enzyme design planner: text → constraints → RFD3 plan.

Takes a natural language enzyme design requirement and produces:
1. Matching reference enzymes from the enriched database
2. Structured constraints derived from the best matches
3. An RFD3 input specification draft

Usage:
  python enzyme_design_planner.py \\
    --goal "Design a zinc-dependent metalloprotease with a deeply buried active site, ~300 residues" \\
    --output-dir /tmp/my_enzyme_plan

  python enzyme_design_planner.py \\
    --ec 1.14.13 --metals iron --cofactors heme --length 400 \\
    --output-dir /tmp/p450_plan
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CONSTRAINTS = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_route_a_constraints.jsonl"
DEFAULT_ENRICHED = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_full.parquet"
PLANNER_SCRIPT = PROJECT_DIR / "foundry" / "scripts" / "plan_text_to_rfd3_constraints.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Natural language input
    parser.add_argument("--goal", type=str, help="Free-text enzyme design goal.")

    # Structured input (can combine with --goal)
    parser.add_argument("--ec", type=str, help="EC number prefix to search (e.g. 1.14 or 3.4.21).")
    parser.add_argument("--reaction", type=str, help="Reaction type filter.")
    parser.add_argument("--metals", nargs="*", help="Required metals (e.g. zinc iron).")
    parser.add_argument("--cofactors", nargs="*", help="Required cofactors (e.g. heme NAD).")
    parser.add_argument("--pocket", choices=["deeply_buried", "semi_buried", "surface_exposed"])
    parser.add_argument("--fold", choices=["helical", "sheet_rich", "mixed", "loop_dominated"])
    parser.add_argument("--length", type=int, help="Target sequence length.")
    parser.add_argument("--min-plddt", type=float, default=80.0, help="Minimum mean pLDDT for references.")

    # Data sources
    parser.add_argument("--constraints-jsonl", type=Path, default=DEFAULT_CONSTRAINTS)
    parser.add_argument("--enriched-parquet", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--top-k", type=int, default=5, help="Number of reference enzymes to retrieve.")

    return parser.parse_args()


def parse_goal_text(goal: str) -> dict[str, Any]:
    """Extract structured hints from free-text goal."""
    lower = goal.lower()
    hints: dict[str, Any] = {}

    # Metals
    metal_patterns = {
        "zinc": r"\bzinc\b|\bzn\b",
        "iron": r"\biron\b|\bfe\b|\bdiiron\b",
        "manganese": r"\bmanganese\b|\bmn\b",
        "magnesium": r"\bmagnesium\b|\bmg\b",
        "copper": r"\bcopper\b|\bcu\b",
        "calcium": r"\bcalcium\b",
    }
    metals = [m for m, p in metal_patterns.items() if re.search(p, lower)]
    if metals:
        hints["metals"] = metals

    # Cofactors
    cofactor_patterns = {
        "NAD": r"\bnadp?\b", "FAD": r"\bfad\b", "FMN": r"\bfmn\b",
        "PLP": r"\bplp\b|pyridoxal", "heme": r"\bheme\b|\bhaem\b",
        "CoA": r"\bcoa\b|coenzyme.a",
    }
    cofactors = [c for c, p in cofactor_patterns.items() if re.search(p, lower)]
    if cofactors:
        hints["cofactors"] = cofactors

    # Pocket
    if "deeply buried" in lower or "deep pocket" in lower or "deeply-buried" in lower:
        hints["pocket"] = "deeply_buried"
    elif "semi-buried" in lower or "semi buried" in lower:
        hints["pocket"] = "semi_buried"
    elif "surface" in lower or "exposed" in lower:
        hints["pocket"] = "surface_exposed"

    # Fold
    if "helical" in lower or "alpha-helical" in lower or "all-alpha" in lower:
        hints["fold"] = "helical"
    elif "sheet" in lower or "beta-sheet" in lower or "all-beta" in lower:
        hints["fold"] = "sheet_rich"

    # Length
    length_match = re.search(r"(\d{2,4})\s*(?:residues|aa|amino acids)", lower)
    if length_match:
        hints["length"] = int(length_match.group(1))

    # Reaction type keywords
    reaction_keywords = {
        "oxidoreductase": r"\boxidoreductase\b|\boxidase\b|\breductase\b|\bdehydrogenase\b|\bmonooxygenase\b|\bp450\b",
        "hydrolase": r"\bhydrolase\b|\bprotease\b|\bpeptidase\b|\blipase\b|\besterase\b|\bnuclease\b",
        "transferase": r"\btransferase\b|\bkinase\b|\bmethyltransferase\b|\bacetyltransferase\b",
        "lyase": r"\blyase\b|\bdecarboxylase\b|\bdehydratase\b|\bsynthase\b",
        "ligase": r"\bligase\b|\bsynthetase\b",
        "isomerase": r"\bisomerase\b|\bmutase\b|\bracemase\b|\bepimerase\b",
    }
    for rt, pattern in reaction_keywords.items():
        if re.search(pattern, lower):
            hints["reaction"] = rt
            break

    # EC from text
    ec_match = re.search(r"\bec\s*(\d+\.\d+(?:\.\d+)*)", lower)
    if ec_match:
        hints["ec"] = ec_match.group(1)

    return hints


def find_reference_enzymes(
    constraints: list[dict[str, Any]],
    *,
    ec_prefix: str | None = None,
    reaction: str | None = None,
    metals: list[str] | None = None,
    cofactors: list[str] | None = None,
    pocket: str | None = None,
    fold: str | None = None,
    min_plddt: float = 80.0,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Score and rank constraints by match to design requirements."""
    scored: list[tuple[float, dict[str, Any]]] = []

    for c in constraints:
        score = 0.0
        c_plddt = c.get("plddt_summary", {}).get("mean", 0)
        if c_plddt < min_plddt:
            continue

        # EC match
        if ec_prefix and c.get("ec_number", "").startswith(ec_prefix):
            depth = len(ec_prefix.split("."))
            score += depth * 3  # deeper match = higher score

        # Reaction type match
        if reaction and c.get("reaction_type") == reaction:
            score += 2

        # Metal match
        if metals:
            c_metals = set(c.get("metal_hint") or [])
            overlap = len(set(metals) & c_metals)
            score += overlap * 2

        # Cofactor match
        if cofactors:
            c_cofactors = set(c.get("cofactor_hint") or [])
            overlap = len(set(cofactors) & c_cofactors)
            score += overlap * 2

        # Pocket match
        if pocket and c.get("pocket_profile") == pocket:
            score += 1.5

        # Fold match
        if fold and c.get("fold_bias") == fold:
            score += 1

        # Confidence bonus
        conf_bonus = {"high": 1, "medium": 0.5, "low": 0}
        score += conf_bonus.get(c.get("confidence", "low"), 0)

        # pLDDT bonus (normalized)
        score += (c_plddt - 80) / 20  # 0-1 bonus for pLDDT 80-100

        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


def build_design_plan(
    goal: str | None,
    references: list[dict[str, Any]],
    *,
    ec: str | None,
    reaction: str | None,
    metals: list[str] | None,
    cofactors: list[str] | None,
    pocket: str | None,
    fold: str | None,
    length: int | None,
) -> dict[str, Any]:
    """Build a unified design plan from references and user requirements."""

    # Aggregate reference information
    ref_mechanisms = [r.get("reaction_mechanism") for r in references if r.get("reaction_mechanism")]
    ref_styles = [r.get("active_site_style") for r in references if r.get("active_site_style") != "unknown"]
    ref_roles = set()
    for r in references:
        for role in (r.get("required_roles") or []):
            ref_roles.add(role)
    ref_lengths = [r.get("sequence_length") for r in references if r.get("sequence_length")]

    # Consensus from references
    from collections import Counter
    consensus_mechanism = Counter(ref_mechanisms).most_common(1)[0][0] if ref_mechanisms else "unknown"
    consensus_style = Counter(ref_styles).most_common(1)[0][0] if ref_styles else "unknown"
    median_length = sorted(ref_lengths)[len(ref_lengths)//2] if ref_lengths else None

    plan = {
        "design_goal": goal,
        "user_specifications": {
            "ec_prefix": ec,
            "reaction_type": reaction,
            "metals": metals,
            "cofactors": cofactors,
            "pocket_profile": pocket,
            "fold_bias": fold,
            "target_length": length or median_length,
        },
        "reference_consensus": {
            "n_references": len(references),
            "reaction_mechanism": consensus_mechanism,
            "active_site_style": consensus_style,
            "required_roles": sorted(ref_roles),
            "median_length": median_length,
            "reference_accessions": [r["accession"] for r in references],
        },
        "rfd3_ready_fields": {},
        "rfd3_pending_fields": {},
    }

    # Fill RFD3-ready fields
    ready = plan["rfd3_ready_fields"]
    target_len = length or median_length
    if target_len:
        ready["length"] = target_len

    if fold == "helical" or consensus_style in ("cofactor_dependent",) and any(r.get("is_non_loopy") for r in references):
        ready["is_non_loopy"] = True

    # Pending fields with explanations
    plan["rfd3_pending_fields"] = {
        "input": "Provide a motif/template structure. Closest references: " + ", ".join(r["accession"] for r in references[:3]),
        "contig": f"Scaffolding around a {consensus_style} active site, target length {target_len}.",
        "select_hotspots": f"Derive from {pocket or 'semi_buried'} pocket of reference structure.",
        "select_fixed_atoms": "Define after motif selection from reference or theozyme.",
    }

    return plan


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Merge text-parsed and explicit arguments
    text_hints = parse_goal_text(args.goal) if args.goal else {}
    ec = args.ec or text_hints.get("ec")
    reaction = args.reaction or text_hints.get("reaction")
    metals = args.metals or text_hints.get("metals")
    cofactors = args.cofactors or text_hints.get("cofactors")
    pocket = args.pocket or text_hints.get("pocket")
    fold = args.fold or text_hints.get("fold")
    length = args.length or text_hints.get("length")

    print(f"Design query:", flush=True)
    if args.goal:
        print(f"  Goal: {args.goal}", flush=True)
    print(f"  EC: {ec or 'any'}", flush=True)
    print(f"  Reaction: {reaction or 'any'}", flush=True)
    print(f"  Metals: {metals or 'any'}", flush=True)
    print(f"  Cofactors: {cofactors or 'any'}", flush=True)
    print(f"  Pocket: {pocket or 'any'}", flush=True)
    print(f"  Fold: {fold or 'any'}", flush=True)
    print(f"  Length: {length or 'from references'}", flush=True)

    # Load constraints
    print(f"\nLoading constraints from {args.constraints_jsonl} ...", flush=True)
    constraints: list[dict[str, Any]] = []
    with args.constraints_jsonl.open("r") as f:
        for line in f:
            constraints.append(json.loads(line))
    print(f"  {len(constraints):,} enzyme constraints loaded.", flush=True)

    # Find references
    print(f"\nFinding top-{args.top_k} reference enzymes ...", flush=True)
    references = find_reference_enzymes(
        constraints,
        ec_prefix=ec,
        reaction=reaction,
        metals=metals,
        cofactors=cofactors,
        pocket=pocket,
        fold=fold,
        min_plddt=args.min_plddt,
        top_k=args.top_k,
    )

    if not references:
        print("  WARNING: No matching references found. Relaxing pLDDT threshold ...", flush=True)
        references = find_reference_enzymes(
            constraints, ec_prefix=ec, reaction=reaction, metals=metals,
            cofactors=cofactors, pocket=pocket, fold=fold,
            min_plddt=50.0, top_k=args.top_k,
        )

    print(f"  Found {len(references)} references:", flush=True)
    for r in references:
        print(f"    {r['accession']} EC={r['ec_number']:12s} "
              f"style={r.get('active_site_style','?'):20s} "
              f"pLDDT={r.get('plddt_summary',{}).get('mean',0):.0f} "
              f"len={r.get('sequence_length',0)}", flush=True)

    # Build plan
    print(f"\nBuilding design plan ...", flush=True)
    plan = build_design_plan(
        goal=args.goal, references=references,
        ec=ec, reaction=reaction, metals=metals, cofactors=cofactors,
        pocket=pocket, fold=fold, length=length,
    )

    # Write outputs
    plan_path = args.output_dir / "design_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    refs_path = args.output_dir / "reference_enzymes.jsonl"
    with refs_path.open("w", encoding="utf-8") as f:
        for r in references:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also generate RFD3 plan via the planner for the top reference
    if references:
        top_ref = references[0]
        task = {
            "task_id": "design_query",
            "substrate_name": top_ref.get("substrate_family"),
            "metal_type": metals or top_ref.get("metal_hint") or [],
            "reaction_class": reaction or top_ref.get("reaction_type"),
            "enzyme_family": ec,
            "fold_class": fold,
        }
        requirement = {
            "task_id": "design_query",
            "goal_text": args.goal or f"Design enzyme similar to {top_ref['accession']}",
            "reaction_type": reaction or top_ref.get("reaction_type"),
            "substrate_or_ts": top_ref.get("substrate_family"),
            "length_preferences": {
                "target_len": length or top_ref.get("sequence_length"),
            },
            "pocket_constraints": {"burial": pocket or top_ref.get("pocket_profile")},
            "required_roles": top_ref.get("required_roles") or [],
        }

        task_path = args.output_dir / "task.json"
        req_path = args.output_dir / "requirement.json"
        task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
        req_path.write_text(json.dumps(requirement, indent=2, ensure_ascii=False))

        # Run planner
        result = subprocess.run(
            [sys.executable, str(PLANNER_SCRIPT),
             "--task-json", str(task_path),
             "--requirement-json", str(req_path),
             "--output-dir", str(args.output_dir)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"\nRFD3 constraint plan generated.", flush=True)
        else:
            print(f"\nPlanner warning: {result.stderr[:200]}", flush=True)

    print(f"\nOutputs written to {args.output_dir}/", flush=True)
    print(f"  design_plan.json          - unified design plan", flush=True)
    print(f"  reference_enzymes.jsonl   - top matching enzymes", flush=True)
    print(f"  constraint_plan.json      - RFD3 constraint plan", flush=True)
    print(f"  rfd3_input_spec_draft.json - RFD3 input spec", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
