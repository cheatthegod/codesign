#!/usr/bin/env python3
"""Generate Route A structured constraints from enriched Prot2Text data.

Converts enriched enzyme annotations (Layer 1) and geometric features (Layer 2)
into structured constraint records suitable for:
  - A text-to-constraint controller (Route A)
  - Downstream RFD3 inference planning
  - Data analysis and filtering

All extraction is deterministic (rule-based). Fields that require LLM enrichment
are marked with confidence="pending_llm".

Outputs:
- prot2text_route_a_constraints.jsonl   One JSON per line per enzyme
- route_a_constraint_summary.json       Coverage statistics
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ENRICHED_PATH = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_enriched_full.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"

# ============================================================================
# Controlled vocabularies
# ============================================================================

EC_REACTION_TYPE = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}

# GO → metal mapping (sorted by specificity, most specific first)
GO_METAL_MAP = {
    "GO:0008270": "zinc",
    "GO:0005506": "iron",
    "GO:0000287": "magnesium",
    "GO:0030145": "manganese",
    "GO:0005507": "copper",
    "GO:0005509": "calcium",
    "GO:0030151": "nickel",
    "GO:0051539": "iron",  # iron-sulfur cluster
}

# GO → cofactor mapping
GO_COFACTOR_MAP = {
    "GO:0030170": "PLP",
    "GO:0050660": "FAD",
    "GO:0010181": "FMN",
    "GO:0020037": "heme",
    "GO:0034617": "TPP",
    "GO:0009235": "cobalamin",
    "GO:0070403": "NAD",
    "GO:0070401": "NADP",
}

# Text patterns for cofactors (applied to function text)
TEXT_COFACTOR_PATTERNS = [
    (re.compile(r"\bnadph?\b", re.I), "NAD"),
    (re.compile(r"\bnadp\b", re.I), "NADP"),
    (re.compile(r"\bfad\b", re.I), "FAD"),
    (re.compile(r"\bfmn\b", re.I), "FMN"),
    (re.compile(r"\bplp\b|pyridoxal.phosphate", re.I), "PLP"),
    (re.compile(r"\btpp\b|thiamine.pyrophosphate|thiamin.diphosphate", re.I), "TPP"),
    (re.compile(r"\bcoa\b|coenzyme.a\b|acetyl-coa", re.I), "CoA"),
    (re.compile(r"\bheme\b|\bhaem\b", re.I), "heme"),
    (re.compile(r"\bbiotin\b", re.I), "biotin"),
    (re.compile(r"\bfolate\b|\btetrahydrofolate\b", re.I), "folate"),
    (re.compile(r"\bs-adenosyl|sam\b", re.I), "SAM"),
]

# Text patterns for metals
TEXT_METAL_PATTERNS = [
    (re.compile(r"\bzinc\b|\bzn2?\+|\bzn\(ii\)", re.I), "zinc"),
    (re.compile(r"\biron\b|\bfe2?\+|\bfe\(ii\)|\bfe\(iii\)|\bdiiron\b", re.I), "iron"),
    (re.compile(r"\bmanganese\b|\bmn2?\+|\bmn\(ii\)", re.I), "manganese"),
    (re.compile(r"\bmagnesium\b|\bmg2?\+", re.I), "magnesium"),
    (re.compile(r"\bcopper\b|\bcu2?\+|\bcu\(ii\)", re.I), "copper"),
    (re.compile(r"\bcalcium\b|\bca2\+", re.I), "calcium"),
    (re.compile(r"\bcobalt\b|\bco2?\+", re.I), "cobalt"),
    (re.compile(r"\bnickel\b|\bni2?\+", re.I), "nickel"),
    (re.compile(r"\bmolybdenum\b|\bmo\b", re.I), "molybdenum"),
]

# Substrate family inference from EC second level and function text
EC2_SUBSTRATE_HINTS = {
    # EC 2.x = transferases
    "2.1": "one_carbon_group",
    "2.3": "acyl_group",
    "2.4": "glycosyl_group",
    "2.6": "amino_group",
    "2.7": "phosphorus_group",
    # EC 3.x = hydrolases
    "3.1": "ester",
    "3.2": "glycoside",
    "3.4": "peptide",
    "3.5": "amide",
    # EC 1.x = oxidoreductases
    "1.1": "alcohol",
    "1.2": "aldehyde",
    "1.3": "ch_ch_group",
    "1.14": "paired_donor",
}

TEXT_SUBSTRATE_PATTERNS = [
    (re.compile(r"\bprotein\b|\bpeptide\b|\bpolypeptide\b", re.I), "protein"),
    (re.compile(r"\bdna\b|\brna\b|\bnucleic.acid\b|\bnucleotide\b", re.I), "nucleic_acid"),
    (re.compile(r"\bcarbohydrate\b|\bsugar\b|\bglucose\b|\bxylose\b|\bmannose\b|\bgalactose\b|\bfructose\b", re.I), "carbohydrate"),
    (re.compile(r"\blipid\b|\bfatty.acid\b|\bsphingolipid\b|\bceramide\b|\bphospholipid\b", re.I), "lipid"),
    (re.compile(r"\bamino.acid\b", re.I), "amino_acid"),
    (re.compile(r"\bsteroid\b|\bsterol\b|\bcholesterol\b|\bcortisol\b|\btestosterone\b|\bestrogen\b", re.I), "steroid"),
    (re.compile(r"\bterpene\b|\bterpenoid\b|\bisoprenoid\b|\bcarotenoid\b|\btrichothecene\b|\bsqualene\b", re.I), "terpene"),
    (re.compile(r"\baromatic\b|\bphenyl\b|\bbenzene\b|\bdiphenyl\b|\bflavonoid\b|\blignin\b", re.I), "aromatic"),
    (re.compile(r"\bnucleotide\b|\bpurine\b|\bpyrimidine\b|\batp\b|\bgtp\b", re.I), "nucleotide"),
]

# ============================================================================
# EC → reaction_mechanism mapping
# ============================================================================

EC1_MECHANISM_MAP = {
    "1": "oxidation",        # oxidoreductases: oxidation/reduction
    "2": "group_transfer",   # transferases
    "3": "hydrolysis",       # hydrolases
    "4": "elimination",      # lyases: elimination to form double bonds
    "5": "isomerization",    # isomerases
    "6": "ligation",         # ligases: bond formation coupled to ATP/NTP hydrolysis
    "7": "group_transfer",   # translocases: transport coupled to group transfer
}

# More specific EC sub-class → mechanism overrides
EC2_MECHANISM_MAP = {
    "1.13": "oxidation",         # oxygenases
    "1.14": "oxidation",         # monooxygenases
    "1.97": "electron_transfer", # other oxidoreductases
    "2.7": "group_transfer",     # phosphotransferases
    "3.1": "hydrolysis",         # ester hydrolases
    "3.4": "hydrolysis",         # peptidases
    "4.1": "decarboxylation",    # carboxy-lyases
    "4.2": "elimination",        # hydro-lyases
    "4.3": "elimination",        # ammonia-lyases
    "4.6": "elimination",        # phosphorus-oxygen lyases (cyclases)
}

# ============================================================================
# active_site_style inference rules
# ============================================================================


def infer_active_site_style(
    ec_number: str,
    metals: list[str],
    cofactors: list[str],
    go_ids: str,
    function_text: str,
) -> str:
    """Infer active site architecture from available annotations."""
    func_lower = (function_text or "").lower()

    # Iron-sulfur cluster
    if "GO:0051539" in (go_ids or "") or "iron-sulfur" in func_lower or "fe-s" in func_lower:
        return "metal_cluster"

    # Diiron / binuclear
    if "diiron" in func_lower or "binuclear" in func_lower or "di-iron" in func_lower:
        return "binuclear_metal"

    # Radical SAM / radical-based
    if "radical sam" in func_lower or "radical" in func_lower and "SAM" in cofactors:
        return "radical_based"

    # PLP → covalent intermediate (Schiff base)
    if "PLP" in cofactors:
        return "covalent_intermediate"

    # Cofactor-dependent (NAD/FAD/FMN/heme/CoA/TPP etc.)
    cofactor_set = set(cofactors) - {"PLP"}
    if cofactor_set & {"NAD", "NADP", "FAD", "FMN", "heme", "CoA", "TPP", "biotin", "folate"}:
        return "cofactor_dependent"

    # Multiple metals → binuclear
    if len(metals) >= 2:
        return "binuclear_metal"

    # Single metal
    if metals:
        # Serine protease with zinc → metal_center
        return "metal_center"

    # Serine protease / catalytic triad patterns
    ec_parts = (ec_number or "").split(".")
    ec2 = f"{ec_parts[0]}.{ec_parts[1]}" if len(ec_parts) >= 2 else ""
    if ec2 == "3.4":
        if re.search(r"\bserine\b.*\b(protease|peptidase|endopeptidase)\b", func_lower):
            return "catalytic_triad"
        if re.search(r"\bcysteine\b.*\b(protease|peptidase)\b", func_lower):
            return "catalytic_dyad"
        return "acid_base"

    # Esterases / lipases often have catalytic triad
    if ec2 == "3.1" and re.search(r"\blipase\b|\besterase\b", func_lower):
        return "catalytic_triad"

    # General acid-base for remaining hydrolases
    if ec_parts and ec_parts[0] == "3":
        return "acid_base"

    # Transferases without cofactors → acid_base or mixed
    if ec_parts and ec_parts[0] == "2":
        return "mixed"

    return "unknown"


# ============================================================================
# required_roles inference rules
# ============================================================================


def infer_required_roles(
    ec_number: str,
    metals: list[str],
    cofactors: list[str],
    active_site_style: str,
    function_text: str,
) -> list[str]:
    """Infer catalytic residue roles from annotations."""
    roles: set[str] = set()
    ec_parts = (ec_number or "").split(".")
    ec1 = ec_parts[0] if ec_parts else ""
    func_lower = (function_text or "").lower()

    # Metal-related roles
    if metals:
        roles.add("metal_ligand")
    if "diiron" in func_lower or "binuclear" in func_lower:
        roles.add("metal_ligand")

    # Cofactor binding
    if cofactors:
        roles.add("cofactor_binding")

    # Electron transfer for oxidoreductases
    if ec1 == "1":
        roles.add("electron_transfer")

    # Hydrolases typically need nucleophile + general acid/base
    if ec1 == "3":
        roles.add("nucleophile")
        roles.add("general_acid")
        roles.add("general_base")
        if active_site_style in ("catalytic_triad", "catalytic_dyad"):
            roles.add("oxyanion_stabilizer")

    # Transferases need substrate positioning
    if ec1 == "2":
        roles.add("substrate_positioning")

    # Ligases
    if ec1 == "6":
        roles.add("substrate_positioning")

    # Lyases
    if ec1 == "4":
        roles.add("general_base")
        if "decarboxyl" in func_lower:
            roles.add("proton_shuttle")

    # Radical chemistry
    if "radical" in func_lower:
        roles.add("radical_generator")

    # Transition state stabilization (common in well-characterized enzymes)
    if active_site_style in ("catalytic_triad", "catalytic_dyad", "metal_center"):
        roles.add("transition_state_stabilizer")

    return sorted(roles)


def infer_reaction_mechanism(ec_number: str, function_text: str) -> str:
    """Infer reaction mechanism from EC number and function text."""
    if not ec_number:
        return "unknown"

    ec_parts = ec_number.strip().split(".")
    ec1 = ec_parts[0]

    # Check more specific EC sub-class first
    if len(ec_parts) >= 2:
        ec2 = f"{ec_parts[0]}.{ec_parts[1]}"
        if ec2 in EC2_MECHANISM_MAP:
            return EC2_MECHANISM_MAP[ec2]

    # Fall back to EC first digit
    return EC1_MECHANISM_MAP.get(ec1, "unknown")


# ============================================================================
# Extraction functions
# ============================================================================


def extract_reaction_type(ec_number: str) -> str | None:
    """Infer reaction type from EC first digit."""
    if not ec_number:
        return None
    first = ec_number.strip().split(".")[0]
    return EC_REACTION_TYPE.get(first)


def extract_metals(go_ids: str, function_text: str) -> list[str]:
    """Extract metal hints from GO terms and function text."""
    metals: set[str] = set()

    # From GO terms
    if go_ids:
        for go_id, metal in GO_METAL_MAP.items():
            if go_id in go_ids:
                metals.add(metal)

    # From function text
    if function_text:
        for pattern, metal in TEXT_METAL_PATTERNS:
            if pattern.search(function_text):
                metals.add(metal)

    return sorted(metals)


def extract_cofactors(go_ids: str, function_text: str) -> list[str]:
    """Extract cofactor hints from GO terms and function text."""
    cofactors: set[str] = set()

    # From GO terms
    if go_ids:
        for go_id, cofactor in GO_COFACTOR_MAP.items():
            if go_id in go_ids:
                cofactors.add(cofactor)

    # From function text
    if function_text:
        for pattern, cofactor in TEXT_COFACTOR_PATTERNS:
            if pattern.search(function_text):
                cofactors.add(cofactor)

    return sorted(cofactors)


def extract_substrate_family(ec_number: str, function_text: str) -> str | None:
    """Infer substrate family from EC sub-class and function text."""
    # Try EC second level
    if ec_number:
        ec_parts = ec_number.strip().split(".")
        if len(ec_parts) >= 2:
            ec2 = f"{ec_parts[0]}.{ec_parts[1]}"
            hint = EC2_SUBSTRATE_HINTS.get(ec2)
            if hint:
                return hint

    # Try function text
    if function_text:
        for pattern, family in TEXT_SUBSTRATE_PATTERNS:
            if pattern.search(function_text):
                return family

    return None


def infer_fold_bias(ss_helix: float, ss_sheet: float, ss_loop: float) -> str:
    """Infer fold bias from secondary structure fractions."""
    if ss_helix > 0.50:
        return "helical"
    if ss_sheet > 0.30:
        return "sheet_rich"
    if ss_loop > 0.60:
        return "loop_dominated"
    return "mixed"


def infer_pocket_profile(buried_frac: float, exposed_frac: float) -> str:
    """Infer pocket accessibility from burial statistics."""
    if buried_frac > 0.35:
        return "deeply_buried"
    if buried_frac > 0.20:
        return "semi_buried"
    return "surface_exposed"


def infer_oligomer_hint(num_chains: int) -> str:
    """Infer oligomeric state from chain count."""
    if num_chains == 1:
        return "monomer"
    if num_chains == 2:
        return "dimer"
    return f"oligomer_{num_chains}"


def compute_confidence(
    ec_number: str,
    metals: list[str],
    cofactors: list[str],
    substrate_family: str | None,
    mean_plddt: float,
) -> tuple[str, list[str]]:
    """Compute confidence level and list evidence sources."""
    evidence: list[str] = []
    score = 0

    if ec_number:
        evidence.append("ec_number")
        score += 2
    if metals:
        evidence.append("metal_annotation")
        score += 1
    if cofactors:
        evidence.append("cofactor_annotation")
        score += 1
    if substrate_family:
        evidence.append("substrate_inference")
        score += 1
    if mean_plddt >= 70:
        evidence.append("high_plddt_structure")
        score += 1

    if score >= 4:
        return "high", evidence
    if score >= 2:
        return "medium", evidence
    return "low", evidence


def build_constraint(row: dict[str, Any]) -> dict[str, Any]:
    """Build a structured constraint record for one enzyme protein."""
    ec = row.get("ec_number", "") or ""
    func = row.get("function", "") or ""
    go = row.get("go_ids", "") or ""

    reaction_type = extract_reaction_type(ec)
    metals = extract_metals(go, func)
    cofactors = extract_cofactors(go, func)
    substrate_family = extract_substrate_family(ec, func)

    ss_helix = row.get("ss_fraction_helix", 0.0) or 0.0
    ss_sheet = row.get("ss_fraction_sheet", 0.0) or 0.0
    ss_loop = row.get("ss_fraction_loop", 0.0) or 0.0
    buried_frac = row.get("buried_fraction", 0.0) or 0.0
    exposed_frac = row.get("exposed_fraction", 0.0) or 0.0
    mean_plddt = row.get("mean_plddt", 0.0) or 0.0
    num_chains = row.get("struct_num_chains", 1) or 1

    fold_bias = infer_fold_bias(ss_helix, ss_sheet, ss_loop)
    pocket_profile = infer_pocket_profile(buried_frac, exposed_frac)
    oligomer_hint = infer_oligomer_hint(num_chains)

    # New Layer 3 fields (rule-based)
    reaction_mechanism = infer_reaction_mechanism(ec, func)
    active_site_style = infer_active_site_style(ec, metals, cofactors, go, func)
    required_roles = infer_required_roles(ec, metals, cofactors, active_site_style, func)

    confidence, evidence = compute_confidence(ec, metals, cofactors, substrate_family, mean_plddt)

    constraint: dict[str, Any] = {
        # Identity
        "accession": row["accession"],
        "enzyme_class": row.get("enzyme_class", ""),
        "ec_number": ec,
        "source_split": row.get("source_split", ""),
        # Structured constraints
        "reaction_type": reaction_type,
        "reaction_mechanism": reaction_mechanism,
        "substrate_family": substrate_family,
        "cofactor_hint": cofactors if cofactors else None,
        "metal_hint": metals if metals else None,
        "active_site_style": active_site_style,
        "required_roles": required_roles if required_roles else None,
        "fold_bias": fold_bias,
        "pocket_profile": pocket_profile,
        "oligomer_hint": oligomer_hint,
        # Structure-derived conditioning hints
        "is_non_loopy": bool(row.get("is_non_loopy", False)),
        "ss_summary": {
            "helix": round(ss_helix, 3),
            "sheet": round(ss_sheet, 3),
            "loop": round(ss_loop, 3),
        },
        "burial_summary": {
            "buried_fraction": round(buried_frac, 3),
            "exposed_fraction": round(exposed_frac, 3),
        },
        "plddt_summary": {
            "mean": round(mean_plddt, 1),
            "min": round(row.get("min_plddt", 0.0) or 0.0, 1),
        },
        # Length for RFD3
        "sequence_length": int(row.get("sequence_length", 0) or 0),
        # Confidence and provenance
        "confidence": confidence,
        "evidence_sources": evidence,
        # Original text (for downstream LLM enrichment or controller)
        "function_text": func,
        "protein_names": row.get("protein_names", "") or "",
    }

    return constraint


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enriched-parquet",
        type=Path,
        default=DEFAULT_ENRICHED_PATH,
        help="Path to prot2text_enriched_full.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--include-non-enzyme",
        action="store_true",
        help="Also generate constraints for non-enzyme proteins (with minimal fields).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Only process these splits. Default: all.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading enriched data from {args.enriched_parquet} ...", flush=True)
    df = pd.read_parquet(args.enriched_parquet)

    if args.splits:
        df = df[df["source_split"].isin(args.splits)].copy()
        print(f"  Filtered to splits: {args.splits}", flush=True)

    if not args.include_non_enzyme:
        df = df[df["enzyme_class"].isin(["enzyme_gold", "enzyme_silver"])].copy()
        print(f"  Filtered to enzymes: {len(df):,} rows.", flush=True)
    else:
        print(f"  Including all proteins: {len(df):,} rows.", flush=True)

    print("[2/3] Generating constraints ...", flush=True)
    output_path = args.output_dir / "prot2text_route_a_constraints.jsonl"

    # Track coverage stats
    stats = {
        "total": 0,
        "has_reaction_type": 0,
        "has_reaction_mechanism": 0,
        "has_substrate_family": 0,
        "has_cofactor": 0,
        "has_metal": 0,
        "has_active_site_style": 0,
        "has_required_roles": 0,
        "has_fold_bias_non_mixed": 0,
        "confidence_high": 0,
        "confidence_medium": 0,
        "confidence_low": 0,
        "fold_bias_counts": {},
        "pocket_profile_counts": {},
        "reaction_type_counts": {},
        "active_site_style_counts": {},
        "reaction_mechanism_counts": {},
    }

    with output_path.open("w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            constraint = build_constraint(row.to_dict())
            fout.write(json.dumps(constraint, ensure_ascii=False) + "\n")

            # Update stats
            stats["total"] += 1
            if constraint["reaction_type"]:
                stats["has_reaction_type"] += 1
                rt = constraint["reaction_type"]
                stats["reaction_type_counts"][rt] = stats["reaction_type_counts"].get(rt, 0) + 1
            if constraint["reaction_mechanism"] and constraint["reaction_mechanism"] != "unknown":
                stats["has_reaction_mechanism"] += 1
                rm = constraint["reaction_mechanism"]
                stats["reaction_mechanism_counts"][rm] = stats["reaction_mechanism_counts"].get(rm, 0) + 1
            if constraint["substrate_family"]:
                stats["has_substrate_family"] += 1
            if constraint["cofactor_hint"]:
                stats["has_cofactor"] += 1
            if constraint["metal_hint"]:
                stats["has_metal"] += 1
            if constraint["active_site_style"] and constraint["active_site_style"] != "unknown":
                stats["has_active_site_style"] += 1
                ass = constraint["active_site_style"]
                stats["active_site_style_counts"][ass] = stats["active_site_style_counts"].get(ass, 0) + 1
            if constraint["required_roles"]:
                stats["has_required_roles"] += 1
            if constraint["fold_bias"] != "mixed":
                stats["has_fold_bias_non_mixed"] += 1

            fb = constraint["fold_bias"]
            stats["fold_bias_counts"][fb] = stats["fold_bias_counts"].get(fb, 0) + 1
            pp = constraint["pocket_profile"]
            stats["pocket_profile_counts"][pp] = stats["pocket_profile_counts"].get(pp, 0) + 1

            conf = constraint["confidence"]
            stats[f"confidence_{conf}"] += 1

            if i % 20000 == 0:
                print(f"  {i:,}/{len(df):,} ({i/len(df):.1%})", flush=True)

    print(f"  Wrote {stats['total']:,} constraints to {output_path}", flush=True)

    print("[3/3] Writing summary ...", flush=True)
    total = stats["total"]
    summary = {
        "total_constraints": total,
        "coverage": {
            "reaction_type": f"{stats['has_reaction_type']:,} ({stats['has_reaction_type']/total:.1%})",
            "reaction_mechanism": f"{stats['has_reaction_mechanism']:,} ({stats['has_reaction_mechanism']/total:.1%})",
            "substrate_family": f"{stats['has_substrate_family']:,} ({stats['has_substrate_family']/total:.1%})",
            "cofactor_hint": f"{stats['has_cofactor']:,} ({stats['has_cofactor']/total:.1%})",
            "metal_hint": f"{stats['has_metal']:,} ({stats['has_metal']/total:.1%})",
            "active_site_style": f"{stats['has_active_site_style']:,} ({stats['has_active_site_style']/total:.1%})",
            "required_roles": f"{stats['has_required_roles']:,} ({stats['has_required_roles']/total:.1%})",
            "fold_bias_non_mixed": f"{stats['has_fold_bias_non_mixed']:,} ({stats['has_fold_bias_non_mixed']/total:.1%})",
        },
        "confidence_distribution": {
            "high": f"{stats['confidence_high']:,} ({stats['confidence_high']/total:.1%})",
            "medium": f"{stats['confidence_medium']:,} ({stats['confidence_medium']/total:.1%})",
            "low": f"{stats['confidence_low']:,} ({stats['confidence_low']/total:.1%})",
        },
        "reaction_type_distribution": {
            k: v for k, v in sorted(stats["reaction_type_counts"].items(), key=lambda x: -x[1])
        },
        "fold_bias_distribution": {
            k: v for k, v in sorted(stats["fold_bias_counts"].items(), key=lambda x: -x[1])
        },
        "pocket_profile_distribution": {
            k: v for k, v in sorted(stats["pocket_profile_counts"].items(), key=lambda x: -x[1])
        },
        "active_site_style_distribution": {
            k: v for k, v in sorted(stats["active_site_style_counts"].items(), key=lambda x: -x[1])
        },
        "reaction_mechanism_distribution": {
            k: v for k, v in sorted(stats["reaction_mechanism_counts"].items(), key=lambda x: -x[1])
        },
    }

    summary_path = args.output_dir / "route_a_constraint_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
