#!/usr/bin/env python3
"""Layer 3: LLM-based high-level semantic constraint extraction.

Supports two backends:
  - OpenRouter (default): uses OpenAI-compatible API via OPENROUTER_API_KEY
  - Anthropic native: uses ANTHROPIC_API_KEY

Extracts enzyme design constraints using structured output (tool_use / function calling):
- required_roles, active_site_style, reaction_mechanism, substrate_family, evidence_spans

All outputs use controlled vocabularies enforced via JSON schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CONSTRAINTS_PATH = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_route_a_constraints.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched"

# ============================================================================
# Controlled vocabularies for LLM output
# ============================================================================

REQUIRED_ROLES = [
    "general_acid", "general_base", "nucleophile", "electrophile",
    "metal_ligand", "proton_shuttle", "oxyanion_stabilizer",
    "radical_generator", "electron_transfer", "substrate_positioning",
    "cofactor_binding", "transition_state_stabilizer",
]

ACTIVE_SITE_STYLES = [
    "catalytic_triad", "catalytic_dyad", "metal_center", "binuclear_metal",
    "metal_cluster", "cofactor_dependent", "radical_based",
    "covalent_intermediate", "acid_base", "mixed", "unknown",
]

REACTION_MECHANISMS = [
    "nucleophilic_substitution", "elimination", "oxidation", "reduction",
    "group_transfer", "isomerization", "ligation", "hydrolysis",
    "radical", "electron_transfer", "decarboxylation", "condensation", "unknown",
]

SUBSTRATE_FAMILIES = [
    "protein", "peptide", "amino_acid", "nucleic_acid", "nucleotide",
    "carbohydrate", "lipid", "fatty_acid", "steroid", "terpene",
    "alcohol", "aldehyde", "ketone", "ester", "ether", "amide",
    "phosphate_ester", "thioester", "glycoside",
    "aromatic", "heterocyclic", "small_molecule",
    "cofactor", "metal_ion", "gas", "unknown",
]

# ============================================================================
# Tool / function schema (shared between backends)
# ============================================================================

TOOL_PROPERTIES = {
    "required_roles": {
        "type": "array",
        "items": {"type": "string", "enum": REQUIRED_ROLES},
        "description": "Catalytic residue roles needed for this enzyme's mechanism. Only include roles you are confident about.",
    },
    "active_site_style": {
        "type": "string",
        "enum": ACTIVE_SITE_STYLES,
        "description": "Architecture pattern of the active site.",
    },
    "reaction_mechanism": {
        "type": "string",
        "enum": REACTION_MECHANISMS,
        "description": "Primary reaction mechanism.",
    },
    "substrate_family_refined": {
        "type": "string",
        "enum": SUBSTRATE_FAMILIES,
        "description": "Refined substrate family based on the full function description.",
    },
    "evidence_spans": {
        "type": "array",
        "items": {"type": "string"},
        "description": "2-4 short key phrases from the function text that support your extraction. Each span should be under 15 words.",
    },
    "confidence": {
        "type": "string",
        "enum": ["high", "medium", "low"],
        "description": "Your confidence in the overall extraction. 'high' = clear from text, 'medium' = inferred from EC/GO, 'low' = guessed.",
    },
}

REQUIRED_FIELDS = [
    "required_roles", "active_site_style", "reaction_mechanism",
    "substrate_family_refined", "evidence_spans", "confidence",
]

# Anthropic format
ANTHROPIC_TOOL = {
    "name": "extract_enzyme_constraints",
    "description": "Extract structured enzyme design constraints from protein annotations.",
    "input_schema": {
        "type": "object",
        "properties": TOOL_PROPERTIES,
        "required": REQUIRED_FIELDS,
    },
}

# OpenAI/OpenRouter format
OPENAI_FUNCTION = {
    "type": "function",
    "function": {
        "name": "extract_enzyme_constraints",
        "description": "Extract structured enzyme design constraints from protein annotations.",
        "parameters": {
            "type": "object",
            "properties": TOOL_PROPERTIES,
            "required": REQUIRED_FIELDS,
        },
    },
}

SYSTEM_PROMPT = """You are an expert enzymologist. Given an enzyme's annotations and function description, extract structured design constraints using the provided tool.

Rules:
- Only include required_roles you are confident about from the description or EC class.
- For active_site_style, infer from the enzyme class and any mentioned catalytic residues or cofactors.
- evidence_spans must be actual phrases from the function text, not your own words.
- If the function text is vague, set confidence to "low" and use "unknown" for uncertain fields.
- Be concise and precise. Do not hallucinate mechanisms not supported by the text."""

# JSON-mode system prompt (compact, for models without good tool_use support)
JSON_SYSTEM_PROMPT = """You are an expert enzymologist. Extract enzyme design constraints as JSON.

Output ONLY a JSON object (no markdown fences, no explanation):

{
  "required_roles": ["general_acid","general_base","nucleophile","metal_ligand","cofactor_binding","electron_transfer","substrate_positioning","transition_state_stabilizer"],
  "active_site_style": "catalytic_triad|catalytic_dyad|metal_center|binuclear_metal|metal_cluster|cofactor_dependent|radical_based|covalent_intermediate|acid_base|mixed|unknown",
  "reaction_mechanism": "oxidation|reduction|hydrolysis|group_transfer|nucleophilic_substitution|elimination|isomerization|ligation|radical|electron_transfer|decarboxylation|condensation|unknown",
  "substrate_family_refined": "protein|peptide|amino_acid|nucleic_acid|nucleotide|carbohydrate|lipid|fatty_acid|steroid|terpene|alcohol|aldehyde|ketone|ester|amide|phosphate_ester|thioester|glycoside|aromatic|heterocyclic|small_molecule|unknown",
  "evidence_spans": ["2-3 short phrases FROM the function text"],
  "confidence": "high|medium|low"
}

Pick only applicable roles. Use "unknown" when uncertain. evidence_spans must be actual text from the input."""


# ============================================================================
# Prompt construction
# ============================================================================


def build_prompt(record: dict[str, Any]) -> str:
    """Build a concise prompt from an enriched constraint record."""
    parts = [
        f"EC: {record.get('ec_number', 'unknown')}",
        f"Reaction type: {record.get('reaction_type', 'unknown')}",
    ]
    if record.get("metal_hint"):
        parts.append(f"Known metals: {', '.join(record['metal_hint'])}")
    if record.get("cofactor_hint"):
        parts.append(f"Known cofactors: {', '.join(record['cofactor_hint'])}")
    if record.get("substrate_family"):
        parts.append(f"Rule-based substrate hint: {record['substrate_family']}")
    parts.append(f"Fold bias: {record.get('fold_bias', 'unknown')}")
    parts.append(f"Pocket profile: {record.get('pocket_profile', 'unknown')}")

    header = " | ".join(parts)
    func_text = (record.get("function_text") or "").strip()
    prot_name = (record.get("protein_names") or "").strip()

    if len(func_text) > 600:
        func_text = func_text[:600] + "..."
    if len(prot_name) > 200:
        prot_name = prot_name[:200] + "..."

    return f"""{header}

Protein: {prot_name}

Function: {func_text}"""


# ============================================================================
# Backend: OpenRouter (OpenAI-compatible)
# ============================================================================


def call_openrouter(client: Any, prompt: str, model: str) -> dict[str, Any] | None:
    """Call OpenRouter API with function calling for structured extraction."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=[OPENAI_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "extract_enzyme_constraints"}},
    )

    choice = response.choices[0]
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            if tc.function.name == "extract_enzyme_constraints":
                return json.loads(tc.function.arguments)

    return None


def call_openrouter_json(client: Any, prompt: str, model: str) -> dict[str, Any] | None:
    """Call OpenRouter API with plain JSON output (for models with poor tool_use support)."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        return None

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    # Find JSON object boundaries if there's surrounding text
    start = content.find("{")
    if start < 0:
        return None
    content = content[start:]

    # Try parsing as-is first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # If truncated (finish_reason=length), try to find the last complete field
    end = content.rfind("}")
    if end > 0:
        try:
            return json.loads(content[:end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: try adding closing brackets
    for suffix in ["}", '"}', '"]}', '"],"confidence":"low"}']:
        try:
            return json.loads(content + suffix)
        except json.JSONDecodeError:
            continue

    return None


# ============================================================================
# Backend: Anthropic native
# ============================================================================


def call_anthropic(client: Any, prompt: str, model: str) -> dict[str, Any] | None:
    """Call Anthropic API with tool_use for structured extraction."""
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[ANTHROPIC_TOOL],
        tool_choice={"type": "tool", "name": "extract_enzyme_constraints"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_enzyme_constraints":
            return block.input

    return None


# ============================================================================
# Processing
# ============================================================================


def process_batch(
    call_fn,
    client: Any,
    records: list[dict[str, Any]],
    model: str,
    output_path: Path,
    done_accessions: set[str],
    rate_limit_delay: float,
) -> tuple[int, int]:
    """Process a batch of records and append results to output file."""
    success = 0
    errors = 0
    total = len(records)
    t0 = time.time()

    with output_path.open("a", encoding="utf-8") as fout:
        for i, record in enumerate(records, 1):
            acc = record["accession"]
            if acc in done_accessions:
                continue

            try:
                prompt = build_prompt(record)
                result = call_fn(client, prompt, model)

                if result is not None:
                    output = {
                        "accession": acc,
                        "ec_number": record.get("ec_number", ""),
                        "llm_required_roles": result.get("required_roles", []),
                        "llm_active_site_style": result.get("active_site_style", "unknown"),
                        "llm_reaction_mechanism": result.get("reaction_mechanism", "unknown"),
                        "llm_substrate_family": result.get("substrate_family_refined", "unknown"),
                        "llm_evidence_spans": result.get("evidence_spans", []),
                        "llm_confidence": result.get("confidence", "low"),
                        "llm_model": model,
                    }
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    fout.flush()
                    success += 1
                    done_accessions.add(acc)
                else:
                    errors += 1
                    print(f"  WARNING: No tool output for {acc}", file=sys.stderr, flush=True)

            except Exception as e:
                errors += 1
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = 30
                    print(f"  Rate limited at {acc}, sleeping {wait}s ...", flush=True)
                    time.sleep(wait)
                    try:
                        result = call_fn(client, build_prompt(record), model)
                        if result is not None:
                            output = {
                                "accession": acc,
                                "ec_number": record.get("ec_number", ""),
                                "llm_required_roles": result.get("required_roles", []),
                                "llm_active_site_style": result.get("active_site_style", "unknown"),
                                "llm_reaction_mechanism": result.get("reaction_mechanism", "unknown"),
                                "llm_substrate_family": result.get("substrate_family_refined", "unknown"),
                                "llm_evidence_spans": result.get("evidence_spans", []),
                                "llm_confidence": result.get("confidence", "low"),
                                "llm_model": model,
                            }
                            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                            fout.flush()
                            success += 1
                            errors -= 1
                            done_accessions.add(acc)
                    except Exception:
                        print(f"  ERROR retry failed for {acc}", file=sys.stderr, flush=True)
                else:
                    print(f"  ERROR {acc}: {err_str[:120]}", file=sys.stderr, flush=True)

            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  {i:,}/{total:,} ({i/total:.1%}) | {rate:.1f} req/s | ETA {eta:.0f}s | ok={success} err={errors}", flush=True)

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

    return success, errors


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--constraints-jsonl", type=Path, default=DEFAULT_CONSTRAINTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backend",
        choices=["openrouter", "anthropic"],
        default="openrouter",
        help="API backend to use.",
    )
    parser.add_argument(
        "--model",
        default="z-ai/glm-5.1",
        help="Model ID. For OpenRouter use e.g. z-ai/glm-5.1 or z-ai/glm-4.7-flash.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Use JSON-mode output instead of tool_use/function calling. "
             "Better compatibility with models that have poor tool_use support (e.g. GLM).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--filter-confidence", choices=["high", "medium", "low"])
    parser.add_argument("--filter-reaction")
    parser.add_argument("--rate-limit-delay", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filename, e.g. '_glm5' -> layer3_llm_enrichment_glm5.jsonl",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup backend
    if args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
            return 1
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        call_fn = call_openrouter_json if args.json_mode else call_openrouter
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
            return 1
        import anthropic
        client = anthropic.Anthropic()
        call_fn = call_anthropic

    print(f"Backend: {args.backend} | Model: {args.model}", flush=True)

    print(f"[1/3] Loading constraints from {args.constraints_jsonl} ...", flush=True)
    records: list[dict[str, Any]] = []
    with args.constraints_jsonl.open("r") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records.", flush=True)

    if args.filter_confidence:
        records = [r for r in records if r.get("confidence") == args.filter_confidence]
        print(f"  Filtered to confidence={args.filter_confidence}: {len(records):,}", flush=True)
    if args.filter_reaction:
        records = [r for r in records if r.get("reaction_type") == args.filter_reaction]
        print(f"  Filtered to reaction={args.filter_reaction}: {len(records):,}", flush=True)
    if args.limit > 0:
        records = records[:args.limit]
        print(f"  Limited to {len(records):,}", flush=True)

    suffix = args.output_suffix or ""
    output_path = args.output_dir / f"layer3_llm_enrichment{suffix}.jsonl"

    done_accessions: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                done_accessions.add(json.loads(line)["accession"])
        remaining = [r for r in records if r["accession"] not in done_accessions]
        print(f"  Resume: {len(done_accessions):,} done, {len(remaining):,} remaining.", flush=True)
        records = remaining

    if not records:
        print("  Nothing to process.", flush=True)
        return 0

    print(f"[2/3] Running LLM enrichment ({len(records):,} records) ...", flush=True)

    success, errors = process_batch(
        call_fn=call_fn,
        client=client,
        records=records,
        model=args.model,
        output_path=output_path,
        done_accessions=done_accessions,
        rate_limit_delay=args.rate_limit_delay,
    )

    print(f"[3/3] Done: {success:,} enriched, {errors:,} errors.", flush=True)

    # Summary
    if output_path.exists():
        total_done = 0
        conf_dist: dict[str, int] = {}
        style_dist: dict[str, int] = {}
        with output_path.open("r") as f:
            for line in f:
                obj = json.loads(line)
                total_done += 1
                conf_dist[obj.get("llm_confidence", "?")] = conf_dist.get(obj.get("llm_confidence", "?"), 0) + 1
                style_dist[obj.get("llm_active_site_style", "?")] = style_dist.get(obj.get("llm_active_site_style", "?"), 0) + 1

        summary = {
            "total_enriched": total_done,
            "model": args.model,
            "backend": args.backend,
            "confidence_distribution": dict(sorted(conf_dist.items(), key=lambda x: -x[1])),
            "active_site_style_distribution": dict(sorted(style_dist.items(), key=lambda x: -x[1])),
        }
        summary_path = args.output_dir / f"layer3_llm_summary{suffix}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
