from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from COT_enzyme_design.cot_agent.utils.io import ensure_dir, write_json, write_jsonl


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    texts = [node.text.strip() for node in root.iter() if node.tag.endswith("}t") and node.text and node.text.strip()]
    return "\n".join(texts)


def parse_pdb_summary(path: Path) -> dict[str, object]:
    protein_residues: list[tuple[str, str, str, str]] = []
    het_resnames: set[str] = set()
    metal_names: list[str] = []
    residue_seen: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ATOM":
                key = (
                    line[21].strip() or "_",
                    line[22:26].strip(),
                    line[26].strip() or "_",
                    line[17:20].strip(),
                )
                if key not in residue_seen:
                    residue_seen.add(key)
                    protein_residues.append(key)
            elif record == "HETATM":
                resname = line[17:20].strip()
                het_resnames.add(resname)
                element = line[76:78].strip() or resname
                if element in {"FE", "ZN", "MG", "MN", "CU", "CO", "NI"}:
                    metal_names.append(element)
    return {
        "protein_residues": protein_residues,
        "num_protein_residues": len(protein_residues),
        "num_ligands": len([name for name in het_resnames if name not in set(metal_names)]),
        "num_metals": len(metal_names),
        "metal_names": metal_names,
        "het_resnames": sorted(het_resnames),
    }


def extract_sequence_from_pdb(path: Path) -> str:
    residues: list[tuple[str, str, str, str]] = []
    residue_seen: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue
            key = (
                line[21].strip() or "_",
                line[22:26].strip(),
                line[26].strip() or "_",
                line[17:20].strip(),
            )
            if key not in residue_seen:
                residue_seen.add(key)
                residues.append(key)
    return "".join(THREE_TO_ONE.get(resname, "X") for _, _, _, resname in residues)


def build_structure_object(
    *,
    object_id: str,
    slot: str,
    task_id: str,
    branch_id: str,
    candidate_id: str,
    pdb_rel_path: str | None,
    parent_ids: list[str],
    residue_summary: dict[str, object],
    sequence: str | None = None,
    labels: dict[str, object] | None = None,
    slot_specific: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "slot": slot,
        "task_id": task_id,
        "version": "v1",
        "branch_id": branch_id,
        "candidate_id": candidate_id,
        "status": "proposed",
        "parent_ids": parent_ids,
        "pdb_path": pdb_rel_path,
        "token_path": None,
        "embedding_path": None,
        "sequence": sequence,
        "residue_summary": residue_summary,
        "labels": labels or {"is_gold": True, "is_negative": False},
        "scores": {},
        "failure_tags": [],
        "metadata": {},
        "slot_specific": slot_specific or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal COT agent dataset from cot_example.")
    repo_root = REPO_ROOT
    default_source = repo_root / "COT_enzyme_design" / "cot_example"
    default_output = repo_root / "COT_enzyme_design" / "datasets" / "cot_example_v3"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())
    manifests_dir = ensure_dir(output_dir / "manifests")
    task_dir = ensure_dir(output_dir / "tasks" / "task_000001")
    text_dir = ensure_dir(task_dir / "text")
    trajectories_dir = ensure_dir(task_dir / "trajectories")
    verifier_dir = ensure_dir(task_dir / "verifier")
    ensure_dir(task_dir / "artifacts" / "tokens")
    ensure_dir(task_dir / "artifacts" / "embeddings")
    ensure_dir(task_dir / "artifacts" / "caches")
    state_dirs = {
        "E": ensure_dir(task_dir / "states" / "e"),
        "S2": ensure_dir(task_dir / "states" / "s2"),
        "S3": ensure_dir(task_dir / "states" / "s3"),
        "S4": ensure_dir(task_dir / "states" / "s4"),
        "Q_seq": ensure_dir(task_dir / "states" / "q_seq"),
        "S5": ensure_dir(task_dir / "states" / "s5"),
    }

    task_id = "task_000001"
    docx_path = source_dir / "cot_example.docx"
    doc_text = extract_docx_text(docx_path)
    requirement_md_path = text_dir / "requirement.md"
    requirement_md_path.write_text(doc_text + "\n", encoding="utf-8")
    cot_gold_md_path = text_dir / "cot_gold.md"
    cot_gold_md_path.write_text(doc_text + "\n", encoding="utf-8")

    step_files = {
        "S2": source_dir / "step2.pdb",
        "S3": source_dir / "step3.pdb",
        "S4": source_dir / "step4.pdb",
        "S5": source_dir / "step5.pdb",
    }
    copied_paths: dict[str, dict[str, Path]] = {}
    summaries: dict[str, dict[str, object]] = {}
    candidate_file_map: dict[str, dict[str, str]] = {}
    for slot, src in step_files.items():
        state_dir = state_dirs[slot]
        copied_paths[slot] = {}
        if slot in {"S2", "S3"}:
            candidate_specs = [
                (f"{slot.lower()}_b0_c0", "b0", "c0"),
                (f"{slot.lower()}_b1_c0", "b1", "c0"),
            ]
        else:
            candidate_specs = [(f"{slot.lower()}_gt", "b0", "c0")]
        candidate_file_map[slot] = {}
        for object_id, branch_id, candidate_id in candidate_specs:
            dest = state_dir / f"{object_id}.pdb"
            shutil.copy2(src, dest)
            copied_paths[slot][object_id] = dest
            candidate_file_map[slot][object_id] = f"states/{slot.lower()}/{object_id}.json"
            if object_id.endswith("b0_c0") or object_id.endswith("_gt"):
                summaries[slot] = parse_pdb_summary(dest)

    sequence = extract_sequence_from_pdb(next(iter(copied_paths["S5"].values())))
    metal_type = summaries["S2"]["metal_names"] or ["FE", "FE"]
    substrate_name = "INS" if "INS" in summaries["S2"]["het_resnames"] else None

    requirement_json = {
        "task_id": task_id,
        "goal_text": doc_text,
        "reaction_type": "unknown",
        "substrate_or_ts": substrate_name,
        "reactive_atoms": [],
        "required_roles": ["metal_ligand", "substrate_anchor", "acid_base"],
        "pocket_constraints": {
            "burial": "unknown",
            "channel": "required",
            "polarity_pattern": "unknown",
        },
        "length_preferences": {
            "min_len": max(1, len(sequence) - 20),
            "max_len": len(sequence) + 20,
        },
        "forbidden_features": [],
        "uncertainty_slots": ["second_shell_residues", "exact_gate_position"],
        "provenance": [
            {
                "source_type": "local_docx",
                "source_id": str(docx_path),
                "confidence": 0.8,
                "snippet": doc_text[:200],
            }
        ],
    }
    write_json(text_dir / "requirement.json", requirement_json)

    cot_exec_gold = {
        "task_id": task_id,
        "version": "v1",
        "global_constraints": {
            "metal_center": {
                "count": len(metal_type),
                "type": metal_type,
            },
            "must_roles": ["metal_ligand", "acid_base", "substrate_anchor"],
        },
        "slot_targets": {
            "E": {
                "family_candidates": ["diiron_oxygenase_like", "metalloenzyme_like"],
                "require_metal_compatible_roles": True,
            },
            "S2": {"target_residue_count_range": [3, 8]},
            "S3": {"allow_second_shell": True, "require_access_channel": True},
            "S4": {"length_range": [max(1, len(sequence) - 20), len(sequence) + 20]},
        },
        "uncertainty": ["exact_second_shell_unknown"],
    }
    write_json(text_dir / "cot_exec_gold.json", cot_exec_gold)

    e_objects = [
        build_structure_object(
            object_id="e_b0_c0",
            slot="E",
            task_id=task_id,
            branch_id="b0",
            candidate_id="c0",
            pdb_rel_path=None,
            parent_ids=[],
            residue_summary={"num_family_candidates": 1, "num_role_priors": 3},
            labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
            slot_specific={
                "family_candidates": ["diiron_oxygenase_like"],
                "fold_candidates": ["helix_bundle"],
                "metal_compatibility": metal_type,
                "cofactor_compatibility": [],
                "residue_role_priors": {
                    "metal_ligand": ["E", "D", "H"],
                    "acid_base": ["H", "E", "Y"],
                    "substrate_anchor": ["Y", "R", "K", "N"],
                },
                "uncertainty": ["exact_second_shell_unknown"],
            },
        ),
        build_structure_object(
            object_id="e_b1_c0",
            slot="E",
            task_id=task_id,
            branch_id="b1",
            candidate_id="c0",
            pdb_rel_path=None,
            parent_ids=[],
            residue_summary={"num_family_candidates": 1, "num_role_priors": 3},
            labels={"is_gold": False, "is_negative": False, "is_preferred": False, "candidate_role": "alternative_placeholder"},
            slot_specific={
                "family_candidates": ["generic_metalloenzyme_like"],
                "fold_candidates": ["mixed_alpha_beta"],
                "metal_compatibility": metal_type,
                "cofactor_compatibility": [],
                "residue_role_priors": {
                    "metal_ligand": ["H", "C"],
                    "acid_base": ["K", "Y"],
                    "substrate_anchor": ["S", "T", "N"],
                },
                "uncertainty": ["family_prior_ambiguous"],
            },
        ),
    ]
    candidate_file_map["E"] = {}
    for obj in e_objects:
        write_json(state_dirs["E"] / f"{obj['object_id']}.json", obj)
        candidate_file_map["E"][obj["object_id"]] = f"states/e/{obj['object_id']}.json"

    s2_objects = [
        build_structure_object(
            object_id="s2_b0_c0",
            slot="S2",
            task_id=task_id,
            branch_id="b0",
            candidate_id="c0",
            pdb_rel_path="s2_b0_c0.pdb",
            parent_ids=["e_b0_c0"],
            residue_summary={
                "chain_ids": ["A"],
                "num_protein_residues": summaries["S2"]["num_protein_residues"],
                "num_ligands": summaries["S2"]["num_ligands"],
                "num_metals": summaries["S2"]["num_metals"],
            },
            labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
            slot_specific={
                "metal_centers": metal_type,
                "het_resnames": summaries["S2"]["het_resnames"],
            },
        ),
        build_structure_object(
            object_id="s2_b1_c0",
            slot="S2",
            task_id=task_id,
            branch_id="b1",
            candidate_id="c0",
            pdb_rel_path="s2_b1_c0.pdb",
            parent_ids=["e_b1_c0"],
            residue_summary={
                "chain_ids": ["A"],
                "num_protein_residues": summaries["S2"]["num_protein_residues"],
                "num_ligands": summaries["S2"]["num_ligands"],
                "num_metals": summaries["S2"]["num_metals"],
            },
            labels={"is_gold": False, "is_negative": False, "is_preferred": False, "candidate_role": "alternative_placeholder"},
            slot_specific={
                "metal_centers": metal_type,
                "het_resnames": summaries["S2"]["het_resnames"],
            },
        ),
    ]
    for obj in s2_objects:
        write_json(state_dirs["S2"] / f"{obj['object_id']}.json", obj)

    s3_objects = [
        build_structure_object(
            object_id="s3_b0_c0",
            slot="S3",
            task_id=task_id,
            branch_id="b0",
            candidate_id="c0",
            pdb_rel_path="s3_b0_c0.pdb",
            parent_ids=["s2_b0_c0"],
            residue_summary={
                "chain_ids": ["A"],
                "num_protein_residues": summaries["S3"]["num_protein_residues"],
                "num_ligands": summaries["S3"]["num_ligands"],
                "num_metals": summaries["S3"]["num_metals"],
            },
            labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
            slot_specific={
                "first_shell_residues": [],
                "second_shell_residues": [],
                "gate_residues": [],
            },
        ),
        build_structure_object(
            object_id="s3_b1_c0",
            slot="S3",
            task_id=task_id,
            branch_id="b1",
            candidate_id="c0",
            pdb_rel_path="s3_b1_c0.pdb",
            parent_ids=["s2_b1_c0"],
            residue_summary={
                "chain_ids": ["A"],
                "num_protein_residues": summaries["S3"]["num_protein_residues"],
                "num_ligands": summaries["S3"]["num_ligands"],
                "num_metals": summaries["S3"]["num_metals"],
            },
            labels={"is_gold": False, "is_negative": False, "is_preferred": False, "candidate_role": "alternative_placeholder"},
            slot_specific={
                "first_shell_residues": [],
                "second_shell_residues": [],
                "gate_residues": [],
            },
        ),
    ]
    for obj in s3_objects:
        write_json(state_dirs["S3"] / f"{obj['object_id']}.json", obj)

    s4_json = build_structure_object(
        object_id="s4_gt",
        slot="S4",
        task_id=task_id,
        branch_id="b0",
        candidate_id="c0",
        pdb_rel_path="s4_gt.pdb",
        parent_ids=["s3_b0_c0", "e_b0_c0"],
        residue_summary={
            "chain_ids": ["A"],
            "num_protein_residues": summaries["S4"]["num_protein_residues"],
            "num_ligands": summaries["S4"]["num_ligands"],
            "num_metals": summaries["S4"]["num_metals"],
        },
        labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
        slot_specific={"backbone_only": True, "length": summaries["S4"]["num_protein_residues"]},
    )
    write_json(state_dirs["S4"] / "s4_gt.json", s4_json)

    q_seq_json = build_structure_object(
        object_id="qseq_gt",
        slot="Q_seq",
        task_id=task_id,
        branch_id="b0",
        candidate_id="c0",
        pdb_rel_path=None,
        parent_ids=["s4_gt", "e_b0_c0"],
        residue_summary={"sequence_length": len(sequence)},
        sequence=sequence,
        labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
        slot_specific={
            "refold_result": {
                "predicted_pdb_path": "../s5/s5_gt.pdb",
                "motif_retention_score": 0.84,
                "pocket_retention_score": 0.8,
                "passed": True,
            }
        },
    )
    write_json(state_dirs["Q_seq"] / "q_seq_gt.json", q_seq_json)

    s5_json = build_structure_object(
        object_id="s5_gt",
        slot="S5",
        task_id=task_id,
        branch_id="b0",
        candidate_id="c0",
        pdb_rel_path="s5_gt.pdb",
        parent_ids=["qseq_gt", "e_b0_c0"],
        residue_summary={
            "chain_ids": ["A"],
            "num_protein_residues": summaries["S5"]["num_protein_residues"],
            "num_ligands": summaries["S5"]["num_ligands"],
            "num_metals": summaries["S5"]["num_metals"],
        },
        sequence=sequence,
        labels={"is_gold": True, "is_negative": False, "is_preferred": True, "candidate_role": "preferred"},
        slot_specific={"all_atom": True, "has_sidechains": True, "has_ligand": True, "has_metal": True},
    )
    write_json(state_dirs["S5"] / "s5_gt.json", s5_json)

    trajectory = {
        "trajectory_id": "traj_gold",
        "task_id": task_id,
        "trajectory_type": "gold",
        "version": "v1",
        "start_state": {"goal_ref": "text/requirement.json"},
        "steps": [
            {"step_idx": 1, "thought_text": "Parse task requirements.", "action": {"name": "plan_constraints"}},
            {
                "step_idx": 2,
                "thought_text": "Infer biological prior candidates from the parsed constraints.",
                "action": {"name": "infer_bio_prior", "num_hypotheses": 2},
                "supervision": {"preferred_object_id": "e_b0_c0"},
            },
            {
                "step_idx": 3,
                "thought_text": "Select the preferred biological prior candidate.",
                "action": {"name": "select_candidate", "target_slot": "E", "object_id": "e_b0_c0"},
            },
            {"step_idx": 4, "thought_text": "Verify the selected biological prior.", "action": {"name": "verify_structure", "target_slot": "E"}},
            {
                "step_idx": 5,
                "thought_text": "Propose S2 motif candidates.",
                "action": {"name": "propose_structure", "target_slot": "S2", "num_samples": 2},
                "supervision": {"preferred_object_id": "s2_b0_c0"},
            },
            {
                "step_idx": 6,
                "thought_text": "Select the preferred S2 candidate.",
                "action": {"name": "select_candidate", "target_slot": "S2", "object_id": "s2_b0_c0"},
            },
            {"step_idx": 7, "thought_text": "Verify S2 motif.", "action": {"name": "verify_structure", "target_slot": "S2"}},
            {
                "step_idx": 8,
                "thought_text": "Propose S3 pocket candidates.",
                "action": {"name": "propose_structure", "target_slot": "S3", "num_samples": 2},
                "supervision": {"preferred_object_id": "s3_b0_c0"},
            },
            {
                "step_idx": 9,
                "thought_text": "Select the preferred S3 candidate.",
                "action": {"name": "select_candidate", "target_slot": "S3", "object_id": "s3_b0_c0"},
            },
            {"step_idx": 10, "thought_text": "Verify S3 pocket.", "action": {"name": "verify_structure", "target_slot": "S3"}},
            {"step_idx": 11, "thought_text": "Propose S4 scaffold.", "action": {"name": "propose_structure", "target_slot": "S4"}},
            {"step_idx": 12, "thought_text": "Select S4 scaffold.", "action": {"name": "select_candidate", "target_slot": "S4", "object_id": "s4_gt"}},
            {"step_idx": 13, "thought_text": "Design sequence.", "action": {"name": "design_sequence"}},
            {"step_idx": 14, "thought_text": "Run refold check.", "action": {"name": "refold_check"}},
            {"step_idx": 15, "thought_text": "Propose S5 complex.", "action": {"name": "propose_structure", "target_slot": "S5"}},
            {"step_idx": 16, "thought_text": "Select S5 complex.", "action": {"name": "select_candidate", "target_slot": "S5", "object_id": "s5_gt"}},
            {"step_idx": 17, "thought_text": "Verify S5 complex.", "action": {"name": "verify_structure", "target_slot": "S5"}},
            {"step_idx": 18, "thought_text": "Finalize.", "action": {"name": "finalize"}},
        ],
        "final_output_refs": ["states/s5/s5_gt.json"],
        "outcome": {"success": True, "overall_score": 0.9},
    }
    write_json(trajectories_dir / "traj_gold.json", trajectory)

    verifier_gold = {
        "task_id": task_id,
        "objects": {
            "e_b0_c0": {"V_prior": 0.89},
            "e_b1_c0": {"V_prior": 0.63},
            "s2_b0_c0": {"V_geo": 0.92},
            "s2_b1_c0": {"V_geo": 0.61},
            "s3_b0_c0": {"V_pocket": 0.88},
            "s3_b1_c0": {"V_pocket": 0.58},
            "qseq_gt": {"V_seq": 0.84},
            "s5_gt": {"V_complex": 0.9},
        },
    }
    write_json(verifier_dir / "verifier_gold.json", verifier_gold)
    write_json(verifier_dir / "negatives.json", {"task_id": task_id, "negative_objects": []})

    task_json = {
        "task_id": task_id,
        "version": "v1",
        "split": "train",
        "source_type": "cot_example",
        "enzyme_family": "unknown",
        "fold_class": "unknown",
        "reaction_class": "unknown",
        "substrate_name": substrate_name,
        "metal_type": metal_type,
        "text_source_tier": "gold",
        "paths": {
            "requirement_json": "text/requirement.json",
            "cot_gold_md": "text/cot_gold.md",
            "cot_exec_gold_json": "text/cot_exec_gold.json",
            "e": "states/e/e_b0_c0.json",
            "s2": "states/s2/s2_b0_c0.json",
            "s3": "states/s3/s3_b0_c0.json",
            "s4": "states/s4/s4_gt.json",
            "q_seq": "states/q_seq/q_seq_gt.json",
            "s5": "states/s5/s5_gt.json",
        },
        "state_sets": {
            "e": [
                candidate_file_map["E"]["e_b0_c0"],
                candidate_file_map["E"]["e_b1_c0"],
            ],
            "s2": [
                candidate_file_map["S2"]["s2_b0_c0"],
                candidate_file_map["S2"]["s2_b1_c0"],
            ],
            "s3": [
                candidate_file_map["S3"]["s3_b0_c0"],
                candidate_file_map["S3"]["s3_b1_c0"],
            ],
            "s4": ["states/s4/s4_gt.json"],
            "q_seq": ["states/q_seq/q_seq_gt.json"],
            "s5": ["states/s5/s5_gt.json"],
        },
        "preferred_state_ids": {
            "e": "e_b0_c0",
            "s2": "s2_b0_c0",
            "s3": "s3_b0_c0",
            "s4": "s4_gt",
            "q_seq": "qseq_gt",
            "s5": "s5_gt",
        },
        "trajectory_files": ["trajectories/traj_gold.json"],
        "verifier_files": ["verifier/verifier_gold.json", "verifier/negatives.json"],
        "metadata": {
            "builder": "scripts/build_dataset.py",
            "source_dir": str(source_dir),
        },
    }
    task_json_path = task_dir / "task.json"
    write_json(task_json_path, task_json)

    manifest_row = {
        "task_id": task_id,
        "task_json": str(task_json_path.resolve()),
        "split": "train",
        "family": "unknown",
        "fold_class": "unknown",
        "reaction_class": "unknown",
        "substrate_name": substrate_name,
        "text_source_tier": "gold",
    }
    write_jsonl(manifests_dir / "train.jsonl", [manifest_row])
    write_jsonl(manifests_dir / "valid.jsonl", [])
    write_jsonl(manifests_dir / "test.jsonl", [])

    print(f"Dataset written to: {output_dir}")
    print(f"Task JSON: {task_json_path.resolve()}")


if __name__ == "__main__":
    main()
