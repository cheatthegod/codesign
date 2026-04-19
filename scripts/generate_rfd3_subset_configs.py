#!/usr/bin/env python3
"""Generate RFD3 dataset + experiment YAML configs for each training subset.

Reads training_subsets/ and produces one dataset YAML + one experiment YAML
per subset, ready to use with RFD3's hydra config system.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SUBSETS_DIR = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "training_subsets"
DATASETS_DIR = PROJECT_DIR / "foundry" / "models" / "rfd3" / "configs" / "datasets"
EXPERIMENTS_DIR = PROJECT_DIR / "foundry" / "models" / "rfd3" / "configs" / "experiment"

# Template for dataset YAML
DATASET_TEMPLATE = """defaults:
  - train/pdb/base_no_weights@train.{safe_name}
  - conditions/unconditional@global_transform_args.train_conditions.unconditional
  - conditions/island@global_transform_args.train_conditions.island
  - conditions/tipatom@global_transform_args.train_conditions.tipatom
  - conditions/sequence_design@global_transform_args.train_conditions.sequence_design
  - conditions/ppi@global_transform_args.train_conditions.ppi
  - _self_

# {description}
# Train size: {train_size:,}

pipeline_target: rfd3.transforms.pipelines.build_atom14_base_pipeline

diffusion_batch_size_train: 4
diffusion_batch_size_inference: 1
crop_size: 256
n_recycles_train: 1
n_recycles_validation: 1
max_atoms_in_crop: 2560

global_transform_args:
  n_atoms_per_token: 14
  central_atom: CB
  sigma_perturb: 2.0
  sigma_perturb_com: 1.0
  association_scheme: dense
  center_option: diffuse

  generate_conformers: False
  generate_conformers_for_non_protein_only: True
  provide_reference_conformer_when_unmasked: True
  ground_truth_conformer_policy: IGNORE
  provide_elements_for_unindexed_components: True
  use_element_for_atom_names_of_atomized_tokens: True

  keep_full_binder_in_spatial_crop: False
  max_binder_length: 170
  max_ppi_hotspots_frac_to_provide: 0.0
  ppi_hotspot_max_distance: 4.5

  max_ss_frac_to_provide: 0.0
  min_ss_island_len: 1
  max_ss_island_len: 6

  train_conditions:
    unconditional:
      frequency: 1.0
    sequence_design:
      frequency: 0.25
    island:
      frequency: 0.25
    tipatom:
      frequency: 0.0
    ppi:
      frequency: 0.0

  meta_conditioning_probabilities:
    calculate_hbonds: 0.0
    calculate_rasa: {rasa_prob}
    keep_protein_motif_rasa: 0.0
    hbond_subsample: 0.0
    unindex_leak_global_index: 0.0
    unindex_insert_random_break: 0.0
    unindex_remove_random_break: 0.0
    add_1d_ss_features: 0.0
    featurize_plddt: 0.9
    add_global_is_non_loopy_feature: {non_loopy_prob}
    add_ppi_hotspots: 0.0
    full_binder_crop: 0.0

train:
  {safe_name}:
    probability: 1.0
    dataset:
      dataset_parser:
        _target_: atomworks.ml.datasets.parsers.GenericDFParser
        base_path: ${{paths.root_dir}}
        pn_unit_iid_colnames: null
      dataset:
        name: {safe_name}
        data: {train_csv}
        columns_to_load:
          - example_id
          - path
      transform:
        crop_contiguous_probability: 0.5
        crop_spatial_probability: 0.5
        b_factor_min: 70

val: null
"""

EXPERIMENT_TEMPLATE = """# @package _global_

defaults:
  - override /datasets: {config_name}
  - override /paths: local_smoke
  - override /logger: csv
  - _self_

# {description}

name: rfd3-{config_name}
project: rfd3-prot2text
tags:
  - prot2text
  - enzyme
  - {tag}
  - enriched
ckpt_path: null

trainer:
  devices_per_node: 1
  num_nodes: 1
  precision: bf16-mixed
  grad_accum_steps: 4
  checkpoint_every_n_epochs: 1
  validate_every_n_epochs: 1000
  prevalidate: False

dataloader:
  train:
    dataloader_params:
      num_workers: 2
      pin_memory: True
      prefetch_factor: 2
    n_fallback_retries: 8
  val:
    dataloader_params:
      num_workers: 1
      pin_memory: True
      prefetch_factor: 2
    n_fallback_retries: 0

datasets:
  val: null
"""


def run() -> int:
    if not SUBSETS_DIR.exists():
        print(f"No subsets directory found at {SUBSETS_DIR}")
        return 1

    subset_dirs = sorted([d for d in SUBSETS_DIR.iterdir() if d.is_dir() and (d / "train.csv").exists()])
    print(f"Found {len(subset_dirs)} subsets.", flush=True)

    generated = []
    for subset_dir in subset_dirs:
        name = subset_dir.name
        safe_name = name.replace("-", "_")
        config_name = f"prot2text_{safe_name}"
        train_csv = str(subset_dir / "train.csv")

        # Load metadata
        meta_path = subset_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        train_size = meta.get("train_size", 0)
        description = meta.get("description", name)

        if train_size < 100:
            print(f"  Skipping {name} (only {train_size} train samples)", flush=True)
            continue

        # Choose conditioning probabilities based on subset type
        rasa_prob = 0.5
        non_loopy_prob = 0.5
        if "high_quality" in name:
            rasa_prob = 0.5
            non_loopy_prob = 0.5
        elif "pocket_deeply_buried" in name:
            rasa_prob = 0.7  # emphasize burial conditioning
            non_loopy_prob = 0.5

        # Write dataset YAML
        dataset_yaml = DATASET_TEMPLATE.format(
            safe_name=safe_name,
            description=description,
            train_size=train_size,
            train_csv=train_csv,
            rasa_prob=rasa_prob,
            non_loopy_prob=non_loopy_prob,
        )
        dataset_path = DATASETS_DIR / f"{config_name}.yaml"
        dataset_path.write_text(dataset_yaml, encoding="utf-8")

        # Write experiment YAML
        tag = name.split("_")[0] if "_" in name else name
        experiment_yaml = EXPERIMENT_TEMPLATE.format(
            config_name=config_name,
            description=description,
            tag=tag,
        )
        experiment_path = EXPERIMENTS_DIR / f"{config_name}_bootstrap.yaml"
        experiment_path.write_text(experiment_yaml, encoding="utf-8")

        print(f"  {name}: {train_size:,} train -> {dataset_path.name} + {experiment_path.name}", flush=True)
        generated.append({"name": name, "config": config_name, "train_size": train_size})

    # Summary
    summary_path = SUBSETS_DIR / "generated_configs.json"
    summary_path.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGenerated {len(generated)} config pairs.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
