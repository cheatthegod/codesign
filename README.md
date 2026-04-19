# CoDesign: Protein-Ligand Co-Design

Protein sequence-structure co-design models for enzyme engineering.

## Models

### DISCO
Diffusion-based Structure and Sequence Co-design model. Based on AlphaFold3 architecture with three key innovations:
- JointDiffusionModule for simultaneous sequence and structure prediction
- Frozen DPLM-650M language model for evolutionary priors
- Recycling mechanism with LigandMPNN structure encoding

### EnzyGen2
ESM2 + EGNN interleaved architecture for enzyme design:
- 33-layer ESM2 Transformer with 3 EGNN layers interleaved
- NCBI taxonomy embedding for species-aware design
- SubstrateEGNN for ligand binding prediction
- Three-stage training: MLM → Motif → Full

## Documentation

- `docs/DISCO_v2_training_report.md` - DISCO training results
- `docs/DISCO_训练全流程详解.md` - DISCO training guide (Chinese)
- `docs/RFD3_training_report.md` - RFD3 training results
- `docs/RFD3_训练全流程详解.md` - RFD3 training guide (Chinese)
- `DISCO/DISCO_完全初学者指南.md` - DISCO complete beginner's guide (9400+ lines)
- `EnzyGen2/EnzyGen2_BEGINNER_COMPLETE_CN.md` - EnzyGen2 complete beginner's guide (8000+ lines)

## Data Processing Scripts

The `scripts/` directory contains the Prot2Text enrichment pipeline:
- `build_enriched_base.py` - Layer 1: enzyme annotation enrichment
- `compute_structure_features.py` - Layer 2: structural feature extraction
- `generate_route_a_constraints.py` - Layer 3: constraint extraction
- `build_training_subsets.py` - Generate focused training subsets
- `build_conditioned_csvs.py` - Build metadata-conditioned CSVs
- `export_prot2text_rfd3_dataset.py` - Export RFD3-ready datasets
