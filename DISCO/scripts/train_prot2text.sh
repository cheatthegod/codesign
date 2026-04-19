#!/usr/bin/env bash
# DISCO Training with Prot2Text Data
#
# Usage:
#   # Train on full Prot2Text dataset (236K proteins):
#   bash scripts/train_prot2text.sh
#
#   # Train on enzyme-only subset (136K enzymes):
#   bash scripts/train_prot2text.sh --config-name=train_enzyme
#
#   # Fine-tune from pretrained checkpoint:
#   bash scripts/train_prot2text.sh load_checkpoint_path=pretrained/DISCO.pt
#
#   # Override parameters:
#   bash scripts/train_prot2text.sh max_steps=100000 optimizer.lr=0.0001
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

TRAIN_ENTRY="${DISCO_TRAIN_ENTRY:-runner/train.py}"
PYTHON_BIN="${DISCO_PYTHON_BIN}"

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "[DISCO] Training entrypoint not found: ${TRAIN_ENTRY}"
  exit 2
fi

# Check if Prot2Text data exists
PROT2TEXT_DIR="/home/ubuntu/cqr_files/protein_design/COT_enzyme_design/Prot2Text-Data"
if [[ ! -d "${PROT2TEXT_DIR}" ]]; then
  echo "[DISCO] Prot2Text data not found at: ${PROT2TEXT_DIR}"
  echo "[DISCO] Please ensure Prot2Text-Data directory exists with:"
  echo "  - rfd3_monomer_64_1024/train.csv"
  echo "  - alphafold_structures/pdb/"
  exit 2
fi

# Default config
CONFIG_NAME="train_prot2text"

# Check if user specified a different config
for arg in "$@"; do
  if [[ "${arg}" == --config-name=* ]]; then
    CONFIG_NAME="${arg#--config-name=}"
  fi
done

echo "[DISCO] Training with Prot2Text data"
echo "[DISCO] Config: ${CONFIG_NAME}"
echo "[DISCO] Data: ${PROT2TEXT_DIR}"

# Detect GPUs
NUM_GPUS="${DISCO_NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

# Filter out --config-name from $@ if present (we pass it via Hydra)
ARGS=()
for arg in "$@"; do
  if [[ "${arg}" != --config-name=* ]]; then
    ARGS+=("${arg}")
  fi
done

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[DISCO] Multi-GPU training with ${NUM_GPUS} GPUs"
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-29500}" \
    "${TRAIN_ENTRY}" \
    --config-name="${CONFIG_NAME}" \
    "${ARGS[@]}"
else
  echo "[DISCO] Single-GPU training"
  "${PYTHON_BIN}" "${TRAIN_ENTRY}" --config-name="${CONFIG_NAME}" "${ARGS[@]}"
fi
