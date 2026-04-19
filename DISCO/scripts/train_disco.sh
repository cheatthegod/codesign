#!/usr/bin/env bash
# DISCO Training Script
#
# Usage:
#   # Single GPU:
#   bash scripts/train_disco.sh train_data_dir=/path/to/pdb_data
#
#   # Multi-GPU (e.g. 4 GPUs):
#   bash scripts/train_disco.sh train_data_dir=/path/to/pdb_data fabric.num_nodes=1
#
#   # Resume from checkpoint:
#   bash scripts/train_disco.sh train_data_dir=/path/to/pdb_data load_checkpoint_path=/path/to/ckpt.pt
#
#   # Override hyperparameters:
#   bash scripts/train_disco.sh train_data_dir=/path/to/pdb_data optimizer.lr=0.0001 max_steps=200000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

TRAIN_ENTRY="${DISCO_TRAIN_ENTRY:-runner/train.py}"
PYTHON_BIN="${DISCO_PYTHON_BIN}"

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "[DISCO] Training entrypoint not found: ${TRAIN_ENTRY}"
  echo "[DISCO] Please ensure runner/train.py exists."
  exit 2
fi

echo "[DISCO] Starting training with entrypoint: ${TRAIN_ENTRY}"
echo "[DISCO] Python: ${PYTHON_BIN}"
echo "[DISCO] Arguments: $@"

# Detect number of GPUs for torchrun
NUM_GPUS="${DISCO_NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[DISCO] Multi-GPU training with ${NUM_GPUS} GPUs"
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-29500}" \
    "${TRAIN_ENTRY}" "$@"
else
  echo "[DISCO] Single-GPU training"
  "${PYTHON_BIN}" "${TRAIN_ENTRY}" "$@"
fi
