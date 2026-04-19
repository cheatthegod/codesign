#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

INPUT_JSON="${1:-input_jsons/unconditional_config.json}"
shift || true

EXPERIMENT="${DISCO_EXPERIMENT:-designable}"
EFFORT="${DISCO_EFFORT:-max}"
SEEDS="${DISCO_SEEDS:-[0,1,2,3,4]}"
CHECKPOINT="${DISCO_CHECKPOINT:-${REPO_ROOT}/pretrained/DISCO.pt}"
PYTHON_BIN="${DISCO_PYTHON_BIN}"

hydra_args=(
  "experiment=${EXPERIMENT}"
  "effort=${EFFORT}"
  "input_json_path=${INPUT_JSON}"
  "seeds=${SEEDS}"
)

if [[ -f "${CHECKPOINT}" ]]; then
  echo "[DISCO] Using local checkpoint: ${CHECKPOINT}"
  hydra_args+=("load_checkpoint_path=${CHECKPOINT}")
else
  echo "[DISCO] Local checkpoint not found at ${CHECKPOINT}; inference.py will auto-download from HF."
fi

if ! "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "[DISCO] Warning: torch.cuda.is_available() is false."
  echo "[DISCO] DISCO inference expects a working CUDA stack and may fail until the NVIDIA driver is visible."
fi

"${PYTHON_BIN}" runner/inference.py "${hydra_args[@]}" "$@"
