#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

status=0
TRAIN_ENTRY="${DISCO_TRAIN_ENTRY:-runner/train.py}"

echo "[DISCO] Repo root: ${REPO_ROOT}"
echo "[DISCO] System python: $(command -v python3 || echo missing)"
python3 --version 2>/dev/null || true
echo

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "[DISCO] Venv python: ${REPO_ROOT}/.venv/bin/python"
  "${REPO_ROOT}/.venv/bin/python" -V
  "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import importlib
import sys

modules = ("torch", "hydra", "lightning", "rdkit", "huggingface_hub")

print("[DISCO] Import check:")
for module_name in modules:
    try:
        importlib.import_module(module_name)
        print(f"  OK  {module_name}")
    except Exception as exc:
        print(f"  BAD {module_name}: {exc}")

try:
    import torch

    print(f"[DISCO] torch: {torch.__version__}")
    print(f"[DISCO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[DISCO] torch.cuda.device_count(): {torch.cuda.device_count()}")
except Exception as exc:
    print(f"[DISCO] torch probe failed: {exc}")
PY
else
  echo "[DISCO] Missing ${REPO_ROOT}/.venv/bin/python"
  status=1
fi
echo

echo "[DISCO] NVIDIA driver probe:"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    status=1
  fi
else
  echo "  nvidia-smi not found"
  status=1
fi
echo

echo "[DISCO] Asset probe:"
for asset in \
  "pretrained/DISCO.pt" \
  "pretrained/components.v20240608.cif" \
  "pretrained/components.v20240608.cif.rdkit_mol.pkl"; do
  if [[ -e "${REPO_ROOT}/${asset}" ]]; then
    echo "  OK  ${asset}"
  else
    echo "  BAD ${asset}"
    status=1
  fi
done
echo

echo "[DISCO] CUTLASS_PATH: ${CUTLASS_PATH:-<unset>}"
echo

echo "[DISCO] Training entrypoint probe:"
if [[ -f "${REPO_ROOT}/${TRAIN_ENTRY}" ]]; then
  echo "  OK  ${TRAIN_ENTRY}"
else
  echo "  MISSING ${TRAIN_ENTRY}"
  echo "  Public repo currently ships inference only; training code is not included."
fi
echo

input_json_count="$(find "${REPO_ROOT}/input_jsons" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
studio_sdf_count="$(find "${REPO_ROOT}/studio-179" -type f -name '*.sdf' | wc -l | tr -d ' ')"

mapfile -t training_files < <(
  find "${REPO_ROOT}" \
    \( \
      -path "${REPO_ROOT}/.git" -o \
      -path "${REPO_ROOT}/.venv" -o \
      -path "${REPO_ROOT}/.uv-cache" -o \
      -path "${REPO_ROOT}/.uv-python" -o \
      -path "${REPO_ROOT}/input_jsons" -o \
      -path "${REPO_ROOT}/pretrained" -o \
      -path "${REPO_ROOT}/studio-179" \
    \) -prune -o \
    -type f \
    \( \
      -iname '*.csv' -o \
      -iname '*.tsv' -o \
      -iname '*.jsonl' -o \
      -iname '*.parquet' -o \
      -iname '*.h5' -o \
      -iname '*.hdf5' -o \
      -iname '*.lmdb' -o \
      -iname '*.npz' -o \
      -iname '*.npy' \
    \) -print
)

echo "[DISCO] Bundled data overview:"
echo "  input_jsons/*.json: ${input_json_count}"
echo "  studio-179/*.sdf: ${studio_sdf_count}"
if [[ "${#training_files[@]}" -eq 0 ]]; then
  echo "  Public training dataset files: none found"
else
  echo "  Training-like files outside inference assets:"
  printf '    %s\n' "${training_files[@]}"
fi
echo

if [[ "${status}" -eq 0 ]]; then
  echo "[DISCO] Environment summary: basic files are present."
else
  echo "[DISCO] Environment summary: not ready for GPU inference/training yet."
fi
