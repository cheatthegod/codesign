#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

echo "[DISCO] Repo root: ${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[DISCO] Error: uv is not installed (or not in PATH)."
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[DISCO] UV cache dir: ${UV_CACHE_DIR}"
echo "[DISCO] Managed Python dir: ${UV_PYTHON_INSTALL_DIR}"
echo "[DISCO] XDG cache dir: ${XDG_CACHE_HOME}"

if [[ "${DISCO_SKIP_DEEPSPEED:-0}" == "1" ]]; then
  echo "[DISCO] DISCO_SKIP_DEEPSPEED=1 -> removing deepspeed dependency from pyproject.toml"
  python3 - <<'PY'
from pathlib import Path

p = Path("pyproject.toml")
line = '    "deepspeed>=0.18.3",\n'
text = p.read_text(encoding="utf-8")
if line in text:
    p.write_text(text.replace(line, ""), encoding="utf-8")
    print("Removed deepspeed dependency from pyproject.toml")
else:
    print("deepspeed dependency already absent")
PY
fi

if [[ "${DISCO_SKIP_SYNC:-0}" == "1" ]]; then
  echo "[DISCO] DISCO_SKIP_SYNC=1 -> skipping uv sync"
else
  echo "[DISCO] Running uv sync ..."
  uv sync
fi

if [[ -n "${TORCH_BACKEND:-}" ]]; then
  echo "[DISCO] Reinstalling torch with backend: ${TORCH_BACKEND}"
  uv pip uninstall torch || true
  uv pip install torch --torch-backend="${TORCH_BACKEND}"
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import importlib
import os
import sys

print(f"[DISCO] Venv python: {sys.executable}")
print(f"[DISCO] Python version: {sys.version.split()[0]}")

for module_name in ("torch", "hydra", "lightning", "rdkit", "huggingface_hub"):
    importlib.import_module(module_name)

import torch

print(f"[DISCO] torch: {torch.__version__}")
print(f"[DISCO] torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[DISCO] Visible CUDA devices: {torch.cuda.device_count()}")
else:
    print("[DISCO] CUDA is not currently available.")
    print("[DISCO] If you expect a GPU, run bash scripts/doctor_env.sh to inspect the driver/backend state.")
PY
else
  echo "[DISCO] Warning: .venv/bin/python not found after setup."
fi

echo "[DISCO] Done. Activate with: source scripts/activate_env.sh"
