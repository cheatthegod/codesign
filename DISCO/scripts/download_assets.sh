#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

ASSET_DIR="${DISCO_ASSET_DIR:-${REPO_ROOT}/pretrained}"
mkdir -p "${ASSET_DIR}"

echo "[DISCO] Downloading checkpoint and CCD assets from Hugging Face ..."
echo "[DISCO] Local asset dir: ${ASSET_DIR}"
"${DISCO_PYTHON_BIN}" - "${ASSET_DIR}" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

asset_dir = os.path.abspath(sys.argv[1])

assets = [
    ("DISCO-Design/DISCO", "DISCO.pt"),
    ("DISCO-Design/DISCO", "components.v20240608.cif"),
    ("DISCO-Design/DISCO", "components.v20240608.cif.rdkit_mol.pkl"),
]

for repo_id, filename in assets:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=asset_dir,
    )
    print(f"downloaded {filename} -> {path}")
PY

echo "[DISCO] Asset download complete."
