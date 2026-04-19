#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[DISCO] This script must be sourced."
  echo "[DISCO] Run: source scripts/activate_env.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  echo "[DISCO] Missing virtual environment at ${REPO_ROOT}/.venv"
  echo "[DISCO] Run: bash scripts/setup_env.sh"
  return 1
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

echo "[DISCO] Activated virtual environment: ${VIRTUAL_ENV}"
echo "[DISCO] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "[DISCO] MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "[DISCO] HF_HOME=${HF_HOME}"
