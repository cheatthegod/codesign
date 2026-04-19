#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export REPO_ROOT
export DISCO_REPO_ROOT="${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${REPO_ROOT}/.uv-python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${HUGGINGFACE_HUB_CACHE}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export DISCO_PYTHON_BIN="${DISCO_PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  export DISCO_PYTHON_BIN="${DISCO_PYTHON_BIN:-python}"
fi
