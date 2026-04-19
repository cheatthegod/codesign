#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

report_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "[EnzyGen2] OK      ${label}: ${path}"
  else
    echo "[EnzyGen2] MISSING ${label}: ${path}"
  fi
}

report_candidates() {
  local label="$1"
  shift

  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      echo "[EnzyGen2] OK      ${label}: ${candidate}"
      return 0
    fi
  done

  echo "[EnzyGen2] MISSING ${label}. Checked:"
  for candidate in "$@"; do
    echo "  - ${candidate}"
  done
}

report_pretraining_set() {
  local extracted="data/pdb_swissprot_data_ligand.json"
  local archive="data/pdb_swissprot_data_ligand.json.tar.gz"

  if [[ -f "${extracted}" ]]; then
    echo "[EnzyGen2] OK      Pretraining set: ${extracted}"
  elif [[ -f "${archive}" ]]; then
    echo "[EnzyGen2] PARTIAL Pretraining archive: ${archive}"
    echo "  - extract it to: ${extracted}"
  else
    echo "[EnzyGen2] MISSING Pretraining set: ${extracted}"
    echo "  - expected download or archive: ${archive}"
  fi
}

echo "[EnzyGen2] Repo root: ${REPO_ROOT}"
echo

echo "[EnzyGen2] Bundled repo-local JSON/TAR assets:"
find "${REPO_ROOT}" \
  -maxdepth 2 \
  \( -path "${REPO_ROOT}/.git" -o -path "${REPO_ROOT}/fairseq" \) -prune -o \
  -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.tar.gz' \) \
  -print | sed "s#${REPO_ROOT}/#  - #"
echo

echo "[EnzyGen2] Expected training/evaluation data under ./data:"
report_file "NCBI map" "data/ncbi2id.json"
report_pretraining_set
report_candidates "ChlR finetune set" "data/rhea_18421_final.json" "data/chloramphenicol_acetyltransferase_final.json"
report_candidates "AadA finetune set" "data/rhea_20245_final.json" "data/aminoglycoside_adenylyltransferase_final.json"
report_candidates "TPMT finetune set" "data/Thiopurine_S_methyltransferas_final.json" "data/thiopurine_methyltransferase_final.json"
report_file "Evaluation set" "data/protein_ligand_enzyme_test.json"
report_file "Evaluation mapping" "data/protein_ligand_enzyme_test_pdb2ec.json"
echo

if [[ -d "${REPO_ROOT}/data" ]]; then
  json_count="$(find "${REPO_ROOT}/data" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
  targz_count="$(find "${REPO_ROOT}/data" -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')"
  echo "[EnzyGen2] Local ./data inventory:"
  echo "  json files: ${json_count}"
  echo "  tar.gz files: ${targz_count}"
else
  echo "[EnzyGen2] Local ./data directory is missing."
fi
echo
echo "[EnzyGen2] Note: this repository ships training code, but the training datasets are expected to be downloaded separately."
