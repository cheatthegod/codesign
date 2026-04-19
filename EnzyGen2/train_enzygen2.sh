#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash train_enzygen2.sh mlm [-- <extra fairseq args>]
  bash train_enzygen2.sh motif [-- <extra fairseq args>]
  bash train_enzygen2.sh full [-- <extra fairseq args>]
  bash train_enzygen2.sh finetune <chlr|aada|tpmt> [-- <extra fairseq args>]

Environment overrides:
  PYTHON_BIN                Python executable to use. Default: python3
  CUDA_VISIBLE_DEVICES      Visible GPUs. Used to infer distributed world size.
  ENZYGEN2_WORLD_SIZE       Override inferred distributed world size.
  ENZYGEN2_ENABLE_PROFILE   Set to 1 to pass --profile. Default: 0
  LOCAL_ROOT                Model/checkpoint root. Default: models
  PRETRAIN_DATA_PATH        Override pretraining dataset path.
  PRETRAINED_ESM_MODEL      Default: esm2_t33_650M_UR50D
  MLM_OUTPUT_DIR            Default: models/EnzyGen2_MLM
  MOTIF_OUTPUT_DIR          Default: models/EnzyGen2_motif
  FULL_OUTPUT_DIR           Default: models/EnzyGen2
  MLM_CHECKPOINT            Default: models/EnzyGen2_MLM/checkpoint_best.pt
  MOTIF_CHECKPOINT          Default: models/EnzyGen2_motif/checkpoint_best.pt
  FULL_CHECKPOINT           Default: models/EnzyGen2/checkpoint_best.pt
  FINETUNE_OUTPUT_DIR       Override finetune output directory
  FINETUNE_DATA_PATH        Override finetune dataset path
EOF
}

count_visible_devices() {
  local devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${devices}" ]]; then
    local parts=()
    IFS=',' read -r -a parts <<<"${devices}"
    echo "${#parts[@]}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local names
    names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${names}" ]]; then
      local count
      count="$(printf '%s\n' "${names}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
      if [[ -n "${count}" && "${count}" != "0" ]]; then
        echo "${count}"
        return 0
      fi
    fi
  fi

  echo "1"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[EnzyGen2] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

resolve_existing_file() {
  local label="$1"
  shift

  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[EnzyGen2] Missing ${label}. Checked:" >&2
  for candidate in "$@"; do
    if [[ -n "${candidate}" ]]; then
      echo "  - ${candidate}" >&2
    fi
  done
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
shift

target=""
if [[ "${mode}" == "finetune" ]]; then
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  target="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi
extra_args=("$@")

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_ROOT="${LOCAL_ROOT:-models}"
PRETRAINED_ESM_MODEL="${PRETRAINED_ESM_MODEL:-esm2_t33_650M_UR50D}"
WORLD_SIZE="${ENZYGEN2_WORLD_SIZE:-$(count_visible_devices)}"
ENABLE_PROFILE="${ENZYGEN2_ENABLE_PROFILE:-0}"
PRETRAIN_DATA_PATH="${PRETRAIN_DATA_PATH:-data/pdb_swissprot_data_ligand.json}"
NCBI_MAP_PATH="data/ncbi2id.json"
MLM_OUTPUT_DIR="${MLM_OUTPUT_DIR:-${LOCAL_ROOT}/EnzyGen2_MLM}"
MOTIF_OUTPUT_DIR="${MOTIF_OUTPUT_DIR:-${LOCAL_ROOT}/EnzyGen2_motif}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-${LOCAL_ROOT}/EnzyGen2}"
MLM_CHECKPOINT="${MLM_CHECKPOINT:-${MLM_OUTPUT_DIR}/checkpoint_best.pt}"
MOTIF_CHECKPOINT="${MOTIF_CHECKPOINT:-${MOTIF_OUTPUT_DIR}/checkpoint_best.pt}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-${FULL_OUTPUT_DIR}/checkpoint_best.pt}"

require_file "${NCBI_MAP_PATH}" "NCBI taxonomy mapping file"

profile_args=()
if [[ "${ENABLE_PROFILE}" == "1" ]]; then
  profile_args+=(--profile)
fi

common_train_args=(
  --num-workers 0
  --task geometric_protein_design
  --dataset-impl-source raw
  --dataset-impl-target coor
  --pretrained-esm-model "${PRETRAINED_ESM_MODEL}"
  --encoder-embed-dim 1280
  --egnn-mode rm-node
  --decoder-layers 3
  --knn 30
  --dropout 0.3
  --optimizer adam
  --adam-betas "(0.9,0.98)"
  --lr-scheduler inverse_sqrt
  --stop-min-lr 1e-10
  --warmup-updates 4000
  --ddp-backend legacy_ddp
  --log-format simple
  --log-interval 10
  --update-freq 1
  --max-update 1000000
  --validate-after-updates 3000
  --validate-interval-updates 3000
  --save-interval-updates 3000
  --valid-subset valid
  --max-sentences-valid 8
  --validate-interval 1
  --save-interval 1
  --keep-interval-updates 10
  --skip-invalid-size-inputs-valid-test
)

case "${mode}" in
  mlm)
    require_file "${PRETRAIN_DATA_PATH}" "pretraining dataset"
    mkdir -p "${MLM_OUTPUT_DIR}"
    cmd=(
      "${PYTHON_BIN}" fairseq_cli/train.py "${PRETRAIN_DATA_PATH}"
      "${profile_args[@]}"
      --distributed-world-size "${WORLD_SIZE}"
      --save-dir "${MLM_OUTPUT_DIR}"
      --data-stage pretraining-mlm
      --criterion geometric_protein_ncbi_loss
      --encoder-factor 1.0
      --decoder-factor 1e-2
      --arch geometric_protein_model_ncbi_esm
      --lr 1e-4
      --warmup-init-lr 5e-5
      --clip-norm 1.5
      --max-tokens 1024
      --max-epoch 10000
      "${common_train_args[@]}"
      "${extra_args[@]}"
    )
    ;;
  motif)
    require_file "${PRETRAIN_DATA_PATH}" "pretraining dataset"
    require_file "${MLM_CHECKPOINT}" "stage-1 MLM checkpoint"
    mkdir -p "${MOTIF_OUTPUT_DIR}"
    cmd=(
      "${PYTHON_BIN}" fairseq_cli/train.py "${PRETRAIN_DATA_PATH}"
      "${profile_args[@]}"
      --distributed-world-size "${WORLD_SIZE}"
      --finetune-from-model "${MLM_CHECKPOINT}"
      --save-dir "${MOTIF_OUTPUT_DIR}"
      --data-stage pretraining-motif
      --criterion geometric_protein_ncbi_loss
      --encoder-factor 1.0
      --decoder-factor 1e-2
      --arch geometric_protein_model_ncbi_esm
      --lr 1e-4
      --warmup-init-lr 5e-5
      --clip-norm 1.5
      --max-tokens 1024
      --max-epoch 10000
      "${common_train_args[@]}"
      "${extra_args[@]}"
    )
    ;;
  full)
    require_file "${PRETRAIN_DATA_PATH}" "pretraining dataset"
    require_file "${MOTIF_CHECKPOINT}" "stage-2 motif checkpoint"
    mkdir -p "${FULL_OUTPUT_DIR}"
    cmd=(
      "${PYTHON_BIN}" fairseq_cli/train.py "${PRETRAIN_DATA_PATH}"
      "${profile_args[@]}"
      --distributed-world-size "${WORLD_SIZE}"
      --finetune-from-model "${MOTIF_CHECKPOINT}"
      --save-dir "${FULL_OUTPUT_DIR}"
      --data-stage pretraining-full
      --criterion geometric_protein_ncbi_substrate_loss
      --encoder-factor 1.0
      --decoder-factor 1e-2
      --binding-factor 0.5
      --arch geometric_protein_model_ncbi_substrate_esm
      --lr 5e-5
      --warmup-init-lr 1e-5
      --clip-norm 0.0001
      --max-tokens 800
      --max-source-positions 800
      --max-target-positions 800
      --max-sentences 1
      --max-epoch 2000
      "${common_train_args[@]}"
      "${extra_args[@]}"
    )
    ;;
  finetune)
    require_file "${FULL_CHECKPOINT}" "pretrained EnzyGen2 checkpoint"

    case "${target}" in
      chlr)
        reaction="18421"
        finetune_data_path="$(
          resolve_existing_file \
            "ChlR finetune dataset" \
            "${FINETUNE_DATA_PATH:-}" \
            "data/rhea_18421_final.json" \
            "data/chloramphenicol_acetyltransferase_final.json"
        )"
        default_finetune_output="${LOCAL_ROOT}/rhea_18421_finetune"
        ;;
      aada)
        reaction="20245"
        finetune_data_path="$(
          resolve_existing_file \
            "AadA finetune dataset" \
            "${FINETUNE_DATA_PATH:-}" \
            "data/rhea_20245_final.json" \
            "data/aminoglycoside_adenylyltransferase_final.json"
        )"
        default_finetune_output="${LOCAL_ROOT}/rhea_20245_finetune"
        ;;
      tpmt)
        reaction="Thiopurine_S_methyltransferas"
        finetune_data_path="$(
          resolve_existing_file \
            "TPMT finetune dataset" \
            "${FINETUNE_DATA_PATH:-}" \
            "data/Thiopurine_S_methyltransferas_final.json" \
            "data/thiopurine_methyltransferase_final.json"
        )"
        default_finetune_output="${LOCAL_ROOT}/rhea_Thiopurine_S_methyltransferas_finetune"
        ;;
      *)
        echo "[EnzyGen2] Unsupported finetune target: ${target}" >&2
        usage
        exit 1
        ;;
    esac

    FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${default_finetune_output}}"
    mkdir -p "${FINETUNE_OUTPUT_DIR}"
    cmd=(
      "${PYTHON_BIN}" fairseq_cli/finetune.py "${finetune_data_path}"
      --distributed-world-size "${WORLD_SIZE}"
      --finetune-from-model "${FULL_CHECKPOINT}"
      --save-dir "${FINETUNE_OUTPUT_DIR}"
      --task geometric_protein_design
      --protein-task "${reaction}"
      --dataset-impl-source raw
      --dataset-impl-target coor
      --data-stage finetuning
      --criterion geometric_protein_ncbi_loss
      --encoder-factor 1.0
      --decoder-factor 1e-2
      --arch geometric_protein_model_ncbi_esm
      --encoder-embed-dim 1280
      --egnn-mode rm-node
      --decoder-layers 3
      --pretrained-esm-model "${PRETRAINED_ESM_MODEL}"
      --knn 30
      --dropout 0.3
      --optimizer adam
      --adam-betas "(0.9,0.98)"
      --lr 5e-5
      --lr-scheduler inverse_sqrt
      --stop-min-lr 1e-10
      --warmup-updates 4000
      --warmup-init-lr 1e-5
      --clip-norm 0.0001
      --ddp-backend legacy_ddp
      --log-format simple
      --log-interval 10
      --max-tokens 1024
      --update-freq 1
      --max-update 300000
      --max-epoch 50
      --validate-after-updates 3000
      --validate-interval-updates 3000
      --save-interval-updates 3000
      --valid-subset valid
      --max-sentences-valid 8
      --validate-interval 1
      --save-interval 1
      --keep-interval-updates 10
      --skip-invalid-size-inputs-valid-test
      "${extra_args[@]}"
    )
    ;;
  *)
    echo "[EnzyGen2] Unsupported mode: ${mode}" >&2
    usage
    exit 1
    ;;
esac

echo "[EnzyGen2] Repo root: ${REPO_ROOT}"
echo "[EnzyGen2] Mode: ${mode}${target:+ (${target})}"
echo "[EnzyGen2] Python: ${PYTHON_BIN}"
echo "[EnzyGen2] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[EnzyGen2] Distributed world size: ${WORLD_SIZE}"
echo "[EnzyGen2] Launching:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
