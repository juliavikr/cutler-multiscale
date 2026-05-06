#!/bin/bash
# Generate Luiz's 5-class TinyImageNet final multiscale pseudo-labels in one job.
#
# Output:
#   ${DATA_ROOT}/tiny-imagenet-5/annotations/v1_final_pseudo.json

#SBATCH --job-name=final-ms-luiz
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/final_multiscale_luiz_%j.out
#SBATCH --error=logs/final_multiscale_luiz_%j.err

set -euo pipefail

module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

if [[ -z "${REPO_ROOT:-}" ]]; then
    SEARCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    while [[ "${SEARCH_DIR}" != "/" ]]; do
        if [[ -f "${SEARCH_DIR}/multiscale/final_multiscale.py" && -d "${SEARCH_DIR}/CutLER" ]]; then
            REPO_ROOT="${SEARCH_DIR}"
            break
        fi
        SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
    done
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "ERROR: Could not find repo root. Submit from the repo or pass REPO_ROOT=/path/to/cutler-multiscale."
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-/home/3191856/data}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/tiny-imagenet-5/train_flat}"
ANNO_DIR="${ANNO_DIR:-${DATA_ROOT}/tiny-imagenet-5/annotations}"
RUN_DIR="${RUN_DIR:-${ANNO_DIR}/final_multiscale_onego}"
FINAL_JSON="${FINAL_JSON:-${ANNO_DIR}/v1_final_pseudo.json}"
DINO_WEIGHTS="${DINO_WEIGHTS:-${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth}"

TAU="${TAU:-0.2}"
FIXED_SIZE="${FIXED_SIZE:-480}"
N_MASKS="${N_MASKS:-1}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-8}"
LOG_EVERY="${LOG_EVERY:-50}"

mkdir -p "${RUN_DIR}" "${ANNO_DIR}" logs

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: DATASET_PATH not found: ${DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${DINO_WEIGHTS}" ]]; then
    echo "ERROR: DINO weights not found: ${DINO_WEIGHTS}"
    echo "Pass DINO_WEIGHTS=/path/to/dino_deitsmall8_pretrain.pth or download the checkpoint first."
    exit 1
fi

CLASS_COUNT="$(find "${DATASET_PATH}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${CLASS_COUNT}" -lt 1 ]]; then
    echo "ERROR: no class folders found under ${DATASET_PATH}"
    exit 1
fi

echo "=== Final multiscale MaskCut: Luiz 5-class TinyImageNet ==="
echo "  REPO_ROOT       : ${REPO_ROOT}"
echo "  DATA_ROOT       : ${DATA_ROOT}"
echo "  DATASET_PATH    : ${DATASET_PATH}"
echo "  CLASS_COUNT     : ${CLASS_COUNT}"
echo "  RUN_DIR         : ${RUN_DIR}"
echo "  FINAL_JSON      : ${FINAL_JSON}"
echo "  DINO_WEIGHTS    : ${DINO_WEIGHTS}"
echo "  SLURM_JOB_ID    : ${SLURM_JOB_ID:-local}"

cd "${REPO_ROOT}"

python multiscale/final_multiscale.py \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N_MASKS}" \
    --dataset-path "${DATASET_PATH}" \
    --num-folder-per-job "${CLASS_COUNT}" \
    --job-index 0 \
    --pretrain_path "${DINO_WEIGHTS}" \
    --out-dir "${RUN_DIR}" \
    --multi-crop \
    --ms-preset final \
    --primary-output final \
    --crop-batch-size "${CROP_BATCH_SIZE}" \
    --log-every "${LOG_EVERY}"

GENERATED="$(find "${RUN_DIR}" -maxdepth 1 -type f -name "imagenet_train_fixsize${FIXED_SIZE}_tau${TAU}_N${N_MASKS}*.json" \
    ! -name "*_normal.json" \
    ! -name "*_raw_multiscale.json" \
    ! -name "*_multiscale.json" \
    ! -name "*_combined.json" \
    ! -name "*_final.json" \
    ! -name "*_candidate_debug.json" \
    ! -name "checkpoint.json" \
    | sort | tail -1)"

if [[ -z "${GENERATED}" ]]; then
    echo "ERROR: could not find generated primary JSON in ${RUN_DIR}"
    ls -lh "${RUN_DIR}"
    exit 1
fi

cp "${GENERATED}" "${FINAL_JSON}"

python - <<PY
import json
path = "${FINAL_JSON}"
d = json.load(open(path))
print("=== Final pseudo-label JSON ===")
print("path:", path)
print("images:", len(d.get("images", [])))
print("annotations:", len(d.get("annotations", [])))
print("anns/image:", round(len(d.get("annotations", [])) / max(1, len(d.get("images", []))), 3))
PY

echo "=== Done ==="
echo "Final pseudo-labels saved to ${FINAL_JSON}"
