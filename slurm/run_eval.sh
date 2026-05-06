#!/bin/bash

#SBATCH --job-name=cutler-eval
# TODO: set your SLURM account â€” export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

# --- Paths (edit before submitting) ---
if [[ -z "${REPO_ROOT:-}" ]]; then
    SEARCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    while [[ "${SEARCH_DIR}" != "/" ]]; do
        if [[ -f "${SEARCH_DIR}/CutLER/cutler/train_net.py" && -f "${SEARCH_DIR}/tools/make_cls_agnostic_coco.py" ]]; then
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

DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
CONFIG="${REPO_ROOT}/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"

if [ -n "${PSEUDO_LABEL_NAME:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/training_${PSEUDO_LABEL_NAME}}"
    CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/model_final.pth}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/cutler_r50_1gpu}"
    CHECKPOINT="${CHECKPOINT:-${HOME}/cutler-multiscale/checkpoints/cutler_cascade_final.pth}"
fi

# COCO val2017 must be registered; point detectron2 to the dataset
export DETECTRON2_DATASETS="${DATA_ROOT}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/eval"
cd "${REPO_ROOT}/CutLER/cutler"

echo "=== CutLER COCO eval ==="
echo "  PSEUDO_LABEL_NAME : ${PSEUDO_LABEL_NAME:-pretrained_cutler}"
echo "  CHECKPOINT        : ${CHECKPOINT}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}/eval"

python train_net.py \
    --num-gpus 1 \
    --eval-only \
    --config-file "${CONFIG}" \
    DATASETS.TEST '("cls_agnostic_coco",)' \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    TEST.DETECTIONS_PER_IMAGE 100 \
    OUTPUT_DIR "${OUTPUT_DIR}/eval" \
    2>&1 | tee "${OUTPUT_DIR}/eval/eval.log"

# Extract AP / AP_S / AP_M summary from the log
echo ""
echo "=== COCO Evaluation Summary ==="
grep -E "AP |AP50|AP75|APs|APm|APl|copypaste" "${OUTPUT_DIR}/eval/eval.log" | tail -20

echo ""
echo "Full results in ${OUTPUT_DIR}/eval/eval.log"
