#!/bin/bash
# Train CutLER Cascade Mask R-CNN on Luiz's 5-class TinyImageNet pseudo-labels.
#
# Usage:
#   PSEUDO_LABEL_NAME=baseline5   sbatch slurm/run_training_luiz.sh
#   PSEUDO_LABEL_NAME=multiscale5 sbatch slurm/run_training_luiz.sh
#
# Expected files:
#   ${DATA_ROOT}/tiny-imagenet-5/annotations/v1_baseline_pseudo.json
#   ${DATA_ROOT}/tiny-imagenet-5/annotations/v1_multiscale_pseudo.json
#   ${DATA_ROOT}/tiny-imagenet-5/train_flat/

#SBATCH --job-name=cutler-train-luiz
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/training_luiz_%x_%j.out
#SBATCH --error=logs/training_luiz_%x_%j.err

set -euo pipefail

if [[ -z "${PSEUDO_LABEL_NAME:-}" ]]; then
    echo "ERROR: PSEUDO_LABEL_NAME is not set."
    echo "Usage: PSEUDO_LABEL_NAME=baseline5 sbatch slurm/run_training_luiz.sh"
    exit 1
fi

source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export DATA_ROOT
export DETECTRON2_DATASETS="${DATA_ROOT}"

ANNO_DIR_5C="${DATA_ROOT}/tiny-imagenet-5/annotations"
IMAGE_ROOT_5C="${DATA_ROOT}/tiny-imagenet-5/train_flat"

case "${PSEUDO_LABEL_NAME}" in
    baseline5)
        PSEUDO_JSON="${ANNO_DIR_5C}/v1_baseline_pseudo.json"
        DATASET_NAME="tinyimagenet_5c_baseline_pseudo"
        ;;
    multiscale5)
        PSEUDO_JSON="${ANNO_DIR_5C}/v1_multiscale_pseudo.json"
        DATASET_NAME="tinyimagenet_5c_multiscale_pseudo"
        ;;
    *)
        echo "ERROR: Unknown PSEUDO_LABEL_NAME '${PSEUDO_LABEL_NAME}'. Expected 'baseline5' or 'multiscale5'."
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/training_${PSEUDO_LABEL_NAME}}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${PSEUDO_JSON}" ]; then
    echo "ERROR: pseudo-label JSON not found: ${PSEUDO_JSON}"
    exit 1
fi

if [ ! -d "${IMAGE_ROOT_5C}" ]; then
    echo "ERROR: image root not found: ${IMAGE_ROOT_5C}"
    exit 1
fi

echo "=== CutLER training: Luiz 5-class TinyImageNet ==="
echo "  PSEUDO_LABEL_NAME : ${PSEUDO_LABEL_NAME}"
echo "  DATASET_NAME      : ${DATASET_NAME}"
echo "  DATA_ROOT         : ${DATA_ROOT}"
echo "  PSEUDO_JSON       : ${PSEUDO_JSON}"
echo "  IMAGE_ROOT        : ${IMAGE_ROOT_5C}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  SLURM_JOB_ID      : ${SLURM_JOB_ID:-local}"

cd "${REPO_ROOT}/CutLER/cutler"

python "${REPO_ROOT}/tools/train_wrapper_luiz.py" \
    --num-gpus 1 \
    --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
    DATASETS.TRAIN "(\"${DATASET_NAME}\",)" \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.005 \
    SOLVER.MAX_ITER 20000 \
    SOLVER.STEPS "(15000,)" \
    SOLVER.WARMUP_ITERS 1000 \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR "${OUTPUT_DIR}"
