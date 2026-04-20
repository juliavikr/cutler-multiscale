#!/bin/bash

#SBATCH --job-name=cutler-train
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

set -euo pipefail

module load miniconda3
conda activate cutler

# --- Paths (edit before submitting) ---
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
PSEUDO_LABELS="${REPO_ROOT}/pseudo_masks/tiny_imagenet"     # JSON from run_maskcut.sh
IMAGE_DIR="${DATA_ROOT}/tiny-imagenet-200/train"
OUTPUT_DIR="${REPO_ROOT}/output/cutler_r50_1gpu"
CONFIG="${REPO_ROOT}/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"

mkdir -p "${OUTPUT_DIR}"

# Single-GPU adjustments vs the default 8-GPU config:
#   IMS_PER_BATCH: 16 -> 2  (2 images fit per GPU with Cascade Mask R-CNN R50)
#   BASE_LR:     0.01 -> 0.00125  (linear scaling: 0.01 * 2/16)
#   STEPS and MAX_ITER unchanged (iteration-based schedule, not epoch-based)

cd "${REPO_ROOT}/CutLER/cutler"

python train_net.py \
    --num-gpus 1 \
    --config-file "${CONFIG}" \
    DATASETS.TRAIN '("imagenet_train",)' \
    DATASETS.TEST '("coco_2017_val",)' \
    DATALOADER.NUM_WORKERS 4 \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.00125 \
    SOLVER.MAX_ITER 160000 \
    SOLVER.STEPS '(80000,)' \
    MODEL.RESNETS.NORM '"BN"' \
    MODEL.FPN.NORM '"BN"' \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training done. Checkpoints in ${OUTPUT_DIR}."
