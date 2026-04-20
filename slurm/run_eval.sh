#!/bin/bash

#SBATCH --job-name=cutler-eval
#SBATCH --account=3355142
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
conda activate cutler

# --- Paths (edit before submitting) ---
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"
OUTPUT_DIR="${REPO_ROOT}/output/cutler_r50_1gpu"             # directory from run_training.sh
CONFIG="${REPO_ROOT}/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"

# Use the final checkpoint if it exists, otherwise the latest
CHECKPOINT="${OUTPUT_DIR}/model_final.pth"
if [ ! -f "${CHECKPOINT}" ]; then
    CHECKPOINT="$(cat "${OUTPUT_DIR}/last_checkpoint")"
    echo "model_final.pth not found — using last checkpoint: ${CHECKPOINT}"
fi

# COCO val2017 must be registered; point detectron2 to the dataset
export DETECTRON2_DATASETS="${DATA_ROOT}"

cd "${REPO_ROOT}/CutLER/cutler"

python train_net.py \
    --num-gpus 1 \
    --eval-only \
    --config-file "${CONFIG}" \
    DATASETS.TEST '("coco_2017_val",)' \
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
