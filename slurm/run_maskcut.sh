#!/bin/bash

#SBATCH --job-name=cutler-maskcut
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/maskcut_%j.out
#SBATCH --error=logs/maskcut_%j.err

set -euo pipefail

module load miniconda3
conda activate cutler

# --- Paths (edit before submitting) ---
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"                          # override via env if needed
DATASET_PATH="${DATA_ROOT}/tiny-imagenet-200/train"         # 200 class subdirectories
OUT_DIR="${REPO_ROOT}/pseudo_masks/tiny_imagenet"
DINO_WEIGHTS="${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth"

mkdir -p "${OUT_DIR}"

# --- MaskCut args ---
# vit-arch=small + patch-size=8 matches the DINO ViT-S/8 checkpoint
# tau=0.2 is the default threshold; N=3 gives up to 3 masks per image
# num-folder-per-job=200 processes all TinyImageNet classes in one job
python "${REPO_ROOT}/CutLER/maskcut/maskcut.py" \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau 0.2 \
    --fixed_size 480 \
    --N 3 \
    --dataset-path "${DATASET_PATH}" \
    --num-folder-per-job 200 \
    --job-index 0 \
    --pretrain_path "${DINO_WEIGHTS}" \
    --out-dir "${OUT_DIR}"

echo "MaskCut done. Annotations saved to ${OUT_DIR}."
