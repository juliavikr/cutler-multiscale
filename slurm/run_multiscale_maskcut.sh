#!/bin/bash

#SBATCH --job-name=cutler-ms-maskcut
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ms_maskcut_%j.out
#SBATCH --error=logs/ms_maskcut_%j.err

set -euo pipefail

module load miniconda3
conda activate cutler

# --- Paths (edit before submitting) ---
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"                       # override via env if needed
DATASET_PATH="${DATA_ROOT}/tiny-imagenet-200/train"         # 200 class subdirectories
OUT_DIR="${REPO_ROOT}/pseudo_masks/tiny_imagenet"
DINO_WEIGHTS="${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth"

mkdir -p "${OUT_DIR}"

# --- Multi-scale MaskCut knobs (override via env) ---
TAU="${TAU:-0.2}"
N_MASKS="${N_MASKS:-3}"
FIXED_SIZE="${FIXED_SIZE:-480}"
USE_CPU="${USE_CPU:-0}"
CROP_SCALES="${CROP_SCALES:-1.0,0.75,0.5}"
CROP_OVERLAP="${CROP_OVERLAP:-0.3}"
CROP_MAX_PER_SCALE="${CROP_MAX_PER_SCALE:-9}"
MERGE_IOU_THRESH="${MERGE_IOU_THRESH:-0.5}"
KEEP_TOPK="${KEEP_TOPK:-20}"
MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-0.0005}"
MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-1.0}"
SMALL_FIRST="${SMALL_FIRST:-1}"

EXTRA_ARGS=(
  --multi-crop
  --crop-scales "${CROP_SCALES}"
  --crop-overlap "${CROP_OVERLAP}"
  --crop-max-per-scale "${CROP_MAX_PER_SCALE}"
  --merge-iou-thresh "${MERGE_IOU_THRESH}"
  --keep-topk "${KEEP_TOPK}"
  --min-mask-area-ratio "${MIN_MASK_AREA_RATIO}"
  --max-mask-area-ratio "${MAX_MASK_AREA_RATIO}"
)

if [ "${SMALL_FIRST}" = "1" ]; then
  EXTRA_ARGS+=(--small-first)
fi
if [ "${USE_CPU}" = "1" ]; then
  EXTRA_ARGS+=(--cpu)
fi

python "${REPO_ROOT}/CutLER/maskcut/multiscale_maskcut.py" \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N_MASKS}" \
    --dataset-path "${DATASET_PATH}" \
    --num-folder-per-job 200 \
    --job-index 0 \
    --pretrain_path "${DINO_WEIGHTS}" \
    --out-dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "Multi-scale MaskCut done. Annotations saved to ${OUT_DIR}."
