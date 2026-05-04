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
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/tiny-imagenet-200/train}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/pseudo_masks/tiny_imagenet}"
DINO_WEIGHTS="${DINO_WEIGHTS:-${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth}"
MASKCUT_SCRIPT="${MASKCUT_SCRIPT:-multiscale/multiscale_maskcut.py}"
NUM_FOLDER_PER_JOB="${NUM_FOLDER_PER_JOB:-200}"
JOB_INDEX="${JOB_INDEX:-0}"

mkdir -p "${OUT_DIR}"

# --- Core experiment knobs (override via env) ---
TAU="${TAU:-0.2}"
N_MASKS="${N_MASKS:-3}"
FIXED_SIZE="${FIXED_SIZE:-480}"
USE_CPU="${USE_CPU:-0}"
MS_PRESET="${MS_PRESET:-small}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-8}"
CROP_N="${CROP_N:-}"
CROP_KEEP_PER_WINDOW="${CROP_KEEP_PER_WINDOW:-}"
PRIMARY_OUTPUT="${PRIMARY_OUTPUT:-multiscale}"
LOG_EVERY="${LOG_EVERY:-50}"

# Optional advanced overrides. Leave unset to use --ms-preset.
CROP_MODE="${CROP_MODE:-}"
CROP_SCALES="${CROP_SCALES:-}"
CROP_OVERLAP="${CROP_OVERLAP:-}"
CROP_MAX_PER_SCALE="${CROP_MAX_PER_SCALE:-}"
MERGE_IOU_THRESH="${MERGE_IOU_THRESH:-}"
KEEP_TOPK="${KEEP_TOPK:-}"
MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-}"
MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-}"
SMALL_FIRST="${SMALL_FIRST:-0}"
TWO_STAGE_CROP="${TWO_STAGE_CROP:-}"
TWO_STAGE_MAX_COVERED_RATIO="${TWO_STAGE_MAX_COVERED_RATIO:-}"
CONTAINMENT_THRESH="${CONTAINMENT_THRESH:-}"
BOX_EXPAND_RATIO="${BOX_EXPAND_RATIO:-}"
MERGE_MAX_ASPECT_RATIO="${MERGE_MAX_ASPECT_RATIO:-}"
CROP_TOP_K="${CROP_TOP_K:-}"
HEATMAP_CROP_SIZES="${HEATMAP_CROP_SIZES:-}"
HEATMAP_TOP_K="${HEATMAP_TOP_K:-}"
HEATMAP_NMS_IOU="${HEATMAP_NMS_IOU:-}"
HEATMAP_PERCENTILE="${HEATMAP_PERCENTILE:-}"
HEATMAP_SPATIAL_RESCUE="${HEATMAP_SPATIAL_RESCUE:-}"
MOSTLITE_PERCENTILE="${MOSTLITE_PERCENTILE:-}"
MOSTLITE_SIM_PERCENTILE="${MOSTLITE_SIM_PERCENTILE:-}"
BORDER_RETRY="${BORDER_RETRY:-0}"
BORDER_RETRY_SCALES="${BORDER_RETRY_SCALES:-}"
BORDER_RETRY_TOUCH_THRESH="${BORDER_RETRY_TOUCH_THRESH:-}"
BORDER_RETRY_SIDES_THRESH="${BORDER_RETRY_SIDES_THRESH:-}"
CROP_SHAPE_REJECT="${CROP_SHAPE_REJECT:-0}"
CROP_FILL_THRESH="${CROP_FILL_THRESH:-}"
CRF_IOU_THRESH="${CRF_IOU_THRESH:-}"

EXTRA_ARGS=(
  --multi-crop
  --ms-preset "${MS_PRESET}"
  --crop-batch-size "${CROP_BATCH_SIZE}"
  --primary-output "${PRIMARY_OUTPUT}"
  --log-every "${LOG_EVERY}"
)

add_arg_if_set() {
  local value="$1"
  shift
  if [ -n "${value}" ]; then
    EXTRA_ARGS+=("$@" "${value}")
  fi
}

add_arg_if_set "${CROP_MODE}" --crop-mode
add_arg_if_set "${CROP_SCALES}" --crop-scales
add_arg_if_set "${CROP_OVERLAP}" --crop-overlap
add_arg_if_set "${CROP_MAX_PER_SCALE}" --crop-max-per-scale
add_arg_if_set "${CROP_N}" --crop-n
add_arg_if_set "${CROP_KEEP_PER_WINDOW}" --crop-keep-per-window
add_arg_if_set "${MERGE_IOU_THRESH}" --merge-iou-thresh
add_arg_if_set "${KEEP_TOPK}" --keep-topk
add_arg_if_set "${MIN_MASK_AREA_RATIO}" --min-mask-area-ratio
add_arg_if_set "${MAX_MASK_AREA_RATIO}" --max-mask-area-ratio
add_arg_if_set "${TWO_STAGE_MAX_COVERED_RATIO}" --two-stage-max-covered-ratio
add_arg_if_set "${CONTAINMENT_THRESH}" --containment-thresh
add_arg_if_set "${BOX_EXPAND_RATIO}" --box-expand-ratio
add_arg_if_set "${MERGE_MAX_ASPECT_RATIO}" --merge-max-aspect-ratio
add_arg_if_set "${CROP_TOP_K}" --crop-top-k
add_arg_if_set "${HEATMAP_CROP_SIZES}" --heatmap-crop-sizes
add_arg_if_set "${HEATMAP_TOP_K}" --heatmap-top-k
add_arg_if_set "${HEATMAP_NMS_IOU}" --heatmap-nms-iou
add_arg_if_set "${HEATMAP_PERCENTILE}" --heatmap-percentile
add_arg_if_set "${HEATMAP_SPATIAL_RESCUE}" --heatmap-spatial-rescue
add_arg_if_set "${MOSTLITE_PERCENTILE}" --mostlite-percentile
add_arg_if_set "${MOSTLITE_SIM_PERCENTILE}" --mostlite-sim-percentile
add_arg_if_set "${BORDER_RETRY_SCALES}" --border-retry-scales
add_arg_if_set "${BORDER_RETRY_TOUCH_THRESH}" --border-retry-touch-thresh
add_arg_if_set "${BORDER_RETRY_SIDES_THRESH}" --border-retry-sides-thresh
add_arg_if_set "${CROP_FILL_THRESH}" --crop-fill-thresh
add_arg_if_set "${CRF_IOU_THRESH}" --crf-iou-thresh

if [ "${SMALL_FIRST}" = "1" ]; then
  EXTRA_ARGS+=(--small-first)
fi
if [ "${BORDER_RETRY}" = "1" ]; then
  EXTRA_ARGS+=(--border-retry)
fi
if [ "${CROP_SHAPE_REJECT}" = "1" ]; then
  EXTRA_ARGS+=(--crop-shape-reject)
fi
if [ "${TWO_STAGE_CROP}" = "1" ]; then
  EXTRA_ARGS+=(--two-stage-crop)
fi
if [ "${USE_CPU}" = "1" ]; then
  EXTRA_ARGS+=(--cpu)
fi
python "${REPO_ROOT}/${MASKCUT_SCRIPT}" \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N_MASKS}" \
    --dataset-path "${DATASET_PATH}" \
    --num-folder-per-job "${NUM_FOLDER_PER_JOB}" \
    --job-index "${JOB_INDEX}" \
    --pretrain_path "${DINO_WEIGHTS}" \
    --out-dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "Multi-scale MaskCut done. Annotations saved to ${OUT_DIR}."
