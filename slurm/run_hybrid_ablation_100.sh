#!/bin/bash

# Run one 100-image hybrid ablation on the 5-class TinyImageNet subset.
# The job creates a deterministic symlinked subset, then runs the main
# multiscale MaskCut implementation with one named ablation setting.
#
# Recommended usage:
#   sbatch --export=ALL,REPO_ROOT=/home/<id>/cv_project/cutler-multiscale,DATA_ROOT=/home/<id>/data,VARIANT=baseline \
#     slurm/run_hybrid_ablation_100.sh
#
# Available VARIANT values:
#   baseline    -> current best hybrid
#   hp90        -> more conservative heatmap threshold
#   hp80        -> more permissive heatmap threshold
#   topk8       -> fewer crop windows
#   tightcrop   -> smaller crop sizes

#SBATCH --job-name=hybrid-ablate
# TODO: set your SLURM account via SBATCH_ACCOUNT or --account=<student_number>.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hybrid_ablation_%j.out
#SBATCH --error=logs/hybrid_ablation_%j.err

set -euo pipefail

module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
SOURCE_DATASET="${SOURCE_DATASET:-${DATA_ROOT}/tiny-imagenet-5/train_flat}"
DINO_WEIGHTS="${DINO_WEIGHTS:-file://${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth}"
MASKCUT_SCRIPT="${MASKCUT_SCRIPT:-multiscale/multiscale_maskcut.py}"

TOTAL_IMAGES="${TOTAL_IMAGES:-100}"
CLASS_COUNT="${CLASS_COUNT:-5}"
IMAGES_PER_CLASS="${IMAGES_PER_CLASS:-$((TOTAL_IMAGES / CLASS_COUNT))}"
VARIANT="${VARIANT:-baseline}"

SUBSET_ROOT="${SUBSET_ROOT:-${DATA_ROOT}/tiny-imagenet-5-ablations/subset_${TOTAL_IMAGES}}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/tiny-imagenet-5/annotations/hybrid_ablations}"
RUN_DIR="${OUT_ROOT}/${VARIANT}_${TOTAL_IMAGES}"

TAU="${TAU:-0.2}"
FIXED_SIZE="${FIXED_SIZE:-480}"
N_MASKS="${N_MASKS:-1}"
MS_PRESET="${MS_PRESET:-small}"
PRIMARY_OUTPUT="${PRIMARY_OUTPUT:-combined}"
LOG_EVERY="${LOG_EVERY:-25}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-8}"
CROP_N="${CROP_N:-1}"
CROP_KEEP_PER_WINDOW="${CROP_KEEP_PER_WINDOW:-1}"

KEEP_TOPK="${KEEP_TOPK:-12}"
HEATMAP_TOP_K="${HEATMAP_TOP_K:-12}"
HEATMAP_PERCENTILE="${HEATMAP_PERCENTILE:-85}"
HEATMAP_CROP_SIZES="${HEATMAP_CROP_SIZES:-0.25,0.35,0.5}"
HEATMAP_SPATIAL_RESCUE="${HEATMAP_SPATIAL_RESCUE:-4}"
MERGE_IOU_THRESH="${MERGE_IOU_THRESH:-0.5}"

case "${VARIANT}" in
  baseline)
    ;;
  hp90)
    HEATMAP_PERCENTILE=90
    ;;
  hp80)
    HEATMAP_PERCENTILE=80
    ;;
  topk8)
    KEEP_TOPK=8
    HEATMAP_TOP_K=8
    ;;
  tightcrop)
    HEATMAP_CROP_SIZES="0.2,0.3,0.4"
    ;;
  *)
    echo "ERROR: Unknown VARIANT '${VARIANT}'."
    echo "Expected one of: baseline, hp90, hp80, topk8, tightcrop"
    exit 1
    ;;
esac

if [ $((IMAGES_PER_CLASS * CLASS_COUNT)) -ne "${TOTAL_IMAGES}" ]; then
    echo "ERROR: TOTAL_IMAGES=${TOTAL_IMAGES} must be divisible by CLASS_COUNT=${CLASS_COUNT}"
    exit 1
fi

if [[ "${DINO_WEIGHTS}" != file://* ]] && [[ "${DINO_WEIGHTS}" != http://* ]] && [[ "${DINO_WEIGHTS}" != https://* ]]; then
    DINO_WEIGHTS="file://${DINO_WEIGHTS}"
fi

mkdir -p "${SUBSET_ROOT}" "${RUN_DIR}"

echo "=== Hybrid ablation run ==="
echo "  REPO_ROOT           : ${REPO_ROOT}"
echo "  SOURCE_DATASET      : ${SOURCE_DATASET}"
echo "  SUBSET_ROOT         : ${SUBSET_ROOT}"
echo "  RUN_DIR             : ${RUN_DIR}"
echo "  VARIANT             : ${VARIANT}"
echo "  TOTAL_IMAGES        : ${TOTAL_IMAGES}"
echo "  IMAGES_PER_CLASS    : ${IMAGES_PER_CLASS}"
echo "  DINO_WEIGHTS        : ${DINO_WEIGHTS}"
echo "  TAU                 : ${TAU}"
echo "  FIXED_SIZE          : ${FIXED_SIZE}"
echo "  N_MASKS             : ${N_MASKS}"
echo "  HEATMAP_PERCENTILE  : ${HEATMAP_PERCENTILE}"
echo "  HEATMAP_TOP_K       : ${HEATMAP_TOP_K}"
echo "  KEEP_TOPK           : ${KEEP_TOPK}"
echo "  HEATMAP_CROP_SIZES  : ${HEATMAP_CROP_SIZES}"

mapfile -t class_dirs < <(find -L "${SOURCE_DATASET}" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#class_dirs[@]}" -ne "${CLASS_COUNT}" ]; then
    echo "ERROR: Expected ${CLASS_COUNT} class folders under ${SOURCE_DATASET}, found ${#class_dirs[@]}"
    exit 1
fi

for class_dir in "${class_dirs[@]}"; do
    class_name="$(basename "${class_dir}")"
    target_dir="${SUBSET_ROOT}/${class_name}"
    mkdir -p "${target_dir}"
    find "${target_dir}" -mindepth 1 -maxdepth 1 -type l -delete

    mapfile -t image_paths < <(find -L "${class_dir}" -maxdepth 1 -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | sort | head -n "${IMAGES_PER_CLASS}")
    if [ "${#image_paths[@]}" -lt "${IMAGES_PER_CLASS}" ]; then
        echo "ERROR: Class ${class_name} has only ${#image_paths[@]} images, expected at least ${IMAGES_PER_CLASS}"
        exit 1
    fi

    for image_path in "${image_paths[@]}"; do
        ln -sfn "${image_path}" "${target_dir}/$(basename "${image_path}")"
    done
done

python "${REPO_ROOT}/${MASKCUT_SCRIPT}" \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N_MASKS}" \
    --dataset-path "${SUBSET_ROOT}" \
    --num-folder-per-job "${CLASS_COUNT}" \
    --job-index 0 \
    --pretrain_path "${DINO_WEIGHTS}" \
    --out-dir "${RUN_DIR}" \
    --multi-crop \
    --ms-preset "${MS_PRESET}" \
    --crop-batch-size "${CROP_BATCH_SIZE}" \
    --crop-n "${CROP_N}" \
    --crop-keep-per-window "${CROP_KEEP_PER_WINDOW}" \
    --primary-output "${PRIMARY_OUTPUT}" \
    --log-every "${LOG_EVERY}" \
    --keep-topk "${KEEP_TOPK}" \
    --heatmap-top-k "${HEATMAP_TOP_K}" \
    --heatmap-percentile "${HEATMAP_PERCENTILE}" \
    --heatmap-crop-sizes "${HEATMAP_CROP_SIZES}" \
    --heatmap-spatial-rescue "${HEATMAP_SPATIAL_RESCUE}" \
    --merge-iou-thresh "${MERGE_IOU_THRESH}"

echo "Hybrid ablation done. Outputs saved to ${RUN_DIR}"
