#!/bin/bash

# Run the full 100-image hybrid ablation suite inside one SLURM job.
# This does not submit any child jobs; it creates the subset once and
# executes each variant sequentially in the same allocation.
#
# Usage:
#   sbatch --export=ALL,REPO_ROOT=/home/<id>/cv_project/cutler-multiscale,DATA_ROOT=/home/<id>/data \
#     slurm/run_hybrid_ablations_100_onejob.sh

#SBATCH --job-name=hybrid-ablate-all
# TODO: set your SLURM account via SBATCH_ACCOUNT or --account=<student_number>.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hybrid_ablation_all_%j.out
#SBATCH --error=logs/hybrid_ablation_all_%j.err

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
VARIANTS_STRING="${VARIANTS:-baseline hp90 hp80 topk8 tightcrop}"

SUBSET_ROOT="${SUBSET_ROOT:-${DATA_ROOT}/tiny-imagenet-5-ablations/subset_${TOTAL_IMAGES}}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/tiny-imagenet-5/annotations/hybrid_ablations}"

TAU="${TAU:-0.2}"
FIXED_SIZE="${FIXED_SIZE:-480}"
N_MASKS="${N_MASKS:-1}"
MS_PRESET="${MS_PRESET:-small}"
PRIMARY_OUTPUT="${PRIMARY_OUTPUT:-combined}"
LOG_EVERY="${LOG_EVERY:-25}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-8}"
CROP_N="${CROP_N:-1}"
CROP_KEEP_PER_WINDOW="${CROP_KEEP_PER_WINDOW:-1}"
HEATMAP_SPATIAL_RESCUE="${HEATMAP_SPATIAL_RESCUE:-4}"
MERGE_IOU_THRESH="${MERGE_IOU_THRESH:-0.5}"

read -r -a VARIANTS <<< "${VARIANTS_STRING}"

if [ $((IMAGES_PER_CLASS * CLASS_COUNT)) -ne "${TOTAL_IMAGES}" ]; then
    echo "ERROR: TOTAL_IMAGES=${TOTAL_IMAGES} must be divisible by CLASS_COUNT=${CLASS_COUNT}"
    exit 1
fi

if [[ "${DINO_WEIGHTS}" != file://* ]] && [[ "${DINO_WEIGHTS}" != http://* ]] && [[ "${DINO_WEIGHTS}" != https://* ]]; then
    DINO_WEIGHTS="file://${DINO_WEIGHTS}"
fi

mkdir -p "${SUBSET_ROOT}" "${OUT_ROOT}"

echo "=== Hybrid ablation suite ==="
echo "  REPO_ROOT        : ${REPO_ROOT}"
echo "  SOURCE_DATASET   : ${SOURCE_DATASET}"
echo "  SUBSET_ROOT      : ${SUBSET_ROOT}"
echo "  OUT_ROOT         : ${OUT_ROOT}"
echo "  TOTAL_IMAGES     : ${TOTAL_IMAGES}"
echo "  IMAGES_PER_CLASS : ${IMAGES_PER_CLASS}"
echo "  VARIANTS         : ${VARIANTS[*]}"
echo "  DINO_WEIGHTS     : ${DINO_WEIGHTS}"

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

run_variant() {
    local variant="$1"
    local keep_topk=12
    local heatmap_top_k=12
    local heatmap_percentile=85
    local heatmap_crop_sizes="0.25,0.35,0.5"
    local run_dir="${OUT_ROOT}/${variant}_${TOTAL_IMAGES}"

    case "${variant}" in
      baseline)
        ;;
      hp90)
        heatmap_percentile=90
        ;;
      hp80)
        heatmap_percentile=80
        ;;
      topk8)
        keep_topk=8
        heatmap_top_k=8
        ;;
      tightcrop)
        heatmap_crop_sizes="0.2,0.3,0.4"
        ;;
      *)
        echo "ERROR: Unknown VARIANT '${variant}'"
        return 1
        ;;
    esac

    mkdir -p "${run_dir}"

    echo
    echo "=== Running variant: ${variant} ==="
    echo "  RUN_DIR             : ${run_dir}"
    echo "  HEATMAP_PERCENTILE  : ${heatmap_percentile}"
    echo "  HEATMAP_TOP_K       : ${heatmap_top_k}"
    echo "  KEEP_TOPK           : ${keep_topk}"
    echo "  HEATMAP_CROP_SIZES  : ${heatmap_crop_sizes}"

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
        --out-dir "${run_dir}" \
        --multi-crop \
        --ms-preset "${MS_PRESET}" \
        --crop-batch-size "${CROP_BATCH_SIZE}" \
        --crop-n "${CROP_N}" \
        --crop-keep-per-window "${CROP_KEEP_PER_WINDOW}" \
        --primary-output "${PRIMARY_OUTPUT}" \
        --log-every "${LOG_EVERY}" \
        --keep-topk "${keep_topk}" \
        --heatmap-top-k "${heatmap_top_k}" \
        --heatmap-percentile "${heatmap_percentile}" \
        --heatmap-crop-sizes "${heatmap_crop_sizes}" \
        --heatmap-spatial-rescue "${HEATMAP_SPATIAL_RESCUE}" \
        --merge-iou-thresh "${MERGE_IOU_THRESH}"
}

for variant in "${VARIANTS[@]}"; do
    run_variant "${variant}"
done

echo
echo "Hybrid ablation suite done. Outputs saved under ${OUT_ROOT}"
