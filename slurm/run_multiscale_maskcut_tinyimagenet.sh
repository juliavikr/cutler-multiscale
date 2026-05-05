#!/bin/bash

#SBATCH --job-name=maskcut-multiscale
# TODO: set your SLURM account — export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/maskcut_multiscale_%j.out
#SBATCH --error=logs/maskcut_multiscale_%j.err

set -euo pipefail

# Environment
module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${HOME}/cutler-multiscale"
cd "${REPO_ROOT}"

ANNO_DIR="${HOME}/data/tiny-imagenet-10classes/annotations"
mkdir -p "${ANNO_DIR}"

TAU=0.15
FIXED_SIZE=480
N=3
PRESET="small"

echo "=== Running multi-scale MaskCut on 10-class TinyImageNet ==="
python multiscale/multiscale_maskcut.py \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N}" \
    --num-folder-per-job 10 \
    --job-index 0 \
    --dataset-path "${HOME}/data/tiny-imagenet-10classes/train_flat/" \
    --pretrain_path "${DATA_ROOT:-${HOME}/data}/weights/dino_deitsmall8_pretrain.pth" \
    --out-dir "${ANNO_DIR}" \
    --multi-crop \
    --ms-preset "${PRESET}" \
    --crop-batch-size 8 \
    --small-first \
    --heatmap-top-k 4

# Rename to canonical output name.
# The preset-based filename includes the preset tag; find it dynamically.
GENERATED=$(ls "${ANNO_DIR}"/imagenet_train_fixsize${FIXED_SIZE}_tau${TAU}_N${N}*multiscale*.json 2>/dev/null | head -1)
FINAL="${ANNO_DIR}/tinyimagenet_10c_multiscale_pseudo.json"

if [[ -z "${GENERATED}" ]]; then
    echo "ERROR: Could not find generated JSON in ${ANNO_DIR}"
    ls "${ANNO_DIR}/"
    exit 1
fi

mv "${GENERATED}" "${FINAL}"
echo "=== Done ==="
echo "Pseudo-labels saved to ${FINAL}"
