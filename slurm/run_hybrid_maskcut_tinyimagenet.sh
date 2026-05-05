#!/bin/bash

# Hybrid heatmap multi-scale MaskCut on the same 10-class TinyImageNet subset as
# slurm/run_maskcut_baseline.sh. Identical locked params (ViT-S/8, tau=0.15, N=3, fixed_size=480,
# DINO weights). Only difference: --multi-crop --ms-preset small.
# This produces the multiscale training pseudo-labels for the report's main comparison.
# Output: ~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_hybrid_pseudo.json

#SBATCH --job-name=hybrid-maskcut
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hybrid_maskcut_%j.out
#SBATCH --error=logs/hybrid_maskcut_%j.err

set -euo pipefail

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${HOME}/cutler-multiscale"
cd "${REPO_ROOT}"

ANNO_DIR="${HOME}/data/tiny-imagenet-10classes/annotations"
mkdir -p "${ANNO_DIR}"

echo "=== Running hybrid heatmap multi-scale MaskCut on 10-class TinyImageNet ==="
python multiscale/multiscale_maskcut_hybrid.py \
    --multi-crop \
    --ms-preset small \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau 0.15 \
    --N 3 \
    --fixed_size 480 \
    --pretrain_path "${HOME}/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth" \
    --dataset-path "${HOME}/data/tiny-imagenet-10classes/train/" \
    --num-folder-per-job 10 \
    --job-index 0 \
    --out-dir "${ANNO_DIR}"

# The hybrid script writes multiple split JSONs; _multiscale.json is the canonical
# training-ready output (filtered/merged crop masks only, no full-image masks).
GENERATED=$(ls -t "${ANNO_DIR}"/*_multiscale.json | head -1)
FINAL="${ANNO_DIR}/tinyimagenet_10c_hybrid_pseudo.json"
mv "${GENERATED}" "${FINAL}"

echo "=== Done ==="
echo "Pseudo-labels saved to ${FINAL}"
