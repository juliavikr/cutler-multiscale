#!/bin/bash

#SBATCH --job-name=maskcut-baseline
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/maskcut_baseline_%j.out
#SBATCH --error=logs/maskcut_baseline_%j.err

set -euo pipefail

# Environment
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${HOME}/cutler-multiscale"
cd "${REPO_ROOT}/CutLER/maskcut"

ANNO_DIR="${HOME}/data/tiny-imagenet/annotations"
mkdir -p "${ANNO_DIR}"

echo "=== Running MaskCut baseline on TinyImageNet train ==="
python maskcut.py \
    --vit-arch small \
    --patch-size 8 \
    --tau 0.15 \
    --fixed_size 480 \
    --N 3 \
    --num-folder-per-job 200 \
    --job-index 0 \
    --dataset-path "${HOME}/data/tiny-imagenet/tiny-imagenet-200/train/" \
    --pretrain_path "${HOME}/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth" \
    --out-dir "${ANNO_DIR}"

echo "=== Merging per-folder JSONs ==="
# merge_jsons.py loops up to 1000 (ImageNet-1K default); warnings about
# missing folders 200-999 are expected and harmless for TinyImageNet.
python merge_jsons.py \
    --base-dir "${ANNO_DIR}" \
    --save-path "${ANNO_DIR}/tinyimagenet_train_baseline_pseudo.json" \
    --num-folder-per-job 200 \
    --fixed-size 480 \
    --tau 0.15 \
    --N 3

echo "=== Done ==="
echo "Pseudo-labels saved to ${ANNO_DIR}/tinyimagenet_train_baseline_pseudo.json"
