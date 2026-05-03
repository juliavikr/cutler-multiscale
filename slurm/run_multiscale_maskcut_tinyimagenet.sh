#!/bin/bash

#SBATCH --job-name=maskcut-multiscale
#SBATCH --account=3152697
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home/3152697/cutler-multiscale/logs/maskcut_multiscale_%j.out
#SBATCH --error=/home/3152697/cutler-multiscale/logs/maskcut_multiscale_%j.err

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

# Crop parameters — keep these in sync with the auto-generated filename below.
CROP_SCALES="1.0,0.75,0.5"
CROP_OVERLAP=0.3
MERGE_IOU=0.5
TAU=0.15
FIXED_SIZE=480
N=3

echo "=== Running multi-scale MaskCut on TinyImageNet train ==="
python multiscale/multiscale_maskcut.py \
    --vit-arch small \
    --patch-size 8 \
    --tau "${TAU}" \
    --fixed_size "${FIXED_SIZE}" \
    --N "${N}" \
    --num-folder-per-job 10 \
    --job-index 0 \
    --dataset-path "${HOME}/data/tiny-imagenet-10classes/train/" \
    --pretrain_path "${HOME}/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth" \
    --out-dir "${ANNO_DIR}" \
    --multi-crop \
    --crop-scales "${CROP_SCALES}" \
    --crop-overlap "${CROP_OVERLAP}" \
    --merge-iou-thresh "${MERGE_IOU}" \
    --crop-batch-size 8 \
    --small-first

# The script writes a filename derived from the hyperparameters; rename to the
# canonical output name.  The scales string uses hyphens in the filename.
SCALES_TAG="${CROP_SCALES//,/-}"
GENERATED="${ANNO_DIR}/imagenet_train_fixsize${FIXED_SIZE}_tau${TAU}_N${N}_mc${SCALES_TAG}_ov${CROP_OVERLAP}_miou${MERGE_IOU}.json"
FINAL="${ANNO_DIR}/tinyimagenet_10c_multiscale_pseudo.json"

mv "${GENERATED}" "${FINAL}"

echo "=== Done ==="
echo "Pseudo-labels saved to ${FINAL}"
