#!/bin/bash
# Visualize pseudo-label masks from a COCO JSON.
# No GPU needed — CPU-only matplotlib rendering.
#
# Usage:
#   sbatch slurm/run_visualize.sh
#
# Output: experiments/visualizations/baseline/<image_id>_<n>masks.png

#SBATCH --job-name=visualize-masks
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err

set -euo pipefail

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${HOME}/cutler-multiscale"
JSON="${HOME}/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json"
IMAGE_ROOT="${HOME}/data/tiny-imagenet-10classes/train"
OUTPUT_DIR="${REPO_ROOT}/experiments/visualizations/baseline"

mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/tools/visualize_pseudo_masks.py" \
    --json        "${JSON}" \
    --image-root  "${IMAGE_ROOT}" \
    --output-dir  "${OUTPUT_DIR}" \
    --num-samples 20

echo ""
echo "Visualizations saved to: ${OUTPUT_DIR}"
