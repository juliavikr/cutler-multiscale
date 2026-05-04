#!/bin/bash

#SBATCH --job-name=maskcut-baseline
#SBATCH --account=3152697
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3152697/cutler-multiscale/logs/maskcut_baseline_%j.out
#SBATCH --error=/home/3152697/cutler-multiscale/logs/maskcut_baseline_%j.err

# =============================================================================
# Purpose: Baseline (single-scale) MaskCut on 10-class TinyImageNet subset.
#
# Locked parameters — MUST match slurm/run_multiscale_maskcut_tinyimagenet.sh
# for the comparison to be valid. See PROJECT_NOTES.md "Locked Experiment
# Parameters" before changing anything below.
#
#   --vit-arch  small     DINO ViT-Small backbone
#   --vit-feat  k         key features from self-attention
#   --patch-size 8        8×8 pixel patches
#   --tau       0.15      affinity graph threshold
#   --N         3         max masks per image
#   --fixed_size 480      resize input to 480×480 square
#   --pretrain_path       dino_deitsmall8_300ep_pretrain.pth
# =============================================================================

set -euo pipefail

# Environment
module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

REPO_ROOT="${HOME}/cutler-multiscale"
cd "${REPO_ROOT}/CutLER/maskcut"

ANNO_DIR="${HOME}/data/tiny-imagenet-10classes/annotations"
mkdir -p "${ANNO_DIR}"

echo "=== Running baseline (single-scale) MaskCut on 10-class TinyImageNet ==="
python maskcut.py \
    --vit-arch small \
    --vit-feat k \
    --patch-size 8 \
    --tau 0.15 \
    --N 3 \
    --fixed_size 480 \
    --pretrain_path "/mnt/beegfsstudents/home/3152697/weights/dino_deitsmall8_pretrain.pth" \
    --dataset-path "${HOME}/data/tiny-imagenet-10classes/train/" \
    --num-folder-per-job 10 \
    --job-index 0 \
    --out-dir "${ANNO_DIR}"

# maskcut.py auto-names the output based on params; standardise to a fixed name.
# With num-folder-per-job == len(dataset folders), it writes the no-suffix form:
#   imagenet_train_fixsize480_tau0.15_N3.json
GENERATED="${ANNO_DIR}/imagenet_train_fixsize480_tau0.15_N3.json"
FINAL="${ANNO_DIR}/tinyimagenet_10c_baseline_pseudo.json"
mv "${GENERATED}" "${FINAL}"

echo "=== Done ==="
echo "Pseudo-labels saved to ${FINAL}"
