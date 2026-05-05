#!/bin/bash
# Train CutLER Cascade Mask R-CNN on TinyImageNet pseudo-labels.
#
# Usage:
#   PSEUDO_LABEL_NAME=baseline  sbatch slurm/run_training.sh
#   PSEUDO_LABEL_NAME=hybrid    sbatch slurm/run_training.sh
#
# PSEUDO_LABEL_NAME selects the annotation JSON and names the output directory.
# Tip: pass --job-name to sbatch for descriptive log filenames, e.g.:
#   PSEUDO_LABEL_NAME=baseline sbatch --job-name=training_baseline slurm/run_training.sh
#   → logs/training_training_baseline_<jobid>.out
# (SLURM does not expand shell variables in #SBATCH directives; %x = job name, %j = job ID.)

#SBATCH --job-name=cutler-train
# TODO: set your SLURM account — export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/training_%x_%j.out
#SBATCH --error=logs/training_%x_%j.err

set -euo pipefail

# --- Validate required env var ---
if [[ -z "${PSEUDO_LABEL_NAME:-}" ]]; then
    echo "ERROR: PSEUDO_LABEL_NAME is not set."
    echo "Usage: PSEUDO_LABEL_NAME=baseline sbatch slurm/run_training.sh"
    exit 1
fi

# --- Environment ---
source /software/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cutler

# --- Paths ---
export DETECTRON2_DATASETS="${HOME}/data"
ANNO_DIR="${HOME}/data/tiny-imagenet-10classes/annotations"

case "${PSEUDO_LABEL_NAME}" in
    baseline)
        PSEUDO_JSON="${ANNO_DIR}/tinyimagenet_10c_baseline_pseudo.json"
        ;;
    hybrid)
        PSEUDO_JSON="${ANNO_DIR}/tinyimagenet_10c_multiscale_pseudo.json"
        ;;
    *)
        echo "ERROR: Unknown PSEUDO_LABEL_NAME '${PSEUDO_LABEL_NAME}'. Expected 'baseline' or 'hybrid'."
        exit 1
        ;;
esac

OUTPUT_DIR="${HOME}/cutler-multiscale/experiments/training_${PSEUDO_LABEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=== CutLER training ==="
echo "  PSEUDO_LABEL_NAME : ${PSEUDO_LABEL_NAME}"
echo "  PSEUDO_JSON       : ${PSEUDO_JSON}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  SLURM_JOB_ID      : ${SLURM_JOB_ID:-local}"

cd "${HOME}/cutler-multiscale/CutLER/cutler"

# train_wrapper.py pre-registers our TinyImageNet datasets then runs train_net.py
python "${HOME}/cutler-multiscale/tools/train_wrapper.py" \
    --num-gpus 1 \
    --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
    DATASETS.TRAIN "(\"tinyimagenet_${PSEUDO_LABEL_NAME}_pseudo\",)" \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.005 \
    SOLVER.MAX_ITER 8000 \
    SOLVER.STEPS "(6000,)" \
    SOLVER.WARMUP_ITERS 1000 \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR "${OUTPUT_DIR}"

