#!/bin/bash
#SBATCH --job-name=maskcut_multiscale_coco
#SBATCH --output=logs/maskcut_multiscale_coco_%j.out
#SBATCH --error=logs/maskcut_multiscale_coco_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --account=3355142

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT=${HOME}/cutler-multiscale
BACKBONE=${HOME}/data/weights/dino_deitsmall8_pretrain.pth
DATASET=${HOME}/data/coco_val500_wrapped
OUT_DIR=${HOME}/data/coco/pseudo_labels/coco_multiscale_pseudo
ANNO_OUT=${HOME}/data/coco/annotations/coco_multiscale_pseudo.json

# ── Setup ────────────────────────────────────────────────────────────────────
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate cutler

# Wrap flat COCO val500 images into a single subfolder so MaskCut's
# folder-per-class iteration finds them in one pass
mkdir -p ${HOME}/data/coco_val500_wrapped/coco
if [ ! "$(ls -A ${HOME}/data/coco_val500_wrapped/coco)" ]; then
    echo "Symlinking COCO val500 images into wrapped dir..."
    ln -s ${HOME}/data/coco_val500/*.jpg ${HOME}/data/coco_val500_wrapped/coco/
fi

mkdir -p $OUT_DIR
mkdir -p $(dirname $ANNO_OUT)
mkdir -p ${PROJECT}/logs

echo "=== Multiscale MaskCut on COCO val500 ==="
echo "Start: $(date)"

# Run hybrid heatmap-guided multi-scale MaskCut.
# --primary-output multiscale: write the crop-masks-only split (training-ready).
# --ms-preset small: heatmap_crop_sizes=0.25/0.35/0.50, topk=12, rescue=4.
# --max-mask-area-ratio 0.5: COCO images are ~640px; at fixed_size=480 a 0.5 cap
#   = 0.5 * 480^2 = 115200 px^2, which is well above any useful object.
python ${PROJECT}/multiscale/multiscale_maskcut.py \
    --vit-arch small \
    --patch-size 8 \
    --tau 0.15 \
    --fixed_size 480 \
    --pretrain_path $BACKBONE \
    --num-folder-per-job 1 \
    --job-index 0 \
    --dataset-path $DATASET \
    --out-dir $OUT_DIR \
    --N 3 \
    --multi-crop \
    --ms-preset small \
    --primary-output multiscale \
    --max-mask-area-ratio 0.5

echo "MaskCut done: $(date)"

# Merge per-folder JSON outputs into a single annotation file
python ${PROJECT}/CutLER/maskcut/merge_jsons.py \
    --input-dir $OUT_DIR \
    --output $ANNO_OUT

echo "Annotation saved to $ANNO_OUT"
echo "=== Done: $(date) ==="
