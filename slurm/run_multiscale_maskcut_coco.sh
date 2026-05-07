#!/bin/bash
#SBATCH --job-name=maskcut_multiscale_coco
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT=$HOME/cutler-multiscale
DATA_ROOT=/mnt/beegfsstudents/home/3152697
BACKBONE=$DATA_ROOT/weights/dino_deitsmall8_pretrain.pth
DATASET=$HOME/data/coco_val500_wrapped
OUT_DIR=$DATA_ROOT/coco/pseudo_labels/coco_multiscale_pseudo
ANNO_OUT=$DATA_ROOT/coco/annotations/coco_multiscale_pseudo.json

# ── Setup ────────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate cutler

# Create the wrapped flat dir if it doesn't already exist
mkdir -p $HOME/data/coco_val500_wrapped/coco
if [ ! "$(ls -A $HOME/data/coco_val500_wrapped/coco)" ]; then
    echo "Symlinking COCO val500 images into wrapped dir..."
    ln -s $HOME/data/coco_val500/*.jpg $HOME/data/coco_val500_wrapped/coco/
fi

mkdir -p $OUT_DIR

echo "=== Multiscale MaskCut on COCO val500 ==="
echo "Start: $(date)"

cd $PROJECT/CutLER/maskcut

# NOTE on --max-mask-area-ratio:
#   COCO images are ~640px. At fixed_size=480, a 0.02 cap = 0.02 * 480^2 = 4608 px^2.
#   COCO "small" threshold is area < 32^2 = 1024 px^2.
#   So 0.02 is reasonable for COCO (won't incorrectly cap small objects).
#   This is different from TinyImageNet (64px originals → cap was 81 px^2 — too tight).

python maskcut_multiscale.py \
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
    --scales 1.0 0.75 0.5 \
    --max-mask-area-ratio 0.5 \
    --cpu-only False

echo "MaskCut done: $(date)"

# ── Convert output to COCO JSON ───────────────────────────────────────────────
python $PROJECT/tools/merge_maskcut_output.py \
    --input-dir $OUT_DIR \
    --output $ANNO_OUT \
    --image-root $HOME/data/coco_val500

echo "Annotation saved to $ANNO_OUT"
echo "=== Done: $(date) ==="
