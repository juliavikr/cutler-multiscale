#!/bin/bash
#SBATCH --job-name=maskcut_baseline_coco
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT=$HOME/cutler-multiscale
DATA_ROOT=/mnt/beegfsstudents/home/3152697
BACKBONE=$DATA_ROOT/weights/dino_deitsmall8_pretrain.pth
DATASET=$HOME/data/coco_val500_wrapped
OUT_DIR=$DATA_ROOT/coco/pseudo_labels/coco_baseline_pseudo
ANNO_OUT=$DATA_ROOT/coco/annotations/coco_baseline_pseudo.json

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

echo "=== Baseline MaskCut on COCO val500 ==="
echo "Start: $(date)"

cd $PROJECT/CutLER/maskcut

python maskcut.py \
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
    --cpu-only False

echo "MaskCut done: $(date)"

# ── Convert output to COCO JSON ───────────────────────────────────────────────
# maskcut.py writes one JSON per class-folder; merge into single annotation file
python $PROJECT/tools/merge_maskcut_output.py \
    --input-dir $OUT_DIR \
    --output $ANNO_OUT \
    --image-root $HOME/data/coco_val500

echo "Annotation saved to $ANNO_OUT"
echo "=== Done: $(date) ==="
