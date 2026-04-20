#!/bin/bash

#SBATCH --job-name=cutler-download-ckpt
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_checkpoint_%j.out
#SBATCH --error=logs/download_checkpoint_%j.err

set -euo pipefail

CKPT_DIR="${HOME}/cutler-multiscale/checkpoints"
ANNO_DIR="${HOME}/data/coco/annotations"

mkdir -p "${CKPT_DIR}" "${ANNO_DIR}"

echo "=== Downloading CutLER Cascade Mask R-CNN checkpoint ==="
wget -c --progress=bar:force \
    http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth \
    -O "${CKPT_DIR}/cutler_cascade_final.pth"

echo "=== Downloading pre-generated MaskCut annotations ==="
wget -c --progress=bar:force \
    http://dl.fbaipublicfiles.com/cutler/maskcut/imagenet_train_fixsize480_tau0.15_N3.json \
    -O "${ANNO_DIR}/imagenet_train_fixsize480_tau0.15_N3.json"

echo "=== Verifying downloads ==="
for FILE in \
    "${CKPT_DIR}/cutler_cascade_final.pth" \
    "${ANNO_DIR}/imagenet_train_fixsize480_tau0.15_N3.json"
do
    if [ -f "${FILE}" ]; then
        SIZE=$(du -sh "${FILE}" | cut -f1)
        echo "  OK  ${SIZE}  ${FILE}"
    else
        echo "  MISSING: ${FILE}"
        exit 1
    fi
done

echo ""
echo "Done. Checkpoint: ${CKPT_DIR}/cutler_cascade_final.pth"
echo "      Annotations: ${ANNO_DIR}/imagenet_train_fixsize480_tau0.15_N3.json"
