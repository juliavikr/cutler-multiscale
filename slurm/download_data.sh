#!/bin/bash

#SBATCH --job-name=cutler-download-data
# TODO: set your SLURM account — export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download_data_%j.out
#SBATCH --error=logs/download_data_%j.err

set -euo pipefail

DATA_DIR="${HOME}/data/coco"
mkdir -p "${DATA_DIR}"

echo "=== Downloading COCO val2017 images ==="
wget -c --progress=bar:force \
    http://images.cocodataset.org/zips/val2017.zip \
    -O "${DATA_DIR}/val2017.zip"

echo "=== Downloading COCO 2017 annotations ==="
wget -c --progress=bar:force \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
    -O "${DATA_DIR}/annotations_trainval2017.zip"

echo "=== Unzipping val2017 images ==="
unzip -q "${DATA_DIR}/val2017.zip" -d "${DATA_DIR}"

echo "=== Unzipping annotations ==="
unzip -q "${DATA_DIR}/annotations_trainval2017.zip" -d "${DATA_DIR}"

echo "=== Cleaning up zip files ==="
rm "${DATA_DIR}/val2017.zip" "${DATA_DIR}/annotations_trainval2017.zip"

echo "=== Final directory structure ==="
find "${DATA_DIR}" -maxdepth 2 | sort

echo ""
echo "Done. COCO data is at ${DATA_DIR}"
echo "Set DATA_ROOT=${HOME}/data before submitting training jobs."
