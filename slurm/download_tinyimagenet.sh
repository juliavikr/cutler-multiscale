#!/bin/bash

#SBATCH --job-name=cutler-download-tinyimagenet
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_tinyimagenet_%j.out
#SBATCH --error=logs/download_tinyimagenet_%j.err

set -euo pipefail

DATA_DIR="${HOME}/data/tiny-imagenet"
mkdir -p "${DATA_DIR}"

echo "=== Downloading Tiny ImageNet 200 ==="
wget -c --progress=bar:force \
    http://cs231n.stanford.edu/tiny-imagenet-200.zip \
    -O "${DATA_DIR}/tiny-imagenet-200.zip"

echo "=== Unzipping ==="
unzip -q "${DATA_DIR}/tiny-imagenet-200.zip" -d "${DATA_DIR}/"

echo "=== Cleaning up zip ==="
rm "${DATA_DIR}/tiny-imagenet-200.zip"

echo "=== Train folder structure (first 10 classes) ==="
ls "${DATA_DIR}/tiny-imagenet-200/train/" | head -10

echo "=== Class folder count ==="
N=$(ls "${DATA_DIR}/tiny-imagenet-200/train/" | wc -l)
echo "Found ${N} class folders in train/ (expected 200)"

echo ""
echo "Done. Tiny ImageNet is at ${DATA_DIR}/tiny-imagenet-200"
