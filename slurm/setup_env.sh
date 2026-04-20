#!/bin/bash
# This script was already run manually on April 20 2026.
# Kept here for reproducibility — do not resubmit unless rebuilding from scratch.

#SBATCH --job-name=cutler-setup
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

set -euo pipefail

module load miniconda3

# Create environment (skip if it already exists)
if conda info --envs | grep -q "^cutler "; then
    echo "Conda env 'cutler' already exists — skipping creation."
else
    conda create -y -n cutler python=3.8
fi

conda activate cutler

# PyTorch + CUDA 11.6 (matches cog.yaml cuda version)
pip install torch==1.11.0+cu116 torchvision==0.12.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Detectron2 built against torch 1.11 / CUDA 11.6
pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.11/index.html

# Core vision + scientific stack
pip install \
    opencv-python==4.6.0.66 \
    scikit-image==0.19.2 \
    scikit-learn==1.1.1 \
    scipy \
    shapely==1.8.2 \
    timm==0.5.4 \
    faiss-gpu==1.7.2 \
    numpy==1.20.0

# COCO tools and utilities
pip install \
    pycocotools==2.0.6 \
    pyyaml==6.0 \
    fvcore==0.1.5.post20220512 \
    colored==1.4.4 \
    gdown==4.5.4 \
    tqdm

# Dense CRF (for MaskCut post-processing)
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

echo "Environment setup complete."
conda list | tee logs/cutler_env_packages.txt
