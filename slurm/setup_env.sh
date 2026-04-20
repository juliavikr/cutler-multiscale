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
eval "$(conda shell.bash hook)"

# Create environment (skip if it already exists)
if conda info --envs | grep -q "^cutler "; then
    echo "Conda env 'cutler' already exists — skipping creation."
else
    conda create -y -n cutler python=3.9
fi

conda activate cutler

# 1. PyTorch + CUDA 12.1 (Bocconi A100 cluster)
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Detectron2 — build from source against the installed torch (no prebuilt wheel for cu121)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Remaining dependencies
pip install \
    scipy \
    pycocotools==2.0.6 \
    opencv-python==4.6.0.66 \
    timm==0.5.4 \
    scikit-image==0.19.2 \
    scikit-learn==1.1.1 \
    shapely==1.8.2 \
    faiss-gpu==1.7.2 \
    numpy==1.20.0 \
    pyyaml==6.0 \
    fvcore==0.1.5.post20220512 \
    colored==1.4.4 \
    gdown==4.5.4 \
    tqdm

# 4. Dense CRF (for MaskCut post-processing)
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

echo ""
echo "=== Verification ==="
python - <<'EOF'
import torch, sys
print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:   {torch.version.cuda}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")

import detectron2
print(f"Detectron2: {detectron2.__version__}")
EOF

echo ""
echo "Environment setup complete."
conda list | tee logs/cutler_env_packages.txt
