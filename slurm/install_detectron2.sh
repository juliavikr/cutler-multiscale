#!/bin/bash
# Working install method confirmed on Bocconi HPC cluster (2026-04-20).
# Uses miropsota third-party pre-built wheels — building detectron2 from source
# against torch 2.x fails due to removed APIs; miropsota wheels are the fix.

#SBATCH --job-name=cutler-install-d2
# TODO: set your SLURM account — export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_detectron2_%j.out
#SBATCH --error=logs/install_detectron2_%j.err

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

# 1. PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Detectron2 0.6 pre-built against torch 2.5.0 / CUDA 12.1 (miropsota wheels)
#    Building from source fails with torch 2.x due to removed torch.cuda.amp APIs.
pip install detectron2==0.6+fd27788pt2.5.0cu121 \
    -f https://miropsota.github.io/torch_packages_builder/

# 3. numpy<2 — detectron2 0.6 is incompatible with numpy 2.x
pip install "numpy<2"

# 4. Remaining dependencies
pip install \
    scipy \
    pycocotools==2.0.6 \
    opencv-python==4.6.0.66 \
    timm==0.5.4 \
    scikit-image==0.19.2 \
    scikit-learn==1.1.1 \
    shapely==1.8.2 \
    pyyaml==6.0 \
    fvcore==0.1.5.post20220512 \
    colored==1.4.4 \
    gdown==4.5.4 \
    tqdm

# 5. Dense CRF (for MaskCut post-processing)
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

echo ""
echo "=== Verification ==="
python - <<'EOF'
import sys, torch, numpy as np
print(f"Python:         {sys.version.split()[0]}")
print(f"PyTorch:        {torch.__version__}")
print(f"numpy:          {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:   {torch.version.cuda}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")

import detectron2
print(f"Detectron2:     {detectron2.__version__}")
EOF

echo ""
echo "Environment setup complete."
conda list | tee logs/cutler_env_packages.txt
