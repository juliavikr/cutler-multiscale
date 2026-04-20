#!/bin/bash
# Fixes AttributeError: module 'PIL.Image' has no attribute 'LINEAR'
# Pillow 10.0 removed Image.LINEAR (deprecated alias for Image.BILINEAR).
# Detectron2 0.6 still references it; pinning Pillow<10 is the fix.

#SBATCH --job-name=cutler-fix-pillow
#SBATCH --account=3355142
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fix_pillow_%j.out
#SBATCH --error=logs/fix_pillow_%j.err

set -euo pipefail

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate cutler

pip install "Pillow<10"

echo "=== Verification ==="
python -c "
import PIL
from PIL import Image
print(f'Pillow version: {PIL.__version__}')
# Confirm the attribute that detectron2 requires is present
assert hasattr(Image, 'LINEAR'), 'Image.LINEAR still missing — check Pillow version'
print('Image.LINEAR: OK')
"
