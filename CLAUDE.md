# cutler-multiscale

## Project Overview

Computer vision course project implementing CutLER as a baseline and extending it with Multi-Scale MaskCut for improved small-object detection in unsupervised/self-supervised settings.

## Goals

1. **Baseline**: Reproduce CutLER results using the original implementation
2. **Improvement**: Replace single-scale MaskCut with a multi-scale variant that merges proposals across image pyramid levels
3. **Evaluation**: Benchmark on COCO and potentially a small-object-heavy dataset (e.g., VisDrone, DOTA)

## Repository Structure

```
cutler-multiscale/
├── CutLER/                  # Upstream CutLER repo (Facebook Research) — do not edit directly
├── multiscale/              # Our custom code: multi-scale MaskCut extension
│   ├── maskcut_ms.py        # Multi-scale MaskCut implementation
│   ├── merge_proposals.py   # NMS/merging logic across scales
│   └── ...
├── experiments/             # Config files and run scripts
├── notebooks/               # Analysis and visualization notebooks
├── results/                 # Evaluation outputs (gitignored)
├── CLAUDE.md                # This file
├── PROJECT_NOTES.md         # Status and running log
└── README.md
```

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `baseline` | Reproducing original CutLER results |
| `multiscale` | Multi-scale MaskCut development |

## Key Dependencies

- Python 3.9+
- PyTorch 1.12+ / CUDA 11.3+
- Detectron2 (Facebook Research)
- DINO ViT-S/8 pretrained weights
- OpenCV, scikit-image, scipy

## CutLER Pipeline Summary

1. **MaskCut** — iterative normalized cuts on DINO self-attention maps → binary pseudo-masks
2. **CascadedMasks** — multiple rounds of MaskCut, background suppression between rounds
3. **Detector Training** — train a Mask R-CNN on the pseudo-labeled COCO-unlabeled set
4. **Self-Training** — iterative refinement with model-predicted masks as new pseudo-labels

## Multi-Scale Extension Plan

- Build an image pyramid (scales: 0.5×, 1×, 1.5×, 2×)
- Run MaskCut independently at each scale
- Back-project proposals to original image coordinates
- Merge via Soft-NMS / WBF, weighting by saliency score
- Goal: recover small objects missed at the native 1× scale

## Running the Baseline

See `CutLER/README.md` for full setup. Quick reference:

```bash
# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Download DINO weights
# (see CutLER/maskcut/README.md for exact commands)

# Run MaskCut pseudo-label generation
cd CutLER
python maskcut/maskcut.py --vit-arch small --patch-size 8 \
    --tau 0.15 --num-iter 3 \
    --dataset-path /path/to/coco/train2017 \
    --output-dir /path/to/output
```

## Notes for Claude

- `CutLER/` is a git submodule-style clone; do not commit changes inside it unless intentional
- Our custom code lives in `multiscale/` and `experiments/`
- Prefer small, focused commits tied to experiment milestones
- Keep `PROJECT_NOTES.md` updated with phase, blockers, and results
