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
├── CutLER/                        # Upstream CutLER repo (Facebook Research) — do not edit directly
├── multiscale/                    # Our custom code: multi-scale MaskCut extension
│   ├── multiscale_maskcut.py      # Main implementation (v2, current)
│   ├── multiscale_maskcut_hybrid.py  # Heatmap-only snapshot (ablation reference)
│   ├── multiscale_maskcut_legacy.py  # v1 IoU-NMS snapshot (for reference)
│   ├── MULTISCALE_MASKCUT.md      # Code guide and CLI reference
│   ├── STRATEGY_COMPARISON.md    # Comparison of all crop proposal strategies
│   └── EVALUATION_PROCESS.md     # Evaluation methodology and metrics
├── tools/                         # Utility scripts (visualize, register, train wrapper)
├── experiments/                   # environment.yml, rank_small_ap.py
├── presentation/                  # Report draft, slides, design decisions
├── results/                       # Results tables committed; large eval outputs gitignored
├── slurm/                         # SLURM job scripts for the HPC cluster
├── logs/                          # SLURM stdout/stderr logs (gitignored except .gitkeep)
├── debug/                         # Single-image debug outputs (contact sheets, overlays)
├── CLAUDE.md                      # This file
├── PROJECT_NOTES.md               # Status, parameters, results, audit trail
├── PROJECT_OVERVIEW.md            # Plain-English pipeline explanation
└── README.md                      # Setup, reproduction, and run commands
```

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `baseline` | Reproducing original CutLER results |
| `multiscale` | Multi-scale MaskCut development |

## Compute: Bocconi HPC Cluster

All training and heavy computation runs on the Bocconi University HPC cluster.
**Do not run training or pseudo-label generation on the local Mac.**

| Setting | Value |
|---------|-------|
| Login node | `slogin.hpc.unibocconi.it` |
| Username | `3355142` |
| Partition | `stud` |
| QOS | `stud` |
| SLURM account | `3355142` |
| GPU | NVIDIA A100 80GB |
| Conda env | `cutler` (Python 3.9, CUDA 12.1, PyTorch) |

### Development Workflow

```
[Mac — write code with Claude Code]
        │
        │  git push origin <branch>
        ▼
[GitHub — cutler-multiscale]
        │
        │  git pull  (on cluster)
        ▼
[HPC cluster — sbatch slurm/<script>.sh]
```

1. Write and edit code locally on Mac using Claude Code
2. `git push` to GitHub
3. SSH into the cluster: `ssh 3355142@slogin.hpc.unibocconi.it`
4. `git pull` on the cluster to get the latest code
5. Submit jobs: `sbatch slurm/run_maskcut.sh`, etc.
6. Monitor: `squeue -u 3355142`
7. Fetch results/logs back to Mac as needed

### Key Cluster Paths

```bash
# Project root on cluster
PROJECT=${HOME}/cutler-multiscale

# Data
DATA_ROOT=${HOME}/data               # large files live here

# Activate env
module load miniconda3
conda activate cutler
```

### Cluster Environment (installed 2026-04-20)

- Python 3.9
- CUDA 12.1
- PyTorch (CUDA 12.1 build)
- Detectron2
- pycocotools, scipy, scikit-image, pydensecrf, timm, faiss-gpu

## Key Dependencies

- Python 3.9
- PyTorch / CUDA 12.1
- Detectron2 (Facebook Research)
- DINO ViT-S/8 pretrained weights
- OpenCV, scikit-image, scipy, pydensecrf

## CutLER Pipeline Summary

1. **MaskCut** — iterative normalized cuts on DINO self-attention maps → binary pseudo-masks
2. **CascadedMasks** — multiple rounds of MaskCut, background suppression between rounds
3. **Detector Training** — train a Mask R-CNN on the pseudo-labeled dataset
4. **Self-Training** — iterative refinement with model-predicted masks as new pseudo-labels

## Multi-Scale Extension Plan

- Build an image pyramid (scales: 0.5×, 1×, 1.5×, 2×)
- Run MaskCut independently at each scale
- Back-project proposals to original image coordinates
- Merge via Soft-NMS / WBF, weighting by saliency score
- Goal: recover small objects missed at the native 1× scale

## Running Jobs on the Cluster

```bash
# After ssh + git pull:
cd cutler-multiscale

# Pseudo-label generation (MaskCut)
sbatch slurm/run_maskcut.sh

# Detector training (single A100)
sbatch slurm/run_training.sh

# COCO evaluation
sbatch slurm/run_eval.sh

# Monitor jobs
squeue -u 3355142

# Tail logs
tail -f logs/maskcut_<jobid>.out
```

## Notes for Claude

- `CutLER/` is a git submodule; do not commit changes inside it unless intentional
- Our custom code lives in `multiscale/` and `experiments/`
- All compute happens on the HPC cluster — do not suggest running training locally
- Prefer small, focused commits tied to experiment milestones
- Keep `PROJECT_NOTES.md` updated with phase, blockers, and results
- SLURM scripts in `slurm/` use `DATA_ROOT` env var for dataset paths — set before submitting
