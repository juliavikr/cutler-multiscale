# Project Notes — cutler-multiscale

## Current Status: Phase 1 — Setup

**Date started:** 2026-04-20

---

## Compute Environment

**Cluster:** Bocconi University HPC (`slogin.hpc.unibocconi.it`)
**Username:** 3355142
**GPU:** NVIDIA A100 80GB
**Scheduler:** SLURM — partition `stud`, qos `stud`, account `3355142`
**Conda env:** `cutler` — Python 3.9, CUDA 12.1, PyTorch

**Local Mac:** Code editing only via Claude Code. No training or heavy compute.

**Workflow:** edit locally → `git push` → `ssh` into cluster → `git pull` → `sbatch`

---

## Phase Log

### Phase 1: Setup (current)
- [x] Cloned CutLER upstream repo as git submodule into `CutLER/`
- [x] Created project structure (CLAUDE.md, .gitignore, branches)
- [x] Created SLURM scripts for cluster (setup, maskcut, training, eval)
- [x] Installed CutLER conda environment on cluster (2026-04-20) — `setup_env.sh`
- [ ] Confirm cluster data paths (`DATA_ROOT`) and update SLURM scripts
- [ ] Download DINO ViT-S/8 weights onto cluster
- [ ] Verify CutLER baseline runs end-to-end on a small data sample on cluster
- [ ] Document exact installed package versions in `experiments/environment.yml`

### Phase 2: Baseline Reproduction (upcoming)
- [ ] Run MaskCut pseudo-label generation on TinyImageNet (`sbatch slurm/run_maskcut.sh`)
- [ ] Train Cascade Mask R-CNN detector on pseudo-labels (`sbatch slurm/run_training.sh`)
- [ ] Evaluate on COCO val2017 — record AP, AP50, AP75, APs, APm (`sbatch slurm/run_eval.sh`)
- [ ] Compare to reported numbers in CutLER paper

### Phase 3: Multi-Scale MaskCut (upcoming)
- [ ] Implement image pyramid construction
- [ ] Run MaskCut at each scale
- [ ] Implement multi-scale proposal merging (Soft-NMS or WBF)
- [ ] Regenerate pseudo-labels with multi-scale method
- [ ] Retrain detector, evaluate — focus on APs (small object AP)

### Phase 4: Analysis & Write-up (upcoming)
- [ ] Ablation: effect of each scale, merging strategy, number of iterations
- [ ] Visualization of recovered small objects
- [ ] Final results table
- [ ] Course report

---

## Blockers / Open Questions

- Need to confirm actual `DATA_ROOT` path on cluster (where datasets are stored)
- Need to confirm whether TinyImageNet is already available on the cluster or needs downloading

---

## Results Tracker

| Method | AP | AP50 | AP75 | APs | APm | Notes |
|--------|----|------|------|-----|-----|-------|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | COCO val, unsupervised |
| CutLER (ours) | — | — | — | — | — | to be filled |
| MS-MaskCut (ours) | — | — | — | — | — | to be filled |

---

## Environment (cluster — installed 2026-04-20)

| Package | Version |
|---------|---------|
| Python | 3.9 |
| CUDA | 12.1 |
| PyTorch | TBD (CUDA 12.1 build) |
| Detectron2 | TBD |
| pycocotools | 2.0.6 |
| scipy | latest |
| scikit-image | 0.19.2 |
| timm | 0.5.4 |
| pydensecrf | latest (from source) |

_Fill in exact versions after running `conda list` on the cluster._
