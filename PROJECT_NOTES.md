# Project Notes — cutler-multiscale

## Current Status: Phase 3 — Multi-Scale MaskCut (in progress)

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

### Phase 1: Setup (complete)
- [x] Cloned CutLER upstream repo as git submodule into `CutLER/`
- [x] Created project structure (CLAUDE.md, .gitignore, branches)
- [x] Created SLURM scripts for cluster (maskcut, training, eval, data download)
- [x] Installed CutLER conda environment on cluster (2026-04-20)
      — `setup_env.sh` (initial attempt, broken) superseded by `install_detectron2.sh`
      — Working method: miropsota pre-built wheels (torch 2.5.1+cu121,
        detectron2==0.6+fd27788pt2.5.0cu121, numpy<2)
      — Building detectron2 from source fails with torch 2.x (removed amp APIs)
- [x] Confirmed cluster data paths; `DATA_ROOT` set to `${HOME}/data` in all SLURM scripts
- [x] Downloaded COCO val2017 images and annotations to `~/data/coco/`
- [x] Downloaded pre-trained CutLER checkpoint and pre-generated MaskCut annotations

### Phase 2: Baseline Reproduction (complete — 2026-04-27)
- [x] Downloaded pre-trained `cutler_cascade_final.pth` checkpoint
- [x] Downloaded pre-generated MaskCut annotations JSON (imagenet_train_fixsize480_tau0.15_N3)
- [x] Evaluated pre-trained model on COCO val2017 (`sbatch slurm/run_eval.sh`)
- [x] Recorded AP, AP50, AP75, APs, APm, APl for both bbox and segm (see Results Tracker)
- [x] Numbers match CutLER paper — reproducibility confirmed

### Phase 3: Multi-Scale MaskCut (upcoming)
- [x] Moved custom multi-scale MaskCut code/docs to top-level `multiscale/` so it can be committed in the parent repo
- [x] Implement image pyramid construction (crop scales 1.0, 0.5)
- [x] Run MaskCut at each scale
- [x] Implement multi-scale proposal merging (IoU-based NMS)
- [x] Regenerate pseudo-labels with multi-scale method on TinyImageNet-5 subset
      — 2500 images, 24771 annotations (~9.9 masks/image)
      — Output: pseudo_masks/tiny_imagenet/imagenet_train_fixsize480_tau0.2_N2_mc1.0-0.5_ov0.3_miou0.5_0_5.json
      — Runtime: ~4h on A100 (gnode02), job 484052, 2026-04-28
- [ ] Retrain detector on pseudo-labels, evaluate on COCO val2017 small+medium
- [ ] Compare APs/APm vs CutLER baseline

### Phase 4: Analysis & Write-up (upcoming)
- [ ] Ablation: effect of each scale, merging strategy, number of iterations
- [ ] Visualization of recovered small objects
- [ ] Final results table
- [ ] Course report

---

## Blockers / Open Questions

- TinyImageNet has extra `images/` subdirectory per class — patched in multiscale_maskcut.py
- SLURM log paths must be absolute — patched in run_multiscale_maskcut.sh
- REPO_ROOT must be hardcoded — patched in run_multiscale_maskcut.sh
- dino.py unconditionally calls torch.hub for weights — patched to check os.path.isfile first
- Full TinyImageNet (100k images) too slow at ~6s/img — using 5-class subset for now
- Added incremental JSON checkpoint after each class folder to survive timeouts

---

## Results Tracker

Evaluated on COCO val2017, class-agnostic, unsupervised (no labels used).

### Bounding Box (BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — | reported in paper |
| CutLER (ours) | **12.33** | **21.98** | **11.90** | **3.66** | **12.72** | **29.60** | cutler_cascade_final.pth, 2026-04-27 |
| MS-MaskCut (ours) | — | — | — | — | — | — | to be filled |

### Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER (paper) | — | — | — | — | — | — | not reported separately |
| CutLER (ours) | **9.78** | **18.92** | **9.19** | **2.44** | **8.77** | **24.29** | cutler_cascade_final.pth, 2026-04-27 |
| MS-MaskCut (ours) | — | — | — | — | — | — | to be filled |

---

## Environment (cluster — installed 2026-04-20)

| Package | Version |
|---------|---------|
| Python | 3.9 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Detectron2 | 0.6+fd27788pt2.5.0cu121 (miropsota wheel) |
| pycocotools | 2.0.6 |
| scipy | latest |
| scikit-image | 0.19.2 |
| timm | 0.5.4 |
| pydensecrf | latest (from source) |

| numpy | <2 (pinned — detectron2 0.6 incompatible with numpy 2.x) |

_Note: detectron2 must be installed from miropsota pre-built wheels. Building from source against torch 2.x fails due to removed `torch.cuda.amp` APIs. See `slurm/install_detectron2.sh`._
