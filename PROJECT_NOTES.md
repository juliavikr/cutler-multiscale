# Project Notes — cutler-multiscale

## Current Status: Phase 3 — Multi-Scale MaskCut (in progress)

**Date started:** 2026-04-20

---

## Locked Experiment Parameters

Both the baseline (single-scale) and multi-scale MaskCut runs on TinyImageNet must use these identical parameters so the resulting pseudo-labels are directly comparable. Any change here invalidates the comparison and requires re-running both.

### Shared parameters (baseline + multiscale)

| Parameter | Value | Purpose |
| --- | --- | --- |
| --vit-arch | small | DINO ViT-Small backbone |
| --vit-feat | k | use key features from attention |
| --patch-size | 8 | 8×8 pixel patches |
| --tau | 0.15 | affinity graph threshold |
| --N | 3 | max masks discovered per image |
| --fixed_size | 480 | resize input to 480×480 square |
| --pretrain_path | ~/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth | DINO weights |

### Multi-scale-only parameters (added on top for multiscale run)

| Parameter | Value | Purpose |
| --- | --- | --- |
| --multi-crop | flag | enable multi-scale mode |
| --crop-scales | 1.0,0.75,0.5 | three zoom levels |
| --crop-overlap | 0.3 | sliding window overlap |
| --merge-iou-thresh | 0.5 | NMS IoU threshold for deduplication |
| --small-first | flag | prefer keeping small masks during NMS (helps APs) |

### Dataset

- Path: ~/data/tiny-imagenet-10classes/train/
- Size: 10 classes × 50 images = 500 images total
- Classes used (TinyImageNet WordNet IDs):
  - n01443537 (goldfish)
  - n02123045 (tabby cat)
  - n02281406 (sulphur butterfly)
  - n02410509 (bison)
  - n02906734 (broom)
  - n03100240 (convertible)
  - n03444034 (go-kart)
  - n04067472 (reel)
  - n04254777 (sock)
  - n07711569 (mashed potato)
- These were chosen for visual diversity across animals, vehicles, and household objects. Verified contents on cluster: 10 folders × 50 images = 500 total images at ~/data/tiny-imagenet-10classes/train/.

### Why 10 classes

Baseline MaskCut on the A100 takes ~6-7 sec/image. Multi-scale runs the algorithm on the full image plus crops at 0.75× and 0.5× scales, processing roughly 10× as many MaskCut calls per image.

Estimated runtimes:
- Baseline on 500 images: ~1 hour
- Multi-scale on 500 images: ~10 hours (fits within cluster's 24h time limit)
- Multi-scale on 2500 images (50 classes): ~50 hours (would not fit)

10 classes × 50 images is the largest TinyImageNet subset where both baseline and multi-scale can be run on the cluster, with margin for re-runs and parameter ablations. Detector training on 500 pseudo-labeled images is small but sufficient for a reproducible comparison.

### Output JSONs (target locations)

- Baseline: ~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json
- Multi-scale: ~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_multiscale_pseudo.json

### Generated artifacts

| Artifact | Status | Location (cluster) | Size | Images | Annotations | Generated |
|----------|--------|--------------------|------|--------|-------------|-----------|
| Baseline pseudo-labels | **exists** | `~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json` | 321 KB | 500 | 748 | 2026-05-01 |
| Multiscale pseudo-labels | **pending** | `~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_multiscale_pseudo.json` | — | — | — | pending Luiz's finalization |

These JSONs are not committed to Git (regeneratable, and excluded by `.gitignore`).
To recreate the baseline JSON: `sbatch slurm/run_maskcut_baseline.sh`

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

### Phase 3: Multi-Scale MaskCut (in progress)
- [x] Moved custom multi-scale MaskCut code/docs to top-level `multiscale/` so it can be committed in the parent repo
- [x] Corrected multi-scale crop logic to crop from the original image, then resize each crop for inference
- [x] Added crop batching and optional two-stage crop skipping to reduce repeated DINO forwards
- [x] Run single-image local debugging with JSON overlay visualization
- [x] Confirmed corrected implementation surfaces table / foreground objects that the old implementation missed
- [x] Keep split outputs for normal, raw multiscale, merged multiscale, and combined masks
- [x] Default primary multi-crop JSON to multiscale-only so combined masks are not used for training by accident
- [x] Add DINO feature-contrast heatmap crop proposal mode as an alternative to grid crops
- [x] Replace area-first crop selection with scored mask candidates and preset-based defaults
- [x] Emit candidate debug JSON with crop boxes, crop scores, mask scores, area, compactness, border touch, and overlap metadata
- [x] Implement image pyramid construction (crop scales 1.0, 0.5)
- [x] Run MaskCut at each scale
- [x] Implement multi-scale proposal merging (IoU-based NMS)
- [x] Regenerate pseudo-labels with multi-scale method on TinyImageNet-5 subset
      — 2500 images, 24771 annotations (~9.9 masks/image)
      — Output: pseudo_masks/tiny_imagenet/imagenet_train_fixsize480_tau0.2_N2_mc1.0-0.5_ov0.3_miou0.5_0_5.json
      — Runtime: ~4h on A100 (gnode02), job 484052, 2026-04-28
- [ ] Regenerate pseudo-labels with corrected original-crop projection / two-stage multi-crop job
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

## Recent Debugging Findings (2026-04-27)

- Initial local multi-scale implementation was functionally wrong for small-object recovery:
  it resized the full image to `fixed_size` first and only then cropped from that low-resolution image.
- This was corrected in `multiscale/multiscale_maskcut.py`:
  windows are still generated on a normalized grid, but each window is now mapped back to the original image, cropped there, resized to `fixed_size` for inference, then projected back to original coordinates.
- Before the fix, a strict run collapsed to a single coarse people-group foreground mask.
- After the fix, the same image produced a richer set of masks including several foreground/table objects, indicating the implementation is now directionally aligned with the project goal.
- Current merge logic is still heuristic and can produce partial or fragmented masks; further tuning or a stronger merge method is still needed.
- Open merge questions: IoU suppression vs Soft-NMS / weighted fusion, crop-scale thresholds, and combining original CutLER proposals with multi-scale proposals.
- Newer runs write four multi-crop views: `normal`, `raw_multiscale`, `multiscale`, and `combined`.
- Use `multiscale` as the training/evaluation candidate for now; `combined` is diagnostic until overlap hierarchy is handled better.
- Heatmap crop mode uses one full-image DINO feature-contrast pass to rank crop proposals before running crop MaskCut.
- Current default is `--ms-preset small`: heatmap crops, small-mask area cap, score-first merge/top-K, and `multiscale` as the primary output.
- The candidate scorer ranks crop masks by small-object area prior, compactness, crop score, CRF agreement, border touch, aspect ratio, and duplicate overlap with normal masks.
- Advanced thresholds remain available as overrides, but normal experiments should start from `--ms-preset small` or `--ms-preset balanced` instead of tuning every knob.

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

### Pseudo-label comparison (2026-04-29)

Single-scale vs multi-scale MaskCut on same 5 TinyImageNet classes (2500 images):

| Metric | Single-scale | Multi-scale |
|--------|-------------|-------------|
| Annotations | 3,315 | 24,771 |
| Masks/image | 1.33 | 9.91 |
| Mean area | 993.6 | 374.4 |
| Small (area<1024) | 1,875 (56.6%) | 23,433 (94.6%) |
| Medium (1024-9216) | 1,440 (43.4%) | 1,338 (5.4%) |

Multi-scale generates 7.5x more annotations, concentrated in small objects.

### Code Update: Graph-based merging + crop ranking (2026-05-01)

**Graph-based mask merging** (`merge_masks`, lines 380-486):
- Old: greedy NMS — if IoU > threshold, drop newer mask
- New: build a graph, connect masks if ANY of:
  - IoU > `--merge-iou-thresh` (0.5) — near-duplicate
  - Intersection/smaller > `--containment-thresh` (0.7) — one contains the other
  - Expanded bounding boxes overlap with `--box-expand-ratio` (0.15) — adjacent fragments
- Connected components are unioned, only committed if area ≤ `--max-mask-area-ratio` and aspect ratio ≤ `--merge-max-aspect-ratio` (5.0)

**Crop ranking** (inside `maskcut_multicrop`, lines 582-597):
- Old: binary skip/keep by coverage ratio
- New: score each window by (1 - coverage_ratio) × (1 + edge_density/128), keep top-N
- Edge density = mean gradient magnitude — high in textured/object-rich areas

**To run with new features:**
TWO_STAGE_CROP=1 CROP_SCALES=1.0,0.75,0.5 N_MASKS=3 sbatch slurm/run_multiscale_maskcut.sh
Use `--crop-top-k 8` to keep only 8 most informative windows per image.
