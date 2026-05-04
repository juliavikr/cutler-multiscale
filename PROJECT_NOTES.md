# Project Notes — cutler-multiscale

## Current Status: Phase 3 — Multi-Scale MaskCut (in progress)

**Started:** 2026-04-20 | **Last updated:** 2026-05-04

Baseline pseudo-labels exist (500 images, 748 annotations, 2026-05-01). Multi-scale pseudo-labels (hybrid heatmap, MOST-lite v2 soft) are **pending** generation with the v2 code. A first detector training run (v1 code, 5-class subset) produced near-zero COCO AP as expected given the tiny training set. The controlled comparison — baseline vs multi-scale, same 10-class dataset, same detector — has not yet been run.

**Immediate next steps:**
1. Fix incomplete `SOLVER.` line in `slurm/run_training.sh` before next training run
2. Generate v2 multi-scale pseudo-labels: `sbatch slurm/run_multiscale_maskcut.sh` (see `README.md` for env-var invocations)
3. Run baseline detector training: `PSEUDO_LABEL_NAME=baseline sbatch slurm/run_training.sh`
4. Run multi-scale detector training: `PSEUDO_LABEL_NAME=multiscale sbatch slurm/run_training.sh`
5. Evaluate both on COCO val2017, compare APs/APm

---

## Experiment Setup

### Locked Parameters

Both baseline and multi-scale runs **must** use these identical settings. Any change invalidates the comparison and requires re-running both.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--vit-arch` | `small` | DINO ViT-Small backbone |
| `--vit-feat` | `k` | key features from attention |
| `--patch-size` | `8` | 8×8 pixel patches |
| `--tau` | `0.15` | affinity graph threshold |
| `--N` | `3` | max masks per image |
| `--fixed_size` | `480` | resize input to 480×480 square |
| `--pretrain_path` | `${DATA_ROOT}/weights/dino_deitsmall8_pretrain.pth` | DINO pretrained weights |

### Method Variants

| Method | Flags | Training output |
|--------|-------|-----------------|
| Baseline | *(no `--multi-crop`)* | `normal` split |
| Hybrid heatmap | `--multi-crop --ms-preset small --primary-output multiscale` | `multiscale` split |
| MOST-lite v2 soft | `--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45 --primary-output multiscale` | `multiscale` split |

Use the `multiscale` split for training and quantitative evaluation. Use `combined`, `raw_multiscale`, and `candidate_debug` for diagnosis only — `combined` can contain overlapping normal + crop masks and should not be used as a training target.

See `README.md` for the exact `sbatch` commands for all three methods.

### Dataset

- **Path (cluster):** `~/data/tiny-imagenet-10classes/train/`
- **Size:** 10 classes × 50 images = **500 images total**
- **Classes:** n01443537 (goldfish), n02123045 (tabby cat), n02281406 (sulphur butterfly), n02410509 (bison), n02906734 (broom), n03100240 (convertible), n03444034 (go-kart), n04067472 (reel), n04254777 (sock), n07711569 (mashed potato)
- **Why 10 classes, not more:** Baseline runs ~6–7 s/image on A100; multi-scale ~10× more MaskCut calls per image. 500 images fits the cluster's 24h time limit with margin for re-runs. 50 classes (~2,500 images) would require ~50h and not fit. See `presentation/04_design_decisions/why_tinyimagenet.md` for full rationale.

---

## Generated Artifacts

Pseudo-label JSONs are **not committed to git** (regeneratable; excluded by `.gitignore`). All live on the cluster at `~/data/tiny-imagenet-10classes/annotations/`.

| Artifact | Status | Filename | Images | Annotations | Date |
|----------|--------|----------|--------|-------------|------|
| Baseline pseudo-labels | **exists** | `tinyimagenet_10c_baseline_pseudo.json` | 500 | 748 | 2026-05-01 |
| Hybrid heatmap pseudo-labels | **pending** | `tinyimagenet_10c_hybrid_pseudo.json` | — | — | — |
| MOST-lite v2 soft pseudo-labels | **pending** | `tinyimagenet_10c_mostlite_v2_soft_pseudo.json` | — | — | — |

To regenerate baseline: `sbatch slurm/run_maskcut_baseline.sh`
To generate multi-scale variants: see `README.md`.

---

## Compute Environment

| Setting | Value |
|---------|-------|
| Cluster | Bocconi HPC (`slogin.hpc.unibocconi.it`) |
| Partition / QOS | `stud` / `stud` |
| GPU | NVIDIA A100 80GB |
| Conda env | `cutler` — Python 3.9, CUDA 12.1, PyTorch 2.5.1+cu121 |
| Detectron2 | 0.6+fd27788pt2.5.0cu121 (miropsota pre-built wheels) |
| numpy | pinned `<2` (Detectron2 0.6 incompatible with numpy 2.x) |

**Workflow:** edit locally → `git push` → SSH to cluster → `git pull` → `sbatch`

**Detectron2 install note:** Must use miropsota pre-built wheels. Building from source against PyTorch 2.x fails due to removed `torch.cuda.amp` APIs. See `slurm/install_detectron2.sh`.

**⚠ Account discrepancy in SLURM scripts:** `run_multiscale_maskcut.sh` uses account `3355142`; `run_maskcut_baseline.sh` and `run_training.sh` use `3152697`. Verify the correct account and update `#SBATCH --account` and log path prefixes in all scripts before submitting.

---

## MaskCut Code Versions

### v1 — IoU-NMS merging
- Merge: greedy NMS — drop newer mask if IoU > threshold
- Crop selection: binary skip/keep by coverage ratio
- Used for: Run 1 (5-class, 2,500 images, 24,771 annotations, job 484052 pseudo-labels + job 486811 training)
- Preserved as: `multiscale/multiscale_maskcut_legacy.py`

### v2 — Graph-based merging + crop ranking (current, 2026-05-01)
- Merge: graph-based — connect masks by IoU, containment, or box adjacency; union connected components
- Crop ranking: score windows by `(1 − coverage_ratio) × (1 + edge_density/128)`, keep top-N
- Preset system: `--ms-preset small / balanced / mostlite / legacy`
- Main file: `multiscale/multiscale_maskcut.py`
- Hybrid snapshot: `multiscale/multiscale_maskcut_hybrid.py`

For full implementation details see `multiscale/MULTISCALE_MASKCUT.md`. For a comparison of all crop proposal strategies see `multiscale/STRATEGY_COMPARISON.md`.

---

## Phase Log

### Phase 1: Setup ✓ (2026-04-20)
- [x] Cloned CutLER as git submodule; set up project structure (CLAUDE.md, .gitignore, branches)
- [x] Installed `cutler` conda env on cluster using miropsota Detectron2 wheels
- [x] Confirmed cluster data paths; `DATA_ROOT` set to `${HOME}/data` in all SLURM scripts
- [x] Downloaded COCO val2017 images and annotations to `~/data/coco/`
- [x] Downloaded pre-trained CutLER checkpoint and pre-generated MaskCut annotations

### Phase 2: Baseline Reproduction ✓ (2026-04-27)
- [x] Evaluated pre-trained `cutler_cascade_final.pth` on COCO val2017
- [x] Results match CutLER paper — reproducibility confirmed (see Results Tracker)

### Phase 3: Multi-Scale MaskCut (in progress)
- [x] Implemented multi-scale MaskCut with correct crop-from-original projection, CRF filtering, graph-based merge
- [x] Added DINO feature-contrast heatmap crop proposal mode
- [x] Added MOST-lite token-cluster crop proposal mode
- [x] Preset system: `small`, `balanced`, `mostlite`, `legacy`
- [x] Split output system: `normal`, `raw_multiscale`, `multiscale`, `combined`, `candidate_debug`
- [x] Single-image local debugging (see `debug/`) confirmed implementation recovers foreground objects the old version missed
- [x] Generated v1 pseudo-labels on 5-class TinyImageNet (job 484052, 2026-04-28): 2,500 images, 24,771 annotations
- [x] Generated v2 baseline pseudo-labels on 10-class TinyImageNet (job ?, 2026-05-01): 500 images, 748 annotations
- [x] Run 1: detector training on v1 multi-scale pseudo-labels (job 486811, 2026-05-01) — near-zero COCO AP (see Training Runs)
- [x] Added `tools/visualize_pseudo_masks.py`, `tools/train_wrapper.py`, `tools/register_tinyimagenet_pseudo.py`
- [ ] **Fix `slurm/run_training.sh` incomplete `SOLVER.` line**
- [ ] Generate v2 multi-scale pseudo-labels: hybrid heatmap + MOST-lite v2 soft on 10-class TinyImageNet
- [ ] Run baseline detector training on 10-class pseudo-labels
- [ ] Run multi-scale detector training on 10-class pseudo-labels
- [ ] Compare APs/APm: baseline vs multi-scale (controlled comparison)

### Phase 4: Analysis & Write-up (upcoming)
- [ ] Direct pseudo-mask evaluation: Small Recall@0.5 vs COCO GT (see `multiscale/EVALUATION_PROCESS.md`)
- [ ] Ablation: crop proposal mode, scales, iterations
- [ ] Visualizations for presentation (overlay grids, area distributions, recall by size)
- [ ] Final results table and course report

---

## Results Tracker

All evaluations on **COCO val2017**, class-agnostic, fully unsupervised. The main controlled comparison (rows 3–4) is still pending.

### Bounding Box Detection (BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — | reported in paper |
| CutLER (ours, reproduced) | **12.33** | **21.98** | **11.90** | **3.66** | **12.72** | **29.60** | `cutler_cascade_final.pth`, 2026-04-27 |
| Baseline trained (10-class) | — | — | — | — | — | — | training pending |
| Hybrid multiscale trained (10-class) | — | — | — | — | — | — | pseudo-labels pending |

### Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER (paper) | — | — | — | — | — | — | not reported separately |
| CutLER (ours, reproduced) | **9.78** | **18.92** | **9.19** | **2.44** | **8.77** | **24.29** | `cutler_cascade_final.pth`, 2026-04-27 |
| Baseline trained (10-class) | — | — | — | — | — | — | training pending |
| Hybrid multiscale trained (10-class) | — | — | — | — | — | — | pseudo-labels pending |

### Pseudo-Label Statistics — v1 code, 5-class TinyImageNet (2026-04-29)

| Metric | Single-scale | Multi-scale (v1) |
|--------|-------------|------------------|
| Total annotations | 3,315 | 24,771 |
| Masks / image | 1.33 | 9.91 |
| Mean area | 993.6 | 374.4 |
| Small (area < 1024) | 1,875 (56.6%) | 23,433 (94.6%) |
| Medium (1024–9216) | 1,440 (43.4%) | 1,338 (5.4%) |

Multi-scale v1 generates 7.5× more annotations concentrated in small objects. 9.9 masks/image is likely too many — v2 is more conservative with preset-based filtering.

---

## Training Runs

### Run 1 — v1 code, 5-class TinyImageNet (job 486811, 2026-05-01)

- **Model:** Cascade Mask R-CNN R50 FPN, class-agnostic, 160k iterations, batch size 2, 1 GPU
- **Training data:** 2,500 images (5 TinyImageNet classes), 24,771 v1 multi-scale annotations
- **Evaluation:** COCO val2017

BBOX:
| AP | AP50 | AP75 | APs | APm | APl |
|----|------|------|-----|-----|-----|
| 0.0005 | 0.0015 | 0.0003 | 0.0009 | 0.0006 | 0.0008 |

SEGM:
| AP | AP50 | AP75 | APs | APm | APl |
|----|------|------|-----|-----|-----|
| 0.0004 | 0.0011 | 0.0002 | 0.0004 | 0.0006 | 0.0002 |

**Why near-zero:** 2,500 training images is ~200× less than original CutLER (1.28M images). The detector does not learn generalizable features. This is expected and does not tell us anything about pseudo-label quality.

**What the valid comparison is:** Baseline vs multi-scale trained under identical conditions — same 10-class dataset, same detector architecture, same evaluation. The only variable is pseudo-label quality.

---

## Active Blockers

- **`slurm/run_training.sh` line 74:** `SOLVER.` is an incomplete argument with no value. Training will fail at submission. Needs to be completed before the next run.
- **SLURM account inconsistency:** `run_multiscale_maskcut.sh` has `#SBATCH --account=3355142`; `run_maskcut_baseline.sh` and `run_training.sh` have `#SBATCH --account=3152697`. Log path prefixes also differ. Align all scripts to the submitting user's account before running.
- **Speed regression:** Multi-scale currently runs ~48 s/image on A100 (target: ~6 s/image). Use `slurm/run_speedtest.sh` to profile. Limits scale-up beyond 500 images until resolved. Full TinyImageNet (100k images) feasible once fixed — use `--job-index` to split across jobs.

### Resolved (for reference)
- TinyImageNet has extra `images/` subdirectory per class — patched in `multiscale_maskcut.py`
- SLURM log paths must be absolute — patched in SLURM scripts
- `dino.py` unconditionally called `torch.hub` for weights — patched to check `os.path.isfile` first
- Building Detectron2 from source fails against PyTorch 2.x — resolved by using miropsota wheels

---

## Document Map

| File | Purpose |
|------|---------|
| `README.md` | Project overview, results table, and exact `sbatch` commands for all three methods |
| `PROJECT_NOTES.md` | This file — status, parameters, artifacts, results, audit trail |
| `PROJECT_OVERVIEW.md` | Plain-English explanation of the full pipeline (for non-experts / course report background) |
| `multiscale/MULTISCALE_MASKCUT.md` | Full guide to `multiscale_maskcut.py`: code structure, CLI arguments, how to run |
| `multiscale/STRATEGY_COMPARISON.md` | Detailed comparison of all crop proposal strategies (normal, grid, hybrid, MOST-lite, combined) |
| `multiscale/EVALUATION_PROCESS.md` | Evaluation methodology: metrics, dataset setup, experiment order, figures for presentation |
| `presentation/` | Report draft, slide outline, results templates, design decision docs |
| `slurm/` | All SLURM job scripts |
| `tools/` | `visualize_pseudo_masks.py`, `train_wrapper.py`, `register_tinyimagenet_pseudo.py` |
| `experiments/` | `environment.yml`, `rank_small_ap.py` |
| `debug/` | Single-image debug outputs: contact sheets, mask overlays, candidate JSON records |
| `CutLER/` | Upstream CutLER submodule (Facebook Research) — do not edit directly |
