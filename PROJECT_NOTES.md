# Project Notes — cutler-multiscale

## TL;DR

We have reproduced CutLER's published COCO results (AP=12.33, APs=3.66). We are now running two parallel experiment tracks: a **5-class track** (250 images) to get results in time for the presentation on 2026-05-07, and a **10-class track** (500 images) that continues in parallel for the final report. Both tracks compare single-scale baseline MaskCut against hybrid heatmap multi-scale MaskCut end-to-end, training a Cascade Mask R-CNN and evaluating on COCO val2017. The headline metric is APs (small-object AP).

> **Scope reduction — 2026-05-05:** Reduced active experiment to 5-class TinyImageNet (250 images) due to presentation timeline. The 10-class hybrid pseudo-label job (488887) exceeded the available compute window before training could begin. The 10-class plan remains intact below and continues as the report track.

---

## Project Goal

We extend CutLER by replacing its single-scale MaskCut pseudo-label generator with a multi-scale variant that runs additional MaskCut passes inside DINO-heatmap-guided crops and merges the results. The hypothesis is that cropping before MaskCut gives more patches-per-object for small objects, improving pseudo-label recall and ultimately APs after detector training. We compare baseline vs. hybrid multi-scale end-to-end on COCO val2017.

---

## Status Checklist

### Phase 1 — Setup
- [x] Cluster access, conda env, all dependencies installed
- [x] COCO val2017 + pre-trained CutLER checkpoint downloaded
- [x] Repo structure with CutLER submodule, branches, .gitignore
- [x] All SLURM scripts working (account=3355142, partition=stud, qos=stud)

### Phase 2 — Baseline reproduction
- [x] Pre-trained CutLER eval on COCO val2017 — AP=12.33, APs=3.66 (matches paper)

---

### Track A — 5-class subset (presentation, deadline 2026-05-07)

250 images (5 classes × 50). SLURM scripts updated 2026-05-05.

- [ ] Download / prepare `~/data/tiny-imagenet-5classes/` on cluster (5 of the original 10 class folders)
- [ ] Baseline pseudo-labels → `tinyimagenet_5c_baseline_pseudo.json` (`sbatch slurm/run_maskcut_baseline.sh`, ~30min)
- [ ] Hybrid pseudo-labels → `tinyimagenet_5c_hybrid_pseudo.json` (`sbatch slurm/run_hybrid_maskcut_tinyimagenet.sh`, ~6h)
- [ ] Train detector A on baseline: `PSEUDO_LABEL_NAME=baseline sbatch slurm/run_training.sh` (~2–4h)
- [ ] Train detector B on hybrid: `PSEUDO_LABEL_NAME=hybrid sbatch slurm/run_training.sh` (~2–4h)
- [ ] Evaluate both on COCO val2017 cls_agnostic, record APs
- [ ] Pseudo-mask visualizations (5–10 example images, baseline vs. hybrid overlay)
- [ ] Slide deck with real numbers

### Track B — 10-class subset (report, no hard deadline)

500 images (10 classes × 50). Original plan, continues after presentation.

- [x] TinyImageNet restructured (10 classes, 50 images each = 500 total)
- [x] Baseline pseudo-labels generated: `tinyimagenet_10c_baseline_pseudo.json` (500 imgs, 748 masks, 2026-05-01)
- [~] Hybrid pseudo-labels: job 488887 ran on 10-class data; check if output exists on cluster
- [ ] Once hybrid JSON confirmed: run `PSEUDO_LABEL_NAME=baseline sbatch slurm/run_training.sh` with 10c paths restored
- [ ] Run `PSEUDO_LABEL_NAME=hybrid sbatch slurm/run_training.sh` with 10c paths
- [ ] Evaluate both, fill results tables
- [ ] Written report with full results

---

## Locked Experiment Parameters

**Any change to these invalidates comparisons and requires re-running both legs.**

### Shared MaskCut settings (both tracks)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--vit-arch` | `small` | DINO ViT-Small backbone |
| `--vit-feat` | `k` | key features from self-attention |
| `--patch-size` | `8` | 8×8 pixel patches |
| `--tau` | `0.15` | affinity graph threshold |
| `--N` | `3` | max masks per image (full-image pass) |
| `--fixed_size` | `480` | resize input to 480×480 |
| `--pretrain_path` | `${HOME}/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth` | DINO ViT-S/8 weights |
| `--job-index` | `0` | single-job run |

### Hybrid-only additional flags

| Flag | Value |
|------|-------|
| `--multi-crop` | enabled |
| `--ms-preset` | `small` |

### Training settings

| Parameter | Track A (5c, presentation) | Track B (10c, report) |
|-----------|----------------------------|-----------------------|
| Config | `cascade_mask_rcnn_R_50_FPN` | same |
| MAX_ITER | 8000 | 20000 |
| SOLVER.STEPS | `(6000,)` | `(15000,)` |
| IMS_PER_BATCH | 8 | 8 |
| BASE_LR | 0.005 | 0.005 |
| GPUs | 1 (A100) | 1 (A100) |

### 5-class subset (Track A)

First 5 of the original 10 classes: `n01443537` (goldfish), `n02123045` (tabby cat), `n02281406` (sulphur butterfly), `n02410509` (bison), `n02906734` (broom)

### 10-class subset (Track B)

All 10 classes: `n01443537` (goldfish), `n02123045` (tabby cat), `n02281406` (sulphur butterfly), `n02410509` (bison), `n02906734` (broom), `n03100240` (convertible), `n03444034` (go-kart), `n04067472` (reel), `n04254777` (sock), `n07711569` (mashed potato)

---

## Compute Setup (cluster)

- Cluster: Bocconi HPC (`slogin.hpc.unibocconi.it`), user `3355142`
- GPU: NVIDIA A100 80GB, partition `stud`, QOS `stud`
- Conda env: `cutler` — Python 3.9, PyTorch 2.5.1+cu121
- Detectron2: 0.6 installed from miropsota pre-built wheels (source build fails against PyTorch 2.x)
- numpy pinned `<2` (Detectron2 0.6 incompatible with numpy 2.x)

---

## Workflow

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

---

## Results Tracker

### COCO val2017 — Bounding Box (BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — |
| CutLER (ours, reproduced) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 |
| Trained on baseline pseudo-labels (5c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on hybrid pseudo-labels (5c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on baseline pseudo-labels (10c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on hybrid pseudo-labels (10c) | TBD | TBD | TBD | TBD | TBD | TBD |

### COCO val2017 — Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | — | — | — | — | — | — |
| CutLER (ours, reproduced) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 |
| Trained on baseline pseudo-labels (5c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on hybrid pseudo-labels (5c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on baseline pseudo-labels (10c) | TBD | TBD | TBD | TBD | TBD | TBD |
| Trained on hybrid pseudo-labels (10c) | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Pseudo-label Statistics

### Track A — 5-class (250 images)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| Images | TBD | TBD |
| Total masks | TBD | TBD |
| Masks / image | TBD | TBD |
| Output file | `tinyimagenet_5c_baseline_pseudo.json` | `tinyimagenet_5c_hybrid_pseudo.json` |
| Cluster path | `~/data/tiny-imagenet-5classes/annotations/` | same dir |

### Track B — 10-class (500 images)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| Images | 500 | TBD (check job 488887 output) |
| Total masks | 748 | TBD |
| Masks / image | 1.5 | TBD |
| Typical range | mostly 1–3 per image | TBD |
| Output file | `tinyimagenet_10c_baseline_pseudo.json` | `tinyimagenet_10c_hybrid_pseudo.json` |
| Cluster path | `~/data/tiny-imagenet-10classes/annotations/` | same dir |

---

## Open Questions / Known Limitations

- **5 classes is a very small training set** — absolute AP numbers will be low; the baseline-vs-hybrid delta is what matters, and even that may be noisy at this scale.
- **Speed**: hybrid multi-scale runs ~90 s/image on A100 vs ~7 s/image for baseline; the 10-class hybrid run (500 images) takes ~12.5h which was the driver for the 5-class pivot.
- **We do not implement Level-1 direct pseudo-mask evaluation** (Small Recall@0.50 against COCO GT masks) — described in `multiscale/EVALUATION_PROCESS.md` but out of scope for the current timeline.
- **MOST-lite v2 soft** is implemented but not included in the main comparison; it is available as a follow-on experiment if time permits.
- **Self-training is not planned** — we compare single-round pseudo-label quality only.

---

## Key References

| File | Purpose |
|------|---------|
| `README.md` | Setup, reproduction commands, end-to-end run instructions |
| `PROJECT_OVERVIEW.md` | Plain-English pipeline explanation (no prior CV knowledge assumed) |
| `multiscale/STRATEGY_COMPARISON.md` | Detailed comparison of all crop proposal strategies |
| `multiscale/EVALUATION_PROCESS.md` | Full evaluation methodology, metrics, and figures for the report |
| `multiscale/MULTISCALE_MASKCUT.md` | Code guide and full CLI reference for `multiscale_maskcut.py` |
| `slurm/README.md` | Complete SLURM script index and submission instructions |
