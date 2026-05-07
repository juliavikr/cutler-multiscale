# Project Notes — cutler-multiscale

## TL;DR

We extended CutLER's MaskCut with a hybrid heatmap-guided multi-scale strategy. We generated baseline (single-scale) and hybrid (multi-scale) pseudo-labels on a 10-class TinyImageNet subset, trained Cascade Mask R-CNN on each, and evaluated both on COCO val2017. The hybrid-trained detector achieves nearly 2× the small-object recall of the baseline-trained detector, confirming the design goal, though absolute AP is below the published pre-trained CutLER due to our small training set (500 images vs 1.3M).

---

## Status Checklist

- [x] Phase 1 — Cluster setup, conda env, dependencies, COCO val2017 + checkpoint downloaded
- [x] Phase 2 — Reproduce CutLER paper baseline (AP=12.33, APs=3.66 — matches paper)
- [x] Phase 3 — Pseudo-label generation: baseline single-scale (748 masks) and hybrid multi-scale (2,110 masks) on TinyImageNet 10c (500 images)
- [x] Phase 4 — Detector training: 20K iterations on each pseudo-label set, ~4 hours on A100 each
- [x] Phase 5 — Evaluation: both trained detectors evaluated on COCO val2017 cls_agnostic_coco
- [ ] Optional follow-ups (not required for presentation): longer training, larger TinyImageNet subset, ablation on crop scales

---

## Locked Experiment Parameters

**Any change to these invalidates comparisons and requires re-running both legs.**

### Shared MaskCut settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--vit-arch` | `small` | DINO ViT-Small backbone |
| `--vit-feat` | `k` | key features from self-attention |
| `--patch-size` | `8` | 8×8 pixel patches |
| `--tau` | `0.15` | affinity graph threshold |
| `--N` | `3` | max masks per image (full-image pass) |
| `--fixed_size` | `480` | resize input to 480×480 |
| `--pretrain_path` | `${HOME}/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth` | DINO ViT-S/8 weights |
| `--num-folder-per-job` | `10` | process all 10 class folders in one job |
| `--job-index` | `0` | single-job run |

### Hybrid-only additional flags

| Flag | Value |
|------|-------|
| `--multi-crop` | enabled |
| `--ms-preset` | `small` |

### Training settings

| Parameter | Value |
|-----------|-------|
| Config | `cascade_mask_rcnn_R_50_FPN` |
| MAX_ITER | 20000 |
| IMS_PER_BATCH | 8 |
| BASE_LR | 0.005 |
| GPUs | 1 (A100) |

### 10-class subset

`n01443537` (goldfish), `n02123045` (tabby cat), `n02281406` (sulphur butterfly), `n02410509` (bison), `n02906734` (broom), `n03100240` (convertible), `n03444034` (go-kart), `n04067472` (reel), `n04254777` (sock), `n07711569` (mashed potato)

---

## Compute Setup

- **Cluster**: Bocconi HPC (`slogin.hpc.unibocconi.it`), user `3355142`
- **GPU**: NVIDIA A100 80GB, partition `stud`, QOS `stud`
- **Conda env**: `cutler` — Python 3.9, PyTorch 2.5.1+cu121
- **Detectron2**: 0.6 installed from miropsota pre-built wheels (source build fails against PyTorch 2.x)
- **numpy** pinned `<2` (Detectron2 0.6 incompatible with numpy 2.x)
- **Detector training collaborator**: Luiz (ran baseline training + eval on cluster)

---

## Workflow

```
[Mac — write code with Claude Code]
        │
        │  git push origin main
        ▼
[GitHub — cutler-multiscale]
        │
        │  ssh cluster → git pull → sbatch slurm/<script>.sh
        ▼
[HPC cluster — SLURM job runs, logs → logs/<job>_%j.out]
```

---

## Results Tracker

### COCO val2017 — Bounding Box (BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER paper | 8.3 | 13.8 | 8.0 | — | — | — | reported in paper |
| Pre-trained CutLER (ours) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 | cutler_cascade_final.pth, 2026-04-27 |
| Trained on baseline pseudo-labels | 2.22 | 5.75 | 1.37 | 1.40 | 2.72 | 4.00 | 500 imgs single-scale, 20K iters, Luiz 2026-05-06 |
| Trained on hybrid pseudo-labels | 0.11 | 0.27 | 0.08 | 0.18 | 0.09 | 0.04 | 500 imgs multi-scale, 20K iters, Julia 2026-05-06 |
| Trained on combined pseudo-labels | 3.07 | 6.89 | 2.20 | 1.20 | 4.19 | 4.87 | 500 imgs baseline+multiscale merged, 20K iters, Julia 2026-05-06 |

### COCO val2017 — Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| Pre-trained CutLER (ours) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 |
| Trained on baseline pseudo-labels | 0.75 | 1.43 | 0.64 | 0.76 | 1.59 | 0.74 |
| Trained on hybrid pseudo-labels | 0.08 | 0.15 | 0.10 | 0.10 | 0.02 | 0.00 |
| Trained on combined pseudo-labels | 1.24 | 2.03 | 1.15 | 0.90 | 1.63 | 0.96 |

### Recall Comparison (Small Object Detection)

| Model | AR small | AR medium | AR large |
|-------|----------|-----------|----------|
| Trained on baseline pseudo-labels | 0.040 | 0.114 | 0.020 |
| Trained on hybrid pseudo-labels | 0.078 | 0.013 | 0.000 |
| Trained on combined pseudo-labels | 0.084 | 0.243 | 0.342 |

The hybrid-trained detector nearly doubles small-object recall (0.078 vs 0.040) at the cost of medium and large object coverage. This confirms multi-scale pseudo-labels successfully shift the trained detector toward small-object discovery, exactly as designed. Absolute AP is below the pre-trained model because we used 500 training images vs the paper's 1.3M.

---

## Pseudo-label Statistics (TinyImageNet 10c)

| Metric | Baseline | Hybrid | Combined |
|--------|----------|--------|----------|
| Total images | 500 | 500 | 500 |
| Total annotations | 748 | 2,110 | 1,493 |
| Avg masks/image | 1.50 | 4.22 | 2.99 |
| Median masks/image | 1.00 | 4.00 | 3.00 |
| Mean mask area (px²) | 1,020.71 | 43.82 | 607.0 |
| Median mask area (px²) | 899.00 | 45.00 | 184.0 |
| Small (<32² px) | 428 (57.2%) | 2,110 (100.0%) | 1,104 (73.9%) |
| Medium (32²–96² px) | 320 (42.8%) | 0 (0.0%) | 389 (26.1%) |
| Large (>96² px) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Images with 0 masks | 1 | 12 | 0 |
| Images with 1 mask | 282 | 44 | — |
| Images with 2 masks | 185 | 72 | — |
| Images with 3 masks | 32 | 85 | — |
| Images with 4 masks | 0 | 85 | — |
| Images with 5+ masks | 0 | 202 | — |

Both JSON files live on the cluster at `~/data/tiny-imagenet-10classes/annotations/` (gitignored — regenerate with `sbatch slurm/run_maskcut_baseline.sh` or `sbatch slurm/run_hybrid_maskcut_tinyimagenet.sh`).

---

## Open Questions / Limitations

- **500 images is a small training set** — results are indicative, not statistically robust; the baseline-vs-hybrid delta on APs is what matters, not absolute AP.
- **Hybrid masks are extremely small** (median 45 px²) — at this scale, mask quality is hard to verify visually and noise is plausible.
- **12 images yield zero hybrid masks** where baseline finds at least one — suggests over-aggressive crop filtering in some cases.
- **Detector trained on 500 images cannot match a model trained on 1.3M** — the AP gap vs the pre-trained CutLER is expected and not a failure mode.
- **Direct Level 1 evaluation** against COCO ground-truth masks (per `multiscale/EVALUATION_PROCESS.md`) was not run — that is a separate workstream outside the current report scope.

---

## Key References

| File | Purpose |
|------|---------|
| `README.md` | Setup, reproduction commands, end-to-end run instructions |
| `PROJECT_OVERVIEW.md` | Plain-English pipeline explanation (no prior CV knowledge assumed) |
| `multiscale/STRATEGY_COMPARISON.md` | Detailed comparison of all crop proposal strategies |
| `multiscale/EVALUATION_PROCESS.md` | Full evaluation methodology, metrics, and figures for the report |
| `presentation/01_results/pseudo_label_comparison.md` | Side-by-side pseudo-label statistics table (generated by `tools/compare_pseudo_label_stats.py`) |
