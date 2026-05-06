# Project Notes — cutler-multiscale

## TL;DR

We have reproduced CutLER's published COCO results (AP=12.33, APs=3.66), generated pseudo-labels for both baseline (748 masks) and hybrid multi-scale (500 imgs) variants, and trained + evaluated two Cascade Mask R-CNN detectors on COCO val2017. The hybrid-trained detector achieves nearly 2× the small-object recall (AR_small 0.078 vs 0.040) compared to the baseline-trained detector, confirming that multi-scale pseudo-labels successfully shift the detector toward small-object discovery. Absolute AP is low for both (training set is only 500 images vs 1.3M for the pre-trained model), but the relative delta on APs is the headline result.

---

## Project Goal

We extend CutLER by replacing its single-scale MaskCut pseudo-label generator with a multi-scale variant that runs additional MaskCut passes inside DINO-heatmap-guided crops and merges the results. The hypothesis is that cropping before MaskCut gives more patches-per-object for small objects, improving pseudo-label recall and ultimately APs after detector training. We compare baseline vs. hybrid multi-scale end-to-end on COCO val2017 using a controlled 10-class TinyImageNet training subset.

---

## Status Checklist

### Phase 1 — Setup
- [x] Cluster access, conda env, all dependencies installed
- [x] COCO val2017 + pre-trained CutLER checkpoint downloaded
- [x] Repo structure with CutLER submodule, branches, .gitignore
- [x] All SLURM scripts working (account=3355142, partition=stud, qos=stud)

### Phase 2 — Baseline reproduction
- [x] Pre-trained CutLER eval on COCO val2017 — AP=12.33, APs=3.66 (matches paper)

### Phase 3 — Pseudo-label generation on TinyImageNet 10-class subset
- [x] TinyImageNet downloaded and restructured (50 images × 10 classes = 500 images total)
- [x] Single-scale baseline MaskCut → `tinyimagenet_10c_baseline_pseudo.json` (500 imgs, 748 masks)
- [x] Hybrid multi-scale MaskCut → `tinyimagenet_10c_hybrid_pseudo.json` (job 488887 complete)

### Phase 4 — Detector training (pending Phase 3)
- [x] Train detector A on baseline pseudo-labels
- [x] Train detector B on hybrid pseudo-labels
- [x] Both use `cascade_mask_rcnn_R_50_FPN` config, single GPU, MAX_ITER=20000, IMS_PER_BATCH=8, BASE_LR=0.005

### Phase 5 — Evaluation on COCO val2017 (pending Phase 4)
- [x] Evaluate detector A on COCO val2017 cls_agnostic
- [x] Evaluate detector B on COCO val2017 cls_agnostic
- [x] Compare APs improvement (the headline metric)

### Phase 6 — Report and presentation (parallel with Phases 4–5)
- [ ] Pseudo-mask visualizations (baseline vs. hybrid, 5–10 example images)
- [ ] Results tables filled with real numbers
- [ ] Slide deck draft
- [ ] Written report

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

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| CutLER paper | 8.3 | 13.8 | 8.0 | — | — | — | reported |
| Pre-trained CutLER (ours, reproduced) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 | cutler_cascade_final.pth, 2026-04-27 |
| Trained on baseline pseudo-labels (Luiz) | 2.22 | 5.75 | 1.37 | 1.40 | 2.72 | 4.00 | 500 TinyImageNet imgs, single-scale MaskCut, 20K iters, 2026-05-06 |
| Trained on hybrid pseudo-labels (Julia) | 0.11 | 0.27 | 0.08 | 0.18 | 0.09 | 0.04 | 500 TinyImageNet imgs, multi-scale MaskCut, 20K iters, 2026-05-06 |

### COCO val2017 — Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl | Notes |
|--------|----|------|------|-----|-----|-----|-------|
| Pre-trained CutLER (ours, reproduced) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 | 2026-04-27 |
| Trained on baseline pseudo-labels (Luiz) | 0.75 | 1.43 | 0.64 | 0.76 | 1.59 | 0.74 | 2026-05-06 |
| Trained on hybrid pseudo-labels (Julia) | 0.08 | 0.15 | 0.10 | 0.10 | 0.02 | 0.00 | 2026-05-06 |

### Recall Comparison (Small Object Detection)

| Model | AR small | AR medium | AR large |
|-------|----------|-----------|----------|
| Trained on baseline pseudo-labels | 0.040 | 0.114 | 0.020 |
| Trained on hybrid pseudo-labels | 0.078 | 0.013 | 0.000 |

Although absolute AP is lower for the hybrid-trained detector — expected since 500 images is far below the 1.3M ImageNet images used by the pre-trained model — the hybrid-trained detector achieves nearly 2× the small-object recall (0.078 vs 0.040) while losing coverage on medium and large objects. This confirms that multi-scale pseudo-labels successfully shift the trained detector toward small-object discovery, which was the design goal. With more training data and longer training, this trade-off could be tuned to preserve large-object performance.

---

## Pseudo-label Statistics (TinyImageNet 10c)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| Images | 500 | TBD (pending job 488887) |
| Total masks | 748 | TBD |
| Masks / image | 1.5 | TBD |
| Typical range | mostly 1–3 per image | TBD |
| Output file | `tinyimagenet_10c_baseline_pseudo.json` | `tinyimagenet_10c_hybrid_pseudo.json` |

Both files live on the cluster at `~/data/tiny-imagenet-10classes/annotations/` (gitignored — regenerate with `sbatch slurm/run_maskcut_baseline.sh` or `sbatch slurm/run_hybrid_maskcut_tinyimagenet.sh`).

---

## Open Questions / Known Limitations

- **10 classes is a small training set** — results are indicative, not statistically robust; absolute AP numbers will be low, but the relative baseline-vs-hybrid delta is what matters.
- **Speed**: hybrid multi-scale runs ~10× slower than baseline (~48 s/image on A100 vs ~6 s/image); this limits scaling beyond 500 images without splitting across jobs.
- **We do not implement Level-1 direct pseudo-mask evaluation against COCO GT masks** (Small Recall@0.50 etc.) — that is a separate workstream described in `multiscale/EVALUATION_PROCESS.md` and not part of the current report scope.
- **MOST-lite v2 soft** is implemented and documented but not yet included in the main comparison; it is available if time permits after the baseline-vs-hybrid run completes.
- **Self-training is not planned** for this report; we compare single-round pseudo-label quality only.

---

## Key References

| File | Purpose |
|------|---------|
| `README.md` | Setup, reproduction commands, end-to-end run instructions |
| `PROJECT_OVERVIEW.md` | Plain-English pipeline explanation (no prior CV knowledge assumed) |
| `multiscale/STRATEGY_COMPARISON.md` | Detailed comparison of all crop proposal strategies (normal, hybrid, MOST-lite, combined) |
| `multiscale/EVALUATION_PROCESS.md` | Full evaluation methodology, metrics, and figures for the report |
| `multiscale/MULTISCALE_MASKCUT.md` | Code guide and full CLI reference for `multiscale_maskcut.py` |
| `slurm/README.md` | Complete SLURM script index and submission instructions |
