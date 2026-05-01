# Locked Experiment Parameters

**Canonical source:** `PROJECT_NOTES.md` → "Locked Experiment Parameters"

This file is a quick reference for the presentation. For the authoritative table and
rationale, always refer to `PROJECT_NOTES.md`. If the two conflict, `PROJECT_NOTES.md` wins.

## Why params are locked

Baseline and multiscale pseudo-labels must be generated with identical MaskCut settings so
that any difference in downstream detector performance is attributable *only* to the
pseudo-label generation strategy (single-scale vs multi-scale), not to tuning differences.
Changing any parameter below requires re-running **both** pipelines.

## Shared MaskCut parameters (baseline = multiscale)

| Parameter | Value | Purpose |
|---|---|---|
| `--vit-arch` | small | DINO ViT-Small backbone |
| `--vit-feat` | k | key features from self-attention |
| `--patch-size` | 8 | 8×8 px patches (finer than patch-16) |
| `--tau` | 0.15 | affinity graph threshold |
| `--N` | 3 | max masks per image / per crop |
| `--fixed_size` | 480 | resize input to 480×480 px |
| `--pretrain_path` | `~/cutler-multiscale/checkpoints/dino_deitsmall8_300ep_pretrain.pth` | DINO weights |

## Multi-scale-only additions

| Parameter | Value | Purpose |
|---|---|---|
| `--multi-crop` | (flag) | enable multi-scale mode |
| `--crop-scales` | 1.0, 0.75, 0.5 | three zoom levels |
| `--crop-overlap` | 0.3 | sliding window overlap |
| `--merge-iou-thresh` | 0.5 | NMS IoU threshold |
| `--small-first` | (flag) | prefer small masks during NMS |

## Detector training parameters (same for both)

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | Cascade Mask R-CNN R50+FPN | same as CutLER paper |
| `IMS_PER_BATCH` | 8 | single A100 (paper used 16 across 8 GPUs) |
| `BASE_LR` | 0.005 | linear scaling from paper's 0.01 (16→8 images) |
| `MAX_ITER` | 20,000 | 10× shorter than paper's 160k (500 images vs 1.3M) |
| `STEPS` | (15000,) | LR drop at 75% of training |
| `WARMUP_ITERS` | 1,000 | standard warmup |
