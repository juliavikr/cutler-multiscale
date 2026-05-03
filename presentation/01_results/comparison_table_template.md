# Baseline vs Multi-Scale Comparison Table

Use this template for the report and slides. Fill in the MS-MaskCut rows once
multiscale training and evaluation are complete.

Both models are trained on the same 10-class TinyImageNet pseudo-labels and evaluated
on **COCO val2017 class-agnostic**. Training config: `cascade_mask_rcnn_R_50_FPN.yaml`,
single A100, 20k iterations.

## Pseudo-label statistics

| | Baseline (single-scale) | MS-MaskCut |
|---|---|---|
| Images | 500 | 500 |
| Annotations | 748 | **TODO** |
| Masks / image (mean) | 1.50 | **TODO** |
| Small masks (area < 32²) | **TODO** | **TODO** |
| Medium masks | **TODO** | **TODO** |
| Mean mask area (px²) | **TODO** | **TODO** |

## Detection results (COCO val2017, BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl | ΔAPs |
|--------|----|------|------|-----|-----|-----|------|
| CutLER baseline | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 | — |
| MS-MaskCut (ours) | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** |

## Segmentation results (COCO val2017, SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl | ΔAPs |
|--------|----|------|------|-----|-----|-----|------|
| CutLER baseline | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 | — |
| MS-MaskCut (ours) | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** |

## Runtime comparison

| Stage | Baseline | MS-MaskCut |
|---|---|---|
| MaskCut (500 images) | ~1 h | **TODO** (~10 h expected) |
| Detector training (20k iter) | **TODO** | **TODO** |
| COCO evaluation | **TODO** | **TODO** |

## Notes for the write-up

- Lead with ΔAPs — that is the primary claim.
- If overall AP drops, explain the trade-off (more small masks → more false positives at large scale).
- Include the pseudo-label stats table to show *why* the multiscale method produces different results.

## TODO

- [ ] Run multiscale MaskCut on 10-class TinyImageNet with locked params.
- [ ] Run `PSEUDO_LABEL_NAME=multiscale sbatch slurm/run_training.sh`.
- [ ] Run `sbatch slurm/run_eval.sh` with the multiscale checkpoint.
- [ ] Fill all **TODO** cells above.
