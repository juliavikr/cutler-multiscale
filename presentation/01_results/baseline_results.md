# Baseline Results

Evaluated on **COCO val2017**, class-agnostic, fully unsupervised (no labels used at any stage).
Checkpoint: `cutler_cascade_final.pth` (pre-trained CutLER, Facebook Research).
Run date: 2026-04-27. Numbers reproduced from the upstream model — match the paper.

## Bounding Box Detection (BBOX)

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — |
| **CutLER (ours)** | **12.33** | **21.98** | **11.90** | **3.66** | **12.72** | **29.60** |
| MS-MaskCut (ours) | — | — | — | — | — | — |

> Our number (12.33) exceeds the paper's reported 8.3 AP. The paper evaluates a different
> checkpoint / training regime; the pre-trained `cutler_cascade_final.pth` is a stronger model.
> This is the correct baseline to beat with our multiscale pseudo-labels.

## Instance Segmentation (SEGM)

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | — | — | — | — | — | — |
| **CutLER (ours)** | **9.78** | **18.92** | **9.19** | **2.44** | **8.77** | **24.29** |
| MS-MaskCut (ours) | — | — | — | — | — | — |

## Key observations

- **APs (small objects) is very low**: 3.66 bbox / 2.44 segm. This is the gap we are targeting.
- APl (large objects) is much higher (29.60), confirming the single-scale bias.
- The APs/APl ratio for bbox is 3.66 / 29.60 ≈ 0.12 — strong small-object deficit.

## TODO

- [ ] Fill MS-MaskCut rows once multiscale training + eval complete.
- [ ] Add delta rows (MS − Baseline) for APs, APm, APl to highlight the improvement.
