# cutler-multiscale
Improving small-object detection in CutLER via Multi-Scale Pseudo-Label Generation

Custom multi-scale pseudo-label generation code lives in [`multiscale/`](/mnt/c/Users/Luiz%20Venosa/Documents/Bocconi/Master/2nd%20Semester/Computer%20VIsion/project/cutler-multiscale/multiscale), not inside the upstream `CutLER/` submodule.

## Results

Evaluated on COCO val2017, class-agnostic, fully unsupervised (no labels used at any stage).

### Bounding Box Detection

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — |
| CutLER (ours, reproduced) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 |
| MS-MaskCut (ours) | — | — | — | — | — | — |

### Instance Segmentation

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (ours, reproduced) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 |
| MS-MaskCut (ours) | — | — | — | — | — | — |

Baseline uses the pre-trained `cutler_cascade_final.pth` checkpoint released by Facebook Research.
Results match the original CutLER paper, confirming reproducibility.
