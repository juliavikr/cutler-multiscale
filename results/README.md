# Results

This folder stores committed result summaries. Large output files (model checkpoints, pseudo-label JSONs, SLURM logs) are gitignored and live on the cluster at `~/data/` and `~/cutler-multiscale/experiments/`.

**What belongs here:**
- COCO AP summary tables (filled in manually after each eval run)
- Selected visualization PNGs (hand-picked from `experiments/visualizations/` for the report)

**What does NOT belong here:**
- Model checkpoints (`.pth`) — gitignored
- Full pseudo-label JSONs — gitignored
- Raw SLURM logs — gitignored

---

## Bounding Box Detection (BBOX) — COCO val2017

Class-agnostic, fully unsupervised. Evaluated with `sbatch slurm/run_eval.sh`.

| Method | AP | AP50 | AP75 | APs | APm | APl | Status |
|--------|----|------|------|-----|-----|-----|--------|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — | Reference |
| CutLER (ours, reproduced) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 | Done (2026-04-27) |
| Baseline trained (10-class) | — | — | — | — | — | — | Pending |
| Hybrid MS-MaskCut (10-class) | — | — | — | — | — | — | Pending |
| MOST-lite v2 soft (10-class) | — | — | — | — | — | — | Pending |

## Instance Segmentation (SEGM) — COCO val2017

| Method | AP | AP50 | AP75 | APs | APm | APl | Status |
|--------|----|------|------|-----|-----|-----|--------|
| CutLER (paper) | — | — | — | — | — | — | Not reported |
| CutLER (ours, reproduced) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 | Done (2026-04-27) |
| Baseline trained (10-class) | — | — | — | — | — | — | Pending |
| Hybrid MS-MaskCut (10-class) | — | — | — | — | — | — | Pending |
| MOST-lite v2 soft (10-class) | — | — | — | — | — | — | Pending |

---

For full experiment details, training run records, and active blockers see [`PROJECT_NOTES.md`](../PROJECT_NOTES.md).
