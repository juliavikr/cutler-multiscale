# Results

Committed result summaries and presentation-ready artifacts. Large raw outputs (checkpoints, pseudo-label JSONs, SLURM logs) stay off-repo on the cluster.

## CSV column reference

### `detector_results_5class.csv`

Evaluation dataset: COCO val2017, class-agnostic, 5-class TinyImageNet training subset (2500 images).
All AP values are COCO-style Average Precision on a 0–100 scale (not 0–1).

| Column | Meaning |
|--------|---------|
| `variant` | Training pseudo-label set identifier |
| `bbox_ap` | Bounding box AP averaged over IoU thresholds 0.50–0.95 |
| `ap50` | Bounding box AP at IoU ≥ 0.50 |
| `ap75` | Bounding box AP at IoU ≥ 0.75 |
| `aps` | Bounding box AP for small objects (area < 32² px) |
| `apm` | Bounding box AP for medium objects (32²–96² px) |
| `apl` | Bounding box AP for large objects (> 96² px) |
| `segm_ap` | Segmentation mask AP averaged over IoU 0.50–0.95 |
| `segm_ap50` | Segmentation mask AP at IoU ≥ 0.50 |
| `segm_ap75` | Segmentation mask AP at IoU ≥ 0.75 |

Variant identifiers:

| `variant` | Description |
|-----------|-------------|
| `baseline_single_scale` | Original CutLER single-scale MaskCut |
| `old_multiscale_only` | Legacy dense-grid crop masks only (no baseline) |
| `old_combined` | Baseline + legacy dense-grid crop masks merged |
| `new_hybrid_only` | Refined heatmap-guided crop masks only (no baseline) |
| `new_combined_hybrid_best` | Baseline + refined heatmap crop masks merged (final result) |

### `hybrid_ablation_100_summary.csv`

Pipeline statistics for 5 ablation variants run on a fixed 100-image subset
of the 5-class TinyImageNet training set. Each row is one variant.
Column values are counts across all images in the subset.

| Column | Meaning |
|--------|---------|
| `variant` | Ablation configuration name |
| `images` | Number of images processed |
| `full_masks` | Masks produced by full-image MaskCut (baseline stage) |
| `windows` | Total crop windows proposed (heatmap peaks + spatial rescue) |
| `rescue` | Windows added by the spatial rescue grid strategy |
| `generated` | Crop masks generated across all windows before any filtering |
| `crop_candidates` | Candidates entering the scoring and deduplication stage |
| `scored` | Candidates that passed the CRF stability check |
| `crop_merged` | Crop masks surviving the graph-merge deduplication step |
| `merged` | Crop masks surviving the final per-image dedup against full-image masks |
| `final_annotations` | Total annotations in the output JSON (full + crop masks) |

Variant identifiers:

| `variant` | What changed from the baseline preset |
|-----------|----------------------------------------|
| `baseline` | Reference: hp85, topk12, crop sizes 0.25/0.35/0.50 |
| `hp90` | Heatmap percentile raised to 90 (stricter peak threshold) |
| `hp80` | Heatmap percentile lowered to 80 (looser peak threshold) |
| `topk8` | Crop budget reduced from 12 to 8 windows per image |
| `tightcrop` | Smaller crop sizes: 0.20/0.30/0.40 instead of 0.25/0.35/0.50 |

## Current committed artifacts

- `results/detector_results_5class.csv` - final 5-class detector comparison table
- `results/hybrid_ablation_100_summary.csv` - 100-image ablation summary table
- `results/figures/` - curated PNG plots for the report and poster
  - `detector_results_5class.png` - main detector comparison figure
  - `hybrid_ablation/` - ablation summary figures
  - `training_losses/` - training loss progression figures

## What belongs here

- compact AP summary tables
- selected plots and visualization panels
- ablation summary tables
- runtime summary tables

## What does not belong here

- model checkpoints
- full pseudo-label JSONs
- raw cluster logs
- temporary debugging files

## Final detector comparison

Main 5-class study, class-agnostic COCO evaluation:

| Training pseudo-labels | BBOX AP | AP50 | AP75 | APs | APm | APl | SEGM AP | SEGM AP50 | SEGM AP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline single-scale | 2.1447 | 5.0707 | 1.5289 | 1.0761 | 1.8864 | 4.7238 | 0.4792 | 1.0439 | 0.4195 |
| Old multiscale only | 0.2780 | 0.7062 | 0.1927 | 0.3977 | 0.2598 | 1.1470 | 0.1950 | 0.5266 | 0.1304 |
| Old combined | 1.1962 | 2.7841 | 0.9193 | 0.8624 | 1.6792 | 3.5760 | 0.7319 | 1.4976 | 0.6339 |
| New hybrid-only | 0.3026 | 0.9407 | 0.1582 | 0.4065 | 0.2463 | 0.9210 | 0.1946 | 0.7254 | 0.1002 |
| New combined hybrid-best | 2.2557 | 5.2806 | 1.8755 | 1.0375 | 2.8046 | 5.6994 | 1.0814 | 2.1040 | 1.1533 |

## Main conclusion

The refined hybrid method is valuable as a **rescue mechanism** for missed local structure, but not as a standalone pseudo-label generator. The best result comes from **combining** the original single-scale MaskCut masks with the refined hybrid masks.

## Pseudo-label volume summary

| Pseudo-label set | Images | Annotations | Avg masks / image |
|---|---:|---:|---:|
| Baseline single-scale | 2500 | 3315 | 1.326 |
| Old multiscale only | 2500 | 24771 | 9.908 |
| Old combined | 2500 | 10530 | 4.212 |
| New hybrid-only | 2500 | 8337 | 3.335 |
| New combined hybrid-best | 2500 | 8187 | 3.275 |

## 100-image ablation summary

| Variant | Windows | Rescue | Generated | Crop merged | Final merged anns |
|---|---:|---:|---:|---:|---:|
| baseline | 1200 | 386 | 1189 | 398 | 496 |
| hp90 | 1200 | 386 | 1189 | 398 | 496 |
| hp80 | 1200 | 386 | 1189 | 398 | 496 |
| topk8 | 800 | 400 | 792 | 283 | 381 |
| tightcrop | 1200 | 390 | 1195 | 719 | 816 |

Interpretation:

- heatmap percentile had no effect on this subset
- crop budget and crop size were the meaningful levers
- `topk8` was the cleaner conservative option
- `tightcrop` was the aggressive but noisier option

## Related files

- top-level experiment ledger: `PROJECT_NOTES.md`
- method explanation: `PROJECT_OVERVIEW.md`
- ablation CSV: `results/hybrid_ablation_100_summary.csv`
- detector CSV: `results/detector_results_5class.csv`
- plot generators: `experiments/plot_hybrid_ablation_results.py`, `experiments/plot_training_losses.py`
