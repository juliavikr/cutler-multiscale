# Results

This folder is for committed result summaries and presentation-ready artifacts. Large raw outputs such as checkpoints, pseudo-label JSONs, and SLURM logs stay off-repo on the cluster.

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
