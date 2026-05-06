# Pseudo-label Statistics: Baseline vs Hybrid

- Baseline: `/home/3355142/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json`
- Hybrid: `/home/3355142/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_hybrid_pseudo.json`

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| Total images | 500 | 500 |
| Total annotations (masks) | 748 | 2,110 |
| Avg masks / image | 1.50 | 4.22 |
| Median masks / image | 1.00 | 4.00 |
| Mean mask area (px²) | 1,020.71 | 43.82 |
| Median mask area (px²) | 899.00 | 45.00 |
|--------|----------|--------|
| **Size bin** | | |
| Small (<32² px) | 428 (57.2%) | 2,110 (100.0%) |
| Medium (32²–96² px) | 320 (42.8%) | 0 (0.0%) |
| Large (>96² px) | 0 (0.0%) | 0 (0.0%) |
|--------|----------|--------|
| **Masks-per-image distribution** | | |
| Images with 0 masks | 1 | 12 |
| Images with 1 masks | 282 | 44 |
| Images with 2 masks | 185 | 72 |
| Images with 3 masks | 32 | 85 |
| Images with 4 masks | 0 | 85 |
| Images with 5+ masks | 0 | 202 |
