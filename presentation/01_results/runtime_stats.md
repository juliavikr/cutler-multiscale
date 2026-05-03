# Runtime Statistics

## MaskCut throughput (A100 80GB)

| Stage | Rate | Source |
|---|---|---|
| Baseline MaskCut | ~6–7 sec / image | measured, 2026-04-27 |
| Multi-scale MaskCut | ~10× slower (est. 60–70 sec / image) | projected from 3-scale crop count |
| Detector training (20k iter, bs=8) | **TODO** | to be measured |

### Baseline run (10-class, 500 images)

| Metric | Value |
|---|---|
| Images processed | 500 |
| Annotations generated | 748 |
| Masks / image (mean) | 1.50 |
| Total wall time | ~1 hour |
| Job ID | (see cluster logs) |
| Output JSON | `~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json` |
| JSON size on disk | 321 KB |

### Multi-scale run (10-class, 500 images) — TODO

| Metric | Value |
|---|---|
| Images processed | **TODO** |
| Annotations generated | **TODO** |
| Total wall time | **TODO** (expected ~10 h) |
| Job ID | **TODO** |
| Output JSON | `~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_multiscale_pseudo.json` |

## Why 10 classes (500 images)?

The compute constraint drove this decision. Key numbers:

| Subset | Images | Baseline runtime | Multi-scale runtime | Fits in 24h limit? |
|---|---|---|---|---|
| 10 classes | 500 | ~1 h | ~10 h | **Yes** (with margin) |
| 25 classes | 1,250 | ~2.5 h | ~25 h | Borderline |
| 50 classes | 2,500 | ~5 h | ~50 h | No |
| Full TinyImageNet | 100,000 | ~165 h | — | No |

Multi-scale runs the MaskCut algorithm on the full image plus crop windows at 0.75× and 0.5×
scales, processing roughly 10× as many MaskCut calls per image. 10 classes × 50 images is
the largest subset where both baseline and multiscale can complete within the cluster's
24-hour wall-time limit, with enough margin to allow re-runs.

See `04_design_decisions/why_tinyimagenet.md` and `PROJECT_NOTES.md` for full rationale.

## TODO

- [ ] Record actual multiscale runtime from SLURM logs.
- [ ] Record detector training wall time for both runs.
- [ ] Add per-class breakdown if interesting (some classes may have more masks than others).
