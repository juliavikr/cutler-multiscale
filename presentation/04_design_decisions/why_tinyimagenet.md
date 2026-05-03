# Why TinyImageNet (10-class subset)?

**Canonical source:** `PROJECT_NOTES.md` → "Locked Experiment Parameters" → "Why 10 classes"

## Short answer

Compute constraints. Multi-scale MaskCut is ~10× slower than single-scale. 10 classes
(500 images) is the largest subset where both runs complete within the cluster's 24-hour
SLURM wall-time limit, leaving margin for re-runs.

## Longer answer

### Why not full ImageNet (1.3M images)?

CutLER trains on full ImageNet in the paper. We don't have that compute:
- Baseline MaskCut: ~6–7 sec/image on A100 → 1.3M images ≈ 100 days. Not feasible.
- Even a 10% subset (130k images) ≈ 10 days for baseline, 100 days for multiscale.

### Why not full TinyImageNet (200 classes, 100k images)?

- 100k images × 7 sec = ~165 hours for baseline alone. Still doesn't fit.

### Why not 5 classes (250 images)?

We did an earlier 5-class run during development (see Phase 3 log in `PROJECT_NOTES.md`).
10 classes gives twice the data, making the comparison more reliable, and still fits
comfortably within 24 hours for both runs.

### Why these 10 classes?

Chosen for visual diversity: 4 animals, 3 vehicles/objects, 3 household items.
Full class list with WordNet IDs in `PROJECT_NOTES.md` → "Dataset".

## Implications for the report

- **Be upfront** about the scale gap. Our detector trains on 500 images; the paper uses 1.3M.
  The goal is a *controlled relative comparison* (multiscale vs baseline on the same data),
  not reproducing the paper's absolute numbers.
- Frame the 10-class subset as a deliberate design choice, not a limitation to apologise for.
- The controlled setup is actually a strength: identical data, identical training, only the
  pseudo-label generation differs.
