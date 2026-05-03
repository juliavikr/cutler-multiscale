# Compute Budget

## What we had

| Resource | Spec |
|---|---|
| Cluster | Bocconi University HPC (`slogin.hpc.unibocconi.it`) |
| GPU | NVIDIA A100 80GB (1 per job) |
| Partition | `stud` — shared with other students |
| Wall-time limit | 24 hours per job |
| Parallelism | 1 GPU per job (no multi-node) |
| Storage | `~/data/` on cluster (large files); no Git |

## How we spent it

| Stage | Est. cost | Status |
|---|---|---|
| Env setup + debugging | ~2–3 A100 hours | done |
| Baseline MaskCut (500 images) | ~1 A100 hour | done |
| Multi-scale MaskCut (500 images) | ~10 A100 hours | TODO |
| Baseline detector training (20k iter) | **TODO** | TODO |
| Multiscale detector training (20k iter) | **TODO** | TODO |
| COCO evaluations (×2) | ~15 min each | TODO |
| Visualization jobs | ~5 min each | TODO |

## Key tradeoffs made

**Scale of pseudo-label data (500 vs 1.3M images)**
We use 500 images for pseudo-labeling vs. the paper's 1.3M. This makes absolute AP numbers
incomparable to the paper, but the *relative* comparison (baseline vs. multiscale) remains
valid and is the core contribution.

**Single GPU vs. multi-GPU training**
The paper trains with 8 GPUs (IMS_PER_BATCH=16). We use 1 GPU (IMS_PER_BATCH=8) with
proportionally halved learning rate. This is standard linear scaling and has negligible
effect on final accuracy, especially at 20k iterations.

**20k vs. 160k training iterations**
Scaled down proportionally to dataset size: 500/1,300,000 × 160k ≈ 62 iterations would
be mathematically proportional, but we use 20k to give the model enough iterations to
converge on the limited data. This is a practical choice, not a theoretical one.

**Why we didn't ablate scales**
Running ablations (e.g., 2-scale vs 3-scale) would require 2–3× more compute. Given the
24-hour limit and the need to complete both baseline and multiscale full runs, ablations
are out of scope for this project.

## What we would do with more compute

- Train on full TinyImageNet (100k images) or a 50-class subset
- Run scale ablations (1.0+0.5, 1.0+0.75, 1.0+0.75+0.5)
- Try a second round of self-training (CutLER uses 2 rounds)
- Evaluate on a small-object benchmark (VisDrone, DOTA) instead of / in addition to COCO
