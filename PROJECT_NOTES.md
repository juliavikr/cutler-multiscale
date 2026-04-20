# Project Notes — cutler-multiscale

## Current Status: Phase 1 — Setup

**Date started:** 2026-04-20

---

## Phase Log

### Phase 1: Setup (current)
- [x] Cloned CutLER upstream repo into `CutLER/`
- [x] Created project structure (CLAUDE.md, .gitignore, branches)
- [ ] Install environment (detectron2, DINO weights)
- [ ] Verify CutLER baseline runs end-to-end on a small data sample
- [ ] Document exact environment versions in `experiments/environment.yml`

### Phase 2: Baseline Reproduction (upcoming)
- [ ] Run MaskCut pseudo-label generation on COCO unlabeled 2017
- [ ] Train Mask R-CNN detector on pseudo-labels
- [ ] Evaluate on COCO val2017 — record AP, AP50, AP75, APs
- [ ] Compare to reported numbers in CutLER paper

### Phase 3: Multi-Scale MaskCut (upcoming)
- [ ] Implement image pyramid construction
- [ ] Run MaskCut at each scale
- [ ] Implement multi-scale proposal merging (Soft-NMS or WBF)
- [ ] Regenerate pseudo-labels with multi-scale method
- [ ] Retrain detector, evaluate — focus on APs (small object AP)

### Phase 4: Analysis & Write-up (upcoming)
- [ ] Ablation: effect of each scale, merging strategy, number of iterations
- [ ] Visualization of recovered small objects
- [ ] Final results table
- [ ] Course report

---

## Blockers / Open Questions

_None yet._

---

## Results Tracker

| Method | AP | AP50 | AP75 | APs | Notes |
|--------|----|------|------|-----|-------|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | COCO val, unsupervised |
| CutLER (ours) | — | — | — | — | to be filled |
| MS-MaskCut (ours) | — | — | — | — | to be filled |

---

## Environment

- GPU: TBD
- CUDA: TBD
- PyTorch: TBD
- Detectron2: TBD
