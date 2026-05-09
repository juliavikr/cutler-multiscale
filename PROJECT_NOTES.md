# Project Notes - cutler-multiscale

## TL;DR

We extended CutLER's MaskCut with a DINO heatmap-guided hybrid multi-scale strategy. The early 10-class pilot showed that multi-scale crops can shift the detector toward small-object discovery, but the later 5-class study made the main result much clearer:

- the hybrid masks **alone** are still too noisy to train a good detector
- the hybrid masks become useful when they are **merged with the baseline single-scale masks**
- the best result so far is the **new combined hybrid-best** setting, which slightly improves bbox AP over baseline and clearly improves segmentation AP

The ablation study explains why: the important levers were **crop budget** and **crop scale**, not the heatmap percentile threshold.

---

## Project Motivation

CutLER uses MaskCut to generate pseudo-masks from frozen self-supervised DINO features, then trains a detector on those pseudo-labels. In the standard single-scale setting, this works reasonably well for prominent objects, but it has a natural weakness: **small objects occupy very few ViT patches**.

That creates two problems:

1. **Low patch resolution for small objects**

   - if an object covers only a few patches, MaskCut has very little structure to separate it from the background
2. **Bias toward larger, more dominant regions**

   - the full-image graph partitioning tends to favor larger coherent regions, so small or locally distinctive objects are more likely to be missed

This project asks a focused question:

> Can we improve CutLER's pseudo-labels for small or missed objects by selectively re-running MaskCut inside informative crops, without changing the detector training stage?

---

## Proposed Pipeline

### Baseline: CutLER single-scale

The baseline pipeline is:

1. run DINO on the full image
2. run MaskCut once on the full image
3. generate pseudo-masks in COCO format
4. train Cascade Mask R-CNN on those pseudo-labels
5. evaluate class-agnostically on COCO

This gives a clean reference, but it can miss small objects because the entire image is processed at one scale.

### Our refined hybrid method

The refined hybrid method keeps the same detector training and evaluation stages. The only change is in pseudo-label generation:

1. run baseline full-image MaskCut
2. compute a **DINO feature-contrast heatmap**
3. select locally distinctive regions from that heatmap
4. place multi-size crop windows around those regions
5. run MaskCut again inside those crops
6. filter and merge crop masks
7. either:
   - use the crop masks alone as a hybrid pseudo-label set, or
   - merge them with the baseline single-scale masks

The final lesson from the project is that the second option is the useful one:

> the refined hybrid masks are best used as a **supplement** to baseline MaskCut, not as a replacement for it.

---

## Status Checklist

- [X] Phase 1 - Cluster setup, conda env, dependencies, COCO val2017 + checkpoint downloaded
- [X] Phase 2 - Reproduce CutLER paper baseline (AP=12.33, APs=3.66 - matches paper)
- [X] Phase 3 - Initial pseudo-label generation on TinyImageNet 10c
- [X] Phase 4 - Refined pseudo-label generation on TinyImageNet 5c
- [X] Phase 5 - Detector training on baseline, multiscale, combined, and refined hybrid variants
- [X] Phase 6 - COCO class-agnostic evaluation of all major detector runs
- [X] Phase 7 - 100-image hybrid ablation study
- [ ] Optional follow-ups: full-scale `topk8` run, longer training, direct pseudo-label quality evaluation against GT masks

---

## Locked Experiment Parameters

**Any change to these invalidates comparisons and requires re-running both legs.**

### Shared MaskCut settings

| Parameter        | Value     | Notes                            |
| ---------------- | --------- | -------------------------------- |
| `--vit-arch`   | `small` | DINO ViT-Small backbone          |
| `--vit-feat`   | `k`     | key features from self-attention |
| `--patch-size` | `8`     | 8x8 pixel patches                |
| `--fixed_size` | `480`   | resize input to 480x480          |
| `--ms-preset`  | `small` | refined hybrid preset            |

### Refined hybrid defaults

| Parameter          | Value                |
| ------------------ | -------------------- |
| heatmap crop sizes | `0.25, 0.35, 0.50` |
| heatmap percentile | `85`               |
| top-k crop windows | `12`               |
| spatial rescue     | enabled              |
| crop dedup         | enabled              |

### Training settings

| Parameter     | Value                          |
| ------------- | ------------------------------ |
| Config        | `cascade_mask_rcnn_R_50_FPN` |
| MAX_ITER      | 20000                          |
| IMS_PER_BATCH | 8                              |
| BASE_LR       | 0.005                          |
| GPUs          | 1 (A100)                       |

---

## Compute Setup

- **Cluster**: Bocconi HPC
- **GPU**: NVIDIA A100 80GB MIG 4g.40gb, partition `stud`
- **Conda env**: `cutler` - Python 3.9, PyTorch 2.5.1+cu121
- **Detectron2**: 0.6
- **Primary evaluation dataset**: `cls_agnostic_coco`

---



## Pseudo-label Generation Runtime

These numbers are the practical cost of generating pseudo-labels on the cluster. Where exact wall time was not logged for the final document, the notes mark the value as an estimate derived from measured throughput.

### 10-class pilot (500 images)

| Pseudo-label strategy      | Images | Approx. runtime | Basis                                |
| -------------------------- | ------ | --------------- | ------------------------------------ |
| Baseline single-scale      | 500    | ~1 hour         | measured                             |
| Early multi-scale / hybrid | 500    | ~10 hours       | measured project estimate from pilot |
| Combined pseudo-labels     | 500    | a few minutes   | JSON merge only                      |

### Main 5-class study (2500 images)

| Pseudo-label strategy           | Images | Approx. runtime                         | Basis                                                                    |
| ------------------------------- | ------ | --------------------------------------- | ------------------------------------------------------------------------ |
| Baseline single-scale           | 2500   | ~4.5 to 5 hours                         | estimated from ~6-7 sec / image                                          |
| Old multiscale only             | 2500   | very high, substantially above baseline | dense crop-heavy generation; exact final wall time not retained in notes |
| Refined hybrid-best             | 2500   | ~16.5 hours                             | estimated from ~24 sec / image during full run                           |
| Baseline + refined hybrid merge | 2500   | a few minutes                           | JSON merge only                                                          |

### Runtime interpretation

- The refined hybrid method is **much more expensive** than baseline single-scale MaskCut.
- This is why the ablation study was first run on **100 images** instead of immediately scaling every setting to 2500 images.
- The computational tradeoff is central to the project: better recovery of missed local structure costs a large increase in pseudo-label generation time.

---

## Results Tracker

### Reference: pre-trained CutLER

| Method                    | BBOX AP | AP50  | AP75  | APs  | APm   | APl   | SEGM AP | SEGM AP50 | SEGM AP75 | Notes                  |
| ------------------------- | ------- | ----- | ----- | ---- | ----- | ----- | ------- | --------- | --------- | ---------------------- |
| Pre-trained CutLER (ours) | 12.33   | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 | 9.78    | 18.92     | 9.19      | COCO eval sanity check |

### Preliminary 10-class pilot (500 images)

These were the first end-to-end detector runs. They were useful for direction, but the later 5-class study is the main result.

| Method                 | BBOX AP | AP50 | AP75 | APs  | APm  | APl  | SEGM AP | SEGM AP50 | SEGM AP75 | Notes                             |
| ---------------------- | ------- | ---- | ---- | ---- | ---- | ---- | ------- | --------- | --------- | --------------------------------- |
| Baseline pseudo-labels | 2.22    | 5.75 | 1.37 | 1.40 | 2.72 | 4.00 | 0.75    | 1.43      | 0.64      | 500 imgs, single-scale            |
| Hybrid pseudo-labels   | 0.11    | 0.27 | 0.08 | 0.18 | 0.09 | 0.04 | 0.08    | 0.15      | 0.10      | 500 imgs, early hybrid            |
| Combined pseudo-labels | 3.07    | 6.89 | 2.20 | 1.20 | 4.19 | 4.87 | 1.24    | 2.03      | 1.15      | 500 imgs, baseline + early hybrid |

Key takeaway from the pilot: combining baseline and multi-scale masks was promising, but the standalone hybrid detector was too noisy.

### Main 5-class study (final comparison)

This is the main detector comparison to cite in the report and presentation.

| Method                   | BBOX AP | AP50   | AP75   | APs    | APm    | APl    | SEGM AP | SEGM AP50 | SEGM AP75 | Notes                              |
| ------------------------ | ------- | ------ | ------ | ------ | ------ | ------ | ------- | --------- | --------- | ---------------------------------- |
| Baseline single-scale    | 2.1447  | 5.0707 | 1.5289 | 1.0761 | 1.8864 | 4.7238 | 0.4792  | 1.0439    | 0.4195    | `training_baseline5`             |
| Old multiscale only      | 0.2780  | 0.7062 | 0.1927 | 0.3977 | 0.2598 | 1.1470 | 0.1950  | 0.5266    | 0.1304    | `training_multiscale5`           |
| Old combined             | 1.1962  | 2.7841 | 0.9193 | 0.8624 | 1.6792 | 3.5760 | 0.7319  | 1.4976    | 0.6339    | `training_combined5`             |
| New hybrid-only          | 0.3026  | 0.9407 | 0.1582 | 0.4065 | 0.2463 | 0.9210 | 0.1946  | 0.7254    | 0.1002    | `training_hybrid_best5`          |
| New combined hybrid-best | 2.2557  | 5.2806 | 1.8755 | 1.0375 | 2.8046 | 5.6994 | 1.0814  | 2.1040    | 1.1533    | `training_combined_hybrid_best5` |

### Main interpretation

- The **old multiscale-only** detector was the weakest detector.
- The **new hybrid-only** detector is still weak, only slightly better than the old multiscale-only run in bbox AP and essentially unchanged in segm AP.
- The **new combined hybrid-best** detector is the strongest combined setup so far:
  - slightly better than baseline in **bbox AP** (`2.2557` vs `2.1447`)
  - clearly better than baseline in **segm AP** (`1.0814` vs `0.4792`)
  - much better than both the old multiscale-only and old combined runs

So the refined hybrid method works best as a **supplementary small-object recovery stage**, not as a standalone pseudo-label generator.

---

## Pseudo-label Statistics

### Main 5-class pseudo-label counts

| Pseudo-label set         | Images | Annotations | Avg masks / image | Notes                                           |
| ------------------------ | ------ | ----------- | ----------------- | ----------------------------------------------- |
| Baseline single-scale    | 2500   | 3315        | 1.326             | `v1_baseline_pseudo.json`                     |
| Old multiscale only      | 2500   | 24771       | 9.908             | `v1_multiscale_pseudo.json` before refinement |
| Old combined             | 2500   | 10530       | 4.212             | baseline + old multiscale merge                 |
| New hybrid-only          | 2500   | 8337        | 3.335             | refined crop masks,`multiscale.json`          |
| New combined hybrid-best | 2500   | 8187        | 3.275             | baseline + refined hybrid crop-only merge       |

### Baseline + refined hybrid merge stats

| Metric                     | Value |
| -------------------------- | ----- |
| baseline anns              | 3315  |
| refined hybrid crop anns   | 8337  |
| added refined hybrid anns  | 4872  |
| skipped duplicate / inside | 3465  |
| final combined anns        | 8187  |

This refined merge is much more controlled than the earlier old-multiscale merge. The key structural change is that the new hybrid set is far smaller than the old multiscale set (`8337` vs `24771`), which makes the final merged pseudo-label set much more usable for detector training.

---

## Hybrid Ablation Study (100-image screening)

The ablation study was run on a fixed 100-image subset to identify which hybrid controls mattered before scaling up.

### Final multi-crop stats

| Variant                                                 | Windows | Rescue | Generated | Crop merged | Final merged anns | Interpretation           |
| ------------------------------------------------------- | ------- | ------ | --------- | ----------- | ----------------- | ------------------------ |
| `baseline` (`hp85`, `topk12`, `0.25/0.35/0.50`) | 1200    | 386    | 1189      | 398         | 496               | reference hybrid setting |
| `hp90`                                                | 1200    | 386    | 1189      | 398         | 496               | identical to baseline    |
| `hp80`                                                | 1200    | 386    | 1189      | 398         | 496               | identical to baseline    |
| `topk8`                                               | 800     | 400    | 792       | 283         | 381               | conservative, cleaner    |
| `tightcrop` (`0.20/0.30/0.40`)                      | 1200    | 390    | 1195      | 719         | 816               | aggressive, noisier      |

### What the ablation showed

1. **Heatmap percentile was not the important lever here.**

   - `hp80`, `hp85`, and `hp90` produced the same outputs on this subset.
2. **Crop budget mattered.**

   - reducing from `topk12` to `topk8` lowered final masks from `496` to `381`
   - this matched the visual result: fewer duplicates and fewer fragments
3. **Crop scale mattered.**

   - `tightcrop` increased final masks from `496` to `816`
   - visually, it recovered more local structure but also produced many more edge and texture fragments

### Visual interpretation

- `topk8` was the cleanest variant and the safest candidate for detector training
- `baseline` was the middle ground
- `tightcrop` was the most aggressive and the noisiest

The ablation therefore supports the final project interpretation:

> The refined hybrid method is useful when it adds a controlled number of extra crop masks to the baseline set. The decisive controls are crop budget and crop scale, not the exact heatmap percentile threshold.

---

## What Changed Between the Old and New Hybrid Methods

### Old multiscale

- broad crop generation
- many extra masks
- weak control over duplicates / fragments
- standalone detector training performed poorly

### New hybrid

- DINO feature-contrast heatmap used to guide crop selection
- structured multi-size crop proposals around locally distinctive regions
- stronger crop-side filtering and merge control
- still too noisy as a standalone pseudo-label source
- clearly useful when merged with the baseline single-scale masks

This is the main conceptual shift:

> The old method tried to improve recall by adding many crop masks. The refined hybrid method uses DINO to zoom into locally unexplained regions and then adds only the useful extra masks to the baseline pseudo-label set.

---

## Open Questions / Limitations

- Hybrid masks **alone** still do not train a strong detector.
- The best result so far is a **merged** pseudo-label set, not a pure hybrid one.
- The 100-image ablation is a strong screening study, but `topk8` has not yet been promoted to a full 2500-image detector comparison.
- The refined study used a 5-class subset, so absolute AP remains much lower than large-scale published CutLER results.
- Direct level-1 pseudo-label quality evaluation against ground-truth masks was not run in this workstream.

---

## Key References

| File                                            | Purpose                                                   |
| ----------------------------------------------- | --------------------------------------------------------- |
| `README.md`                                   | Setup, reproduction commands, end-to-end run instructions |
| `PROJECT_OVERVIEW.md`                         | Plain-English pipeline explanation                        |
| `multiscale/STRATEGY_COMPARISON.md`           | Detailed comparison of crop proposal strategies           |
| `multiscale/EVALUATION_PROCESS.md`            | Evaluation methodology, metrics, and figures              |
| `results/hybrid_ablation_100_summary.csv`     | 100-image ablation summary data                           |
| `experiments/plot_hybrid_ablation_results.py` | Python plot generator for ablation presentation charts    |
