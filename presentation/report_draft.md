# Report Draft — cutler-multiscale

**Working title:** Multi-Scale MaskCut: Improving Small-Object Pseudo-Label Generation
for Unsupervised Object Detection

**Course:** [TODO: course name and code]
**Authors:** Julia Vik Remøy, [TODO: Luiz's full name]
**Date:** [TODO: submission date]

---

## Abstract

*[TODO: write last, after results are in. ~150 words.]*

Unsupervised object detection methods such as CutLER rely on DINO self-attention maps
and Normalized Cuts to generate pseudo-labels without human annotation. A known weakness
is the poor detection of small objects, caused by single-scale processing at a fixed
resolution. We propose a multi-scale extension of the MaskCut pseudo-label generation
stage that processes sliding crop windows at multiple zoom levels and merges proposals
via graph-based IoU filtering. We evaluate on [TODO] and report [TODO].

---

## 1. Introduction

Object detection requires large annotated datasets, which are expensive to collect and
domain-specific. Unsupervised methods that generate pseudo-labels from self-supervised
features offer a scalable alternative, but existing approaches exhibit a strong bias
toward large, salient objects.

**Problem:** CutLER's MaskCut processes images at a fixed scale, causing small objects
to be systematically under-represented in the generated pseudo-labels, leading to low
APs scores (3.66 bbox on COCO val2017) despite competitive overall AP (12.33).

**Our contribution:** We replace the single-scale MaskCut step with a multi-scale
variant that (i) constructs a sliding-window image pyramid, (ii) runs MaskCut
independently on each crop, (iii) back-projects masks to original coordinates, and
(iv) merges proposals across scales with graph-based NMS. All other pipeline components
(DINO backbone, detector architecture, training config) are held fixed, enabling a
controlled comparison.

*[TODO: Add 1–2 sentences summarizing the result once available.]*

---

## 2. Related Work

**Unsupervised / self-supervised object detection.**
DINO [Caron et al., 2021] demonstrates that ViT self-attention maps segment objects
without supervision. TokenCut [Wang et al., 2022] and MaskCut [Wang et al., 2023] apply
graph partitioning to these maps to produce segmentation masks. CutLER [Wang et al., 2023]
extends this into a full detection pipeline with self-training.

**Multi-scale detection.**
Feature Pyramid Networks [Lin et al., 2017] address scale variation at the feature level.
Image pyramids at test time are standard practice for small-object detection. Our work
applies the multi-scale idea at the pseudo-label generation stage rather than inference.

**Small-object detection.**
*[TODO: cite 1–2 papers on small-object detection challenges, e.g., SNIP, SNIPER, or
TinyPerson. Optional — include only if the report requires a fuller related work section.]*

---

## 3. Method

### 3.1 Background: CutLER and MaskCut

CutLER uses DINO ViT-S/8 features to produce pseudo-masks via iterative Normalized Cuts
on the patch-level affinity matrix. Given an image resized to 480×480, MaskCut extracts
the `k`-features from DINO's self-attention, constructs a pairwise affinity graph, and
solves the Normalized Cuts eigenvector problem to obtain a binary foreground/background
partition. This is repeated N=3 times with background suppression.

### 3.2 Multi-Scale Extension

*[TODO: write up the method from `03_diagrams/multiscale_strategy.md`.]*

We extend the pseudo-label generation stage as follows:

1. **Image pyramid construction.** For each image, we define sliding windows at crop
   scales {1.0, 0.75, 0.5} with an overlap of 0.3.
2. **Per-crop MaskCut.** Each window is cropped from the *original* image (not the
   resized version) and independently resized to 480×480 before running MaskCut.
3. **Back-projection.** Mask coordinates from each crop are projected back to the
   original image coordinate system.
4. **Proposal merging.** All proposals across scales are merged using a graph-based
   IoU filter: two masks are connected if IoU > 0.5 or one contains the other
   (containment > 0.7); connected components are unioned.

The full-scale pass (scale 1.0) is identical to the baseline, so the multi-scale run
is a strict superset of the baseline pseudo-labels before merging.

### 3.3 Implementation

*[TODO: note key implementation details — crop ranking, two-stage crop skipping, etc.]*

---

## 4. Experiments

### 4.1 Dataset

We evaluate on a 10-class TinyImageNet subset (500 images, see `04_design_decisions/why_tinyimagenet.md`
for rationale). Both pseudo-label runs use identical inputs. Detector performance is
measured on COCO val2017 (class-agnostic), following the CutLER evaluation protocol.

### 4.2 Pseudo-Label Statistics

| | Baseline | MS-MaskCut |
|---|---|---|
| Images | 500 | 500 |
| Annotations | 748 | **TODO** |
| Masks / image | 1.50 | **TODO** |
| Small masks (< 32² px) | **TODO** | **TODO** |

### 4.3 Detector Training

Cascade Mask R-CNN R50+FPN, trained for 20k iterations on a single A100 80GB.
`IMS_PER_BATCH=8`, `BASE_LR=0.005` (linear scaling from paper's 8-GPU config).
See `04_design_decisions/locked_params.md` for full config.

---

## 5. Results

*[TODO: fill once multiscale training and eval complete. Copy from
`01_results/comparison_table_template.md`.]*

### 5.1 Bounding Box Detection

| Method | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| CutLER baseline | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 |
| MS-MaskCut (ours) | TODO | TODO | TODO | TODO | TODO | TODO |

### 5.2 Instance Segmentation

| Method | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| CutLER baseline | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 |
| MS-MaskCut (ours) | TODO | TODO | TODO | TODO | TODO | TODO |

---

## 6. Discussion

*[TODO: write once results are in.]*

**If APs improves:** The multi-scale pseudo-labels contain more small-object signal.
The cost is [TODO: note APl change, if any], suggesting a precision-recall tradeoff.

**If APs does not improve:** Discuss why — possible explanations include noisy small-object
pseudo-labels, insufficient training iterations, or that the detector's FPN already
handles scale variation sufficiently.

**Failure cases:** *[TODO: add 1–2 concrete examples from `02_visualizations/`.]*

---

## 7. Conclusion

We proposed a multi-scale extension to CutLER's MaskCut pseudo-label generation stage
to address its systematic weakness on small objects. [TODO: 1 sentence result summary.]
Future work includes evaluating on dedicated small-object benchmarks (VisDrone, DOTA),
running self-training rounds with multi-scale pseudo-labels, and learning the scale
merging weights rather than using heuristic IoU thresholds.

---

## References

- Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV.
- Wang et al. (2023). *Cut and Learn for Unsupervised Object Detection and Instance Segmentation* (CutLER). CVPR.
- Lin et al. (2017). *Feature Pyramid Networks for Object Detection* (FPN). CVPR.
- Cai & Vasconcelos (2018). *Cascade R-CNN*. CVPR.
- *[TODO: add any additional references used.]*
