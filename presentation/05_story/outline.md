# 10-Slide Presentation Outline

## Slide 1 — The Problem

**Title:** Unsupervised Object Detection: Can We Find What We Can't Label?

**Content:**
- Object detection normally requires thousands of annotated bounding boxes.
- Annotation is expensive, slow, and domain-specific.
- CutLER (Wang et al., 2023) shows that DINO self-attention + Normalized Cuts can generate
  useful pseudo-labels without any human annotation.
- **Gap:** the pipeline works well for large objects but misses small ones.

**Visuals:** 1–2 COCO images with missed small objects highlighted.

---

## Slide 2 — Why Small Objects Are Hard

**Title:** Single-Scale Bias in Pseudo-Label Generation

**Content:**
- DINO is run on a fixed 480×480 input. A 30×30 object becomes ~4 patches — barely visible.
- Normalized Cuts finds dominant partitions; small objects rarely dominate.
- CutLER baseline: APs = 3.66 (bbox). APl = 29.60. The gap is 8×.
- Standard detection papers also struggle here, but we have no annotations to compensate.

**Visuals:** APs/APl bar chart. Attention map on a small-object image showing sparse response.

---

## Slide 3 — CutLER & MaskCut Explained

**Title:** How CutLER Generates Pseudo-Labels (No Labels Required)

**Content:**
- DINO ViT-S/8: self-supervised, learns to attend to objects without labels.
- MaskCut: treat the attention affinity matrix as a graph → Normalized Cuts → binary mask.
- Iterate N=3 times with background suppression → up to 3 masks per image.
- Train a standard Cascade Mask R-CNN on these pseudo-labeled images.

**Visuals:** Pipeline diagram from `03_diagrams/pipeline_overview.md` (simplified).

---

## Slide 4 — Our Idea: Multi-Scale MaskCut

**Title:** See More by Looking Closer: Multi-Scale Pseudo-Labels

**Content:**
- Crop the original image at 0.75× and 0.5× windows (overlapping, 0.3 overlap).
- Resize each crop to 480×480 and run MaskCut independently.
- Back-project masks to original coordinates.
- Merge across scales via graph-based IoU-NMS (union connected components).
- Result: more masks, concentrated in small objects — without any labels.

**Visuals:** Crop strategy diagram from `03_diagrams/multiscale_strategy.md`.

---

## Slide 5 — Pipeline Diagram

**Title:** Full Pipeline: From Images to Detection

**Content:**
- Full pipeline figure: images → DINO → MaskCut (with our extension highlighted) →
  pseudo-label JSON → Cascade Mask R-CNN training → COCO evaluation.
- Call out: "Everything here is unsupervised."

**Visuals:** Full pipeline diagram from `03_diagrams/pipeline_overview.md`.

---

## Slide 6 — Experimental Setup

**Title:** Controlled Experiment on TinyImageNet

**Content:**
- Dataset: 10-class TinyImageNet subset (500 images). Same for both runs.
- Locked parameters (same backbone, tau, N, fixed_size for both).
- Detector: Cascade Mask R-CNN R50+FPN, 20k iterations, single A100.
- Evaluation: COCO val2017, class-agnostic AP.
- **Key point:** the *only* variable is the pseudo-label generation strategy.

**Visuals:** Small table of locked params. Side-by-side: baseline JSON stats vs multiscale JSON stats.

---

## Slide 7 — Baseline Results

**Title:** Baseline: CutLER Reproduces (and Exceeds) Paper Numbers

**Content:**
- Our reproduction: AP=12.33, APs=3.66, APl=29.60.
- Paper reports 8.3 AP (different checkpoint; ours uses the released `cutler_cascade_final.pth`).
- Strong at large objects, weak at small. This is the gap we target.
- Pseudo-label stats: 748 annotations / 500 images = 1.5 masks/image.

**Visuals:** Results table with paper vs ours. Bar chart: APs, APm, APl for baseline.

---

## Slide 8 — Multi-Scale Results

**Title:** Multi-Scale MaskCut: Does It Improve Small-Object Detection?

**Content:**
- **TODO** — fill once multiscale training + eval complete.
- Expected: ΔAPs > 0, possibly at the cost of some APl.
- Pseudo-label stats: more annotations, smaller mean area.

**Visuals:** Updated comparison table. Side-by-side mask overlay: baseline vs multiscale on same image.

---

## Slide 9 — Analysis & Failure Cases

**Title:** What Works, What Doesn't

**Content:**
- Success: small, isolated objects (fish, butterflies) — more proposals → higher recall.
- Failure: densely packed objects → fragmented masks; background texture → false proposals.
- Graph-based merge helps but can over-union adjacent objects.
- **TODO:** add concrete examples from visualization outputs.

**Visuals:** 2–3 images: one clear win, one failure. (See `02_visualizations/README.md`.)

---

## Slide 10 — Conclusions & Future Work

**Title:** Summary and Next Steps

**Content:**
- We extended CutLER's MaskCut with multi-scale cropping to improve small-object
  pseudo-label generation — fully unsupervised, no extra annotations needed.
- Controlled comparison on TinyImageNet 10-class shows **[TODO: result summary]**.
- Limitations: small training set, no ablation over scale choices, single dataset.
- Future work: larger dataset, self-training rounds, VisDrone evaluation, learned merge weights.

**Visuals:** Clean summary table. One "teaser" image showing the best multiscale result.
