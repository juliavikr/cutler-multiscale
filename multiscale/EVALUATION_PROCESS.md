# Evaluation Process for Multi-Scale MaskCut

This document defines how to evaluate the current MaskCut variants and decide
which one is best for the project. The main question is:

```text
Does the multiscale method add useful small-object masks that normal MaskCut
misses, without adding too many fragments, duplicates, or background masks?
```

The evaluation should separate three ideas:

1. Did we produce more small masks?
2. Are those small masks actually correct?
3. Does training on them improve downstream small-object performance?

More masks per image is not enough. The output is better only if the extra masks
correspond to real object instances and do not damage pseudo-label quality.

## Methods To Compare

Run every method on the same images with the same base MaskCut settings.

| Method | Output / flags | Purpose |
| --- | --- | --- |
| Normal MaskCut | `normal` split | Full-image baseline |
| Hybrid heatmap multiscale | `--multi-crop --ms-preset small` | Current main candidate |
| MOST-lite multiscale | `--multi-crop --ms-preset mostlite` | Experimental token-cluster proposal candidate |
| Raw multiscale | `raw_multiscale` split | Debug crop-mask discovery before final filtering |
| Combined | `combined` split | Diagnostic view of normal + crop masks |

For final claims, compare `normal`, `hybrid multiscale`, and `MOST-lite`.
Use `raw_multiscale` and `combined` mainly to understand failures.

## Evaluation Levels

Use two complementary evaluations.

### Level 1: Direct Pseudo-Mask Evaluation

This measures pseudo-label quality directly by comparing generated masks to
ground-truth instance masks.

Use this when the image dataset has instance segmentation annotations, such as
COCO val2017. The matching should be class-agnostic: a predicted mask only needs
to overlap a ground-truth object, regardless of category.

This is the best way to answer:

```text
Did multiscale improve small-object mask discovery?
```

### Level 2: Downstream Detector Evaluation

This measures whether the pseudo-labels improve the final trained detector.

Train the same detector with pseudo-labels from each method, then evaluate on
COCO using the standard metrics:

- AP
- AP50
- AP75
- APs
- APm
- APl

This is the best way to answer:

```text
Did better pseudo-masks lead to better detection/segmentation performance?
```

Direct pseudo-mask quality and downstream AP may disagree. A method can find more
small masks but still hurt training if the masks are noisy.

## Dataset Setup

### Recommended Direct Evaluation Dataset

Use a small COCO val2017 subset with ground-truth masks.

Good starting sizes:

| Size | Purpose |
| --- | --- |
| 20 images | fast debugging and visual inspection |
| 100 images | preliminary quantitative comparison |
| 500 images | stronger project result if compute allows |

Prefer images containing multiple small objects. COCO already includes size bins,
so the evaluation can report small / medium / large object performance.

### Training Dataset

For downstream detector evaluation, keep the training dataset fixed across
methods. Only change the pseudo-label source.

Example:

| Run | Training pseudo-labels | Detector config | Eval set |
| --- | --- | --- | --- |
| Baseline | normal MaskCut | same | COCO val2017 |
| Hybrid | hybrid multiscale | same | COCO val2017 |
| MOST-lite | MOST-lite multiscale | same | COCO val2017 |

Every training hyperparameter should stay identical. Otherwise the comparison is
not attributable to the pseudo-label method.

## Direct Pseudo-Mask Metrics

### Size Bins

Use COCO object area definitions:

```text
small:  area < 32^2
medium: 32^2 <= area < 96^2
large:  area >= 96^2
```

These bins are based on ground-truth object area, not predicted mask area.

### Best-IoU Recall

For every ground-truth object, compute the best IoU against any predicted mask in
the same image.

```text
best_iou(gt) = max IoU(gt_mask, predicted_mask)
```

Then report:

| Metric | Meaning |
| --- | --- |
| Small Recall@0.25 | loose small-object discovery |
| Small Recall@0.50 | standard useful-mask threshold |
| Small Recall@0.75 | strict mask-quality threshold |
| Small Mean Best IoU | average best overlap for small GT objects |
| Medium Recall@0.50 | check that medium objects are not harmed |
| Large Recall@0.50 | check that large objects are not harmed |

For this project, the most important metric is:

```text
Small Recall@0.50
```

If hybrid or MOST-lite improves this without extreme noise, that is a strong sign
the method is doing what it was designed to do.

### Average Recall At K

Limit each image to the top K predicted masks and compute recall.

Suggested K values:

```text
K = 1, 3, 5, 10, 20
```

This helps answer whether the method finds useful masks efficiently, not only by
dumping many proposals.

Example table:

| Method | AR@1 small | AR@5 small | AR@10 small | AR@20 small |
| --- | ---: | ---: | ---: | ---: |
| Normal | | | | |
| Hybrid | | | | |
| MOST-lite | | | | |

### Predicted Mask Statistics

Report these without using ground truth:

| Metric | Why it matters |
| --- | --- |
| Masks per image | output density |
| Mean predicted area | whether masks became smaller |
| Median predicted area | robust area summary |
| Percent predicted small masks | whether method shifts toward small masks |
| Percent masks touching crop border | crop truncation risk |
| Runtime per image | practical feasibility |

These are useful, but they do not prove quality. They should support the
ground-truth recall metrics.

## Noise Metrics

Small-object recall can improve while pseudo-label noise also increases. Track
noise explicitly.

### Duplicate Rate

A duplicate happens when multiple predicted masks match the same ground-truth
object.

One practical definition:

```text
duplicate_count = number of extra predicted masks with IoU >= 0.5 to a GT object
                  after the first matched prediction
duplicate_rate = duplicate_count / number of matched GT objects
```

High duplicate rate means the method repeatedly finds the same object.

### Fragment Rate

A fragment happens when several predicted masks cover pieces of the same object
but no single prediction covers it well.

One practical definition:

```text
fragmented_gt = GT object where:
  max IoU with any single prediction < 0.5
  but union IoU of overlapping predictions >= 0.5
```

High fragment rate means the method is finding parts rather than full objects.

### Over-Merge Rate

An over-merge happens when one predicted mask covers multiple GT objects.

One practical definition:

```text
overmerged_prediction = predicted mask with IoU >= 0.25 against two or more GT objects
```

This is important because earlier results showed masks that grouped multiple
people or objects together.

## Manual Visual Rubric

For a small number of debug images, manually score each method. This is useful
when working with images that do not have ground-truth masks.

Use this table per image:

| Category | Count |
| --- | ---: |
| Good small-object masks | |
| Good medium/large masks | |
| Partial object fragments | |
| Duplicate masks | |
| Background masks | |
| Over-merged masks | |
| Missed obvious small objects | |

The key qualitative comparison is:

```text
good small-object masks - fragments - duplicates - background masks
```

This should not replace quantitative evaluation, but it is very useful for
understanding the method before running expensive training.

## Presentation Visualizations

Build visuals that tell the story clearly.

### Figure 1: Method Overlay Grid

Use the same image in four panels:

```text
Normal | Hybrid multiscale | MOST-lite | Combined
```

Purpose:

- shows what normal MaskCut already finds,
- shows what hybrid adds,
- shows what MOST-lite changes,
- shows why combined is useful visually but risky for training.

Use 2-3 strong examples:

- one success case,
- one mixed case,
- one failure case.

### Figure 2: Raw vs Final Multiscale

Use:

```text
Raw multiscale | Final multiscale
```

Purpose:

- shows how much filtering/merging changes the output,
- reveals whether good masks are being removed,
- explains the pipeline honestly.

This is especially useful because the pipeline can produce many raw crop masks
but keep only a few final masks.

### Figure 3: Crop Proposal Comparison

Draw crop boxes on the original image:

```text
Hybrid heatmap crops | MOST-lite token-cluster crops
```

Purpose:

- explains that the main difference is crop proposal selection,
- makes the architecture visually understandable,
- shows whether one method over-focuses on the center or misses side objects.

### Figure 4: Mask Area Distribution

Plot predicted mask areas for each method.

Recommended chart:

```text
x-axis: mask area in pixels, log scale
y-axis: count or density
color: method
```

Purpose:

- shows whether multiscale actually shifts the output toward smaller masks,
- helps explain why this targets APs.

### Figure 5: Recall By Object Size

Bar chart:

```text
x-axis: object size bin: small, medium, large
y-axis: Recall@0.50
color: method
```

Purpose:

- strongest quantitative slide,
- directly answers whether small-object mask discovery improved,
- also shows whether large-object performance was harmed.

### Figure 6: Precision-Recall Style Tradeoff

Scatter plot:

```text
x-axis: masks per image
y-axis: Small Recall@0.50
color: method
```

Purpose:

- shows whether a method is efficient,
- distinguishes "more masks" from "better masks."

An ideal method is up and left: high recall with fewer masks.

### Figure 7: Failure Taxonomy

Create a small slide with examples of:

- missed small object,
- duplicate crop masks,
- partial fragment,
- over-merged group mask,
- crop-border truncation.

Purpose:

- shows honest analysis,
- helps justify future work,
- makes the project look methodologically mature.

## Decision Criteria

Pick the best method using this order:

1. Best Small Recall@0.50.
2. Small Mean Best IoU does not decrease.
3. Duplicate and fragment rates stay reasonable.
4. Masks per image stays manageable.
5. Runtime is feasible for the planned dataset.
6. Downstream APs improves or at least does not harm overall AP.

Suggested rule:

```text
Choose a multiscale method only if it improves Small Recall@0.50 by at least
10-15% relative to normal MaskCut without doubling the duplicate/fragment rate.
```

For the final project, the strongest result would be:

```text
Hybrid or MOST-lite improves pseudo-mask Small Recall@0.50 and improves detector
APs after training.
```

If detector APs does not improve, the project can still be valuable if the direct
pseudo-mask evaluation clearly shows better small-object discovery and the report
explains why training may need cleaner pseudo-label filtering.

## Recommended Experiment Order

### Step 1: Single-Image Debug

Run all methods on one known debug image.

Check:

- overlays,
- raw vs final masks,
- crop proposal boxes,
- candidate debug records.

Goal:

```text
Make sure the outputs are visually reasonable before scaling up.
```

### Step 2: Small COCO Subset

Run direct pseudo-mask evaluation on 20 COCO images with many small objects.

Report:

- Small Recall@0.50,
- Small Mean Best IoU,
- masks per image,
- duplicate rate,
- runtime per image.

Goal:

```text
Quickly reject bad variants.
```

### Step 3: Larger COCO Subset

Run direct evaluation on 100-500 COCO val images.

Goal:

```text
Get stable numbers for the presentation/report.
```

### Step 4: Detector Training

Train with pseudo-labels from the best method and compare against normal MaskCut.

Goal:

```text
Measure whether pseudo-mask improvements transfer to APs/APm/AP.
```

## Tables To Include In The Report

### Pseudo-Mask Quality Table

| Method | Masks/img | Small R@0.5 | Small R@0.75 | Small Best IoU | Duplicate Rate | Fragment Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | | | | | | |
| Hybrid | | | | | | |
| MOST-lite | | | | | | |

### Size-Binned Recall Table

| Method | Small R@0.5 | Medium R@0.5 | Large R@0.5 |
| --- | ---: | ---: | ---: |
| Normal | | | |
| Hybrid | | | |
| MOST-lite | | | |

### Downstream Detector Table

| Training labels | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal pseudo-labels | | | | | | |
| Hybrid pseudo-labels | | | | | | |
| MOST-lite pseudo-labels | | | | | | |

## What A Good Result Looks Like

A good result does not need to find every small object. It should show:

- normal MaskCut is stable but misses small objects,
- multiscale crop MaskCut recovers additional small objects,
- the final filters remove many bad crop masks,
- the best multiscale method improves small-object recall,
- noise remains controlled enough to justify training or future refinement.

An ideal presentation conclusion:

```text
Multi-scale MaskCut improves small-object pseudo-mask discovery by using DINO
features to choose informative local crops. The hybrid heatmap method gives the
best reliability/speed tradeoff, while MOST-lite is a promising token-cluster
proposal direction for cleaner future crop selection.
```
