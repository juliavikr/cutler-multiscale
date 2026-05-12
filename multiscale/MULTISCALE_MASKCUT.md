# Multi-Scale MaskCut Guide

This document explains `multiscale/multiscale_maskcut.py`: what each part does, why it exists, and how to run it.

## Goal

`multiscale/multiscale_maskcut.py` extends the original MaskCut pipeline by:

1. Running MaskCut on multiple crops/scales of each image.
2. Mapping crop-level masks back to full-image coordinates.
3. Merging overlapping candidates with area and IoU rules.
4. Writing COCO-style pseudo-mask annotations compatible with the rest of CutLER.

The main motivation is to improve small-object discovery by "zooming in" through smaller crops.

Important framing: in the final project pipeline, these crop-level masks are most useful as **rescue additions** to the original MaskCut masks, not as a replacement for them.

## Final comparison

The project compared these two methods:

| Method | Role | Required flags |
| --- | --- | --- |
| Baseline MaskCut | Full-image reference point | no `--multi-crop` |
| Hybrid heatmap | Main multiscale method | `--multi-crop --ms-preset small` |

The script always writes split outputs. For training and quantitative
comparison, use the `multiscale` split for the hybrid method. The `combined`
split (normal + crop masks together) is for diagnostic use only — it can
contain overlapping masks that confuse detector training.

## File Structure and Logic

## 1) Imports and preprocessing

Key setup:

1. Loads DINO feature extractor utilities and TokenCut object discovery tools.
2. Loads CRF postprocessing (`densecrf`) and morphology (`binary_fill_holes`).
3. Defines `ToTensor` normalization expected by DINO.

Why:

1. MaskCut works on DINO patch features.
2. CRF and hole-filling reduce noisy mask boundaries.

## 2) Core MaskCut functions (baseline behavior)

Functions:

1. `get_affinity_matrix`
2. `second_smallest_eigenvector`
3. `get_salient_areas`
4. `check_num_fg_corners`
5. `get_masked_affinity_matrix`
6. `maskcut_forward`

What they do:

1. Build a patch graph from cosine similarity.
2. Solve a spectral partitioning problem.
3. Convert eigenvector output into a binary foreground/background split.
4. Apply orientation heuristics so object/background sides are consistent.
5. Iterate up to `N` masks by painting out already used regions.

Why:

1. This is the original MaskCut logic and remains unchanged in spirit.
2. Reusing this block ensures compatibility with existing CutLER pseudo-label generation.

## 3) Single-image wrappers

Functions:

1. `maskcut`
2. `maskcut_from_pil`

What they do:

1. Read image.
2. Resize to `fixed_size`.
3. Extract DINO features.
4. Call `maskcut_forward`.

Why:

1. `maskcut_from_pil` allows running MaskCut on both full images and cropped PIL sub-images.

## 4) Multi-crop window generation

Function:

1. `generate_windows`

Inputs:

1. `image_size` (fixed square canvas size)
2. `crop_scales` (for example `1.0,0.75,0.5`)
3. `crop_overlap` (controls sliding-window stride)
4. `max_windows_per_scale` (optional cap)

What it does:

1. Converts each scale into a crop size.
2. Creates sliding windows with stride `crop_size * (1 - overlap)`.
3. Ensures edge coverage by forcing final border windows.
4. Optionally limits count per scale.

Why:

1. Multiple scales increase chance of isolating small objects.
2. Overlap prevents missing objects near crop borders.

## 5) Crop-level filtering and merge

Functions:

1. `binary_iou`
2. `postprocess_crop_mask`
3. `merge_masks`

What they do:

1. `postprocess_crop_mask` runs CRF + hole filling and rejects unstable masks (low IoU vs pre-CRF mask).
2. `merge_masks` removes very tiny/very large masks, suppresses near-duplicates by IoU, and can prioritize small masks first.

Key controls:

1. `merge_iou_thresh`
2. `min_mask_area_ratio`
3. `max_mask_area_ratio`
4. `keep_topk`
5. `small_first`

Why:

1. Multi-crop produces many overlapping proposals.
2. The merge stage keeps quality while preventing annotation explosion.
3. `small_first` can help APs-focused setups.

## 6) Multi-scale execution pipeline

Function:

1. `maskcut_multicrop`

Step-by-step:

1. Resize full image to fixed square canvas.
2. Build crop windows across scales.
3. Run MaskCut per crop using `maskcut_from_pil`.
4. Refine each crop mask with CRF.
5. Project crop mask back to full-image coordinates.
6. Merge all candidates globally.

Output:

1. Final merged full-canvas binary masks.
2. The fixed-size image used for processing.

## 7) COCO conversion and JSON writing

Functions:

1. `create_image_info`
2. `create_annotation_info`

What they do:

1. Convert binary masks to COCO RLE.
2. Compute area and bbox.
3. Append to global `output` dictionary.
4. Save JSON at end.

Filename behavior:

1. Baseline-style names when not using multi-crop.
2. Multi-crop tag appended when `--multi-crop` is enabled:
   `_mc{scales}_ov{overlap}_miou{merge_iou}`

## CLI Arguments (Multi-scale specific)

Enable mode:

1. `--multi-crop`

Crop generation:

1. `--crop-scales`
2. `--crop-overlap`
3. `--crop-max-per-scale`

Merge/filter:

1. `--merge-iou-thresh`
2. `--keep-topk`
3. `--min-mask-area-ratio`
4. `--max-mask-area-ratio`
5. `--small-first`

Baseline controls still apply:

1. `--tau`
2. `--N`
3. `--fixed_size`
4. `--dataset-path`
5. `--cpu`

## How to Run

### Option A: SLURM Commands

### Baseline

```bash
sbatch slurm/run_maskcut_baseline.sh
```

### Hybrid heatmap

```bash
DATASET_PATH="${HOME}/data/tiny-imagenet-10classes/train" \
OUT_DIR="${HOME}/data/tiny-imagenet-10classes/annotations" \
DINO_WEIGHTS="${HOME}/data/weights/dino_deitsmall8_pretrain.pth" \
TAU=0.15 \
N_MASKS=3 \
FIXED_SIZE=480 \
NUM_FOLDER_PER_JOB=10 \
MS_PRESET=small \
PRIMARY_OUTPUT=multiscale \
sbatch slurm/run_multiscale_maskcut.sh
```

### Option B: Direct Python Flags

From repository root, baseline is the same command without `--multi-crop`.

Hybrid heatmap adds:

```text
--multi-crop --ms-preset small --primary-output multiscale
```

## Parameters and Tuning

The `small` preset values were chosen based on the 100-image ablation study
(see `PROJECT_NOTES.md` §Ablation and `results/hybrid_ablation_100_summary.csv`).
This section documents the rationale for the key non-obvious values.

### Crop sizes: `0.25, 0.35, 0.50`

Three scales give a range from small (120 px at 480×480 input) to medium (240 px).
Larger crops (e.g. 0.75) recovered objects already found by the full-image pass;
smaller crops (< 0.20) were too small for MaskCut to find stable boundaries.

### Crop budget: `heatmap_top_k=12`, `heatmap_spatial_rescue=4`

12 total windows = 8 heatmap-guided + 4 spatial rescue. The ablation showed that
reducing to `topk8` (8 total) produced cleaner masks with fewer duplicates, while
`topk12` was the best balance of recall and noise for the 5-class detector study.
The 4 rescue slots ensure corners and image borders are always sampled.

### Heatmap percentile: `heatmap_percentile=85`

Only the top 15% of heatmap activations are used as crop centres. The ablation
showed that changing this to 80 or 90 had no effect on the 100-image subset,
meaning the specific threshold matters less than having one. 85 is a safe default.

### Area filter: `min_area_ratio=0.0001`, `max_area_ratio=0.02`

- `0.0001` (0.01%) filters out single-patch noise masks (< ~7×7 px at 480×480).
- `0.02` (2%) filters out full-image masks that would duplicate the baseline output.
  Anything larger than ~96×96 px at 480×480 is likely a large object already
  found by the full-image pass.

### Containment threshold: `containment_thresh=0.85`

Intersection-over-smaller: if 85% of a smaller mask overlaps a larger one, the
smaller mask is considered contained and dropped. This is stricter than IoU because
it catches the common case of a crop mask that is a subset of a full-image mask.

### CRF stability filter: `crf_iou_thresh=0.3`

If the IoU between the raw MaskCut bipartition and the CRF-refined mask is below 0.3,
the mask is rejected as unstable. The value is permissive by design — it only removes
masks where CRF disagreed on more than 70% of pixels, which indicates the bipartition
had no clear boundary signal. Raising this (e.g. to 0.5) rejects more masks and
reduces noise, at the cost of recall.

### When to override the preset

Start with `--ms-preset small`. Override individual parameters only after
identifying a specific failure mode in the `candidate_debug` output:

| Symptom | Likely cause | Override |
|---------|-------------|----------|
| Too many duplicate masks | Low merge IoU threshold | Raise `--merge-iou-thresh` |
| Missing objects at image border | Low rescue quota | Raise `--heatmap-spatial-rescue` |
| Too many tiny fragments | Low area minimum | Raise `--min-mask-area-ratio` |
| Crops covering dominant object | Area max too high | Lower `--max-mask-area-ratio` |
| CRF rejecting too many masks | Threshold too strict | Lower `--crf-iou-thresh` |

## Notes and limitations

1. Compute cost increases with number of windows and scales.
2. Very aggressive scales/overlap can create many near-duplicate masks.
3. `keep_topk` is useful to bound annotation volume.
4. This script keeps output format compatible with CutLER training.
5. Do not compare runs with different `tau`, `N`, `fixed_size`, DINO weights, or
   image subsets.
