# Multi-Scale MaskCut Guide

This document explains `multiscale/multiscale_maskcut.py`: what each part does, why it exists, and how to run it.

## Goal

`multiscale/multiscale_maskcut.py` extends the original MaskCut pipeline by:

1. Running MaskCut on multiple crops/scales of each image.
2. Mapping crop-level masks back to full-image coordinates.
3. Merging overlapping candidates with area and IoU rules.
4. Writing COCO-style pseudo-mask annotations compatible with the rest of CutLER.

The main motivation is to improve small-object discovery by "zooming in" through smaller crops.

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

## How to Use

## Option A: Direct Python

From repository root:

```bash
python multiscale/multiscale_maskcut.py \
  --vit-arch small \
  --vit-feat k \
  --patch-size 8 \
  --tau 0.2 \
  --N 3 \
  --fixed_size 480 \
  --dataset-path /path/to/train \
  --num-folder-per-job 200 \
  --job-index 0 \
  --pretrain_path /path/to/dino_deitsmall8_pretrain.pth \
  --out-dir /path/to/output \
  --multi-crop \
  --crop-scales 1.0,0.75,0.5 \
  --crop-overlap 0.3 \
  --crop-max-per-scale 9 \
  --merge-iou-thresh 0.5 \
  --keep-topk 20 \
  --min-mask-area-ratio 0.0005 \
  --max-mask-area-ratio 1.0 \
  --small-first
```

## Option B: SLURM script

Use:

1. `slurm/run_multiscale_maskcut.sh`

Example:

```bash
TAU=0.2 \
N_MASKS=3 \
CROP_SCALES=1.0,0.75,0.5 \
CROP_OVERLAP=0.3 \
CROP_MAX_PER_SCALE=9 \
MERGE_IOU_THRESH=0.5 \
KEEP_TOPK=20 \
MIN_MASK_AREA_RATIO=0.0005 \
MAX_MASK_AREA_RATIO=1.0 \
SMALL_FIRST=1 \
sbatch slurm/run_multiscale_maskcut.sh
```

## Practical starting presets

For small-object emphasis (first sweep):

1. `tau`: `0.15` and `0.2`
2. `N`: `3` and `5`
3. `crop_scales`: `1.0,0.75,0.5` then add `0.35` if budget allows
4. `crop_overlap`: `0.3`
5. `merge_iou_thresh`: `0.5`
6. `small_first`: enabled

## Notes and limitations

1. Compute cost increases with number of windows and scales.
2. Very aggressive scales/overlap can create many near-duplicate masks.
3. `keep_topk` is useful to bound annotation volume.
4. This script keeps output format compatible with CutLER training.
