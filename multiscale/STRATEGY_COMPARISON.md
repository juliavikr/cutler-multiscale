# Multi-Scale MaskCut Strategy Comparison

This document explains the MaskCut variants currently used or worth testing in this
project. The goal is not to maximize the number of masks per image. The practical
goal is to add a small number of useful small-object pseudo-masks without breaking
the masks that normal CutLER/MaskCut already finds.

## Short Recommendation

For the next experiments, compare these three outputs on the same image set:

| Strategy                  | Script / flags                                                              | Best use                                          |
| ------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------- |
| Normal MaskCut            | `multiscale_maskcut.py` without `--multi-crop`, or the `normal` split | Baseline quality check                            |
| Hybrid heatmap multi-crop | `--multi-crop --ms-preset small`                                          | Current main candidate                            |
| MOST-lite v2 soft         | `--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45`                | Experimental token-cluster candidate              |

Keep `combined` as a diagnostic output for now. It is useful visually, but it can
create overlapping or hierarchical pseudo-labels that may confuse detector
training.

## Common Pipeline Pieces

All multi-crop variants share the same high-level structure:

1. Run normal full-image MaskCut.
2. Generate crop boxes that might contain missed objects.
3. Run MaskCut independently inside selected crops.
4. Project crop masks back into the original image.
5. Score and filter crop masks.
6. Merge duplicates or fragments.
7. Write split outputs:
   - `normal`: only full-image MaskCut masks.
   - `raw_multiscale`: all crop masks before final merging.
   - `multiscale`: filtered and merged crop masks only.
   - `combined`: normal masks plus multiscale masks.
   - `candidate_debug`: crop/mask metadata for debugging.

The core question is step 2: how should we decide which crops deserve another
MaskCut pass?

## Strategy 1: Normal Full-Image MaskCut

### What It Does

Normal MaskCut runs DINO feature extraction on the full image, builds an affinity
graph over image patches, and uses normalized cuts to separate foreground-like
regions. It can iterate up to `--N` times, suppressing previous foreground regions
between iterations.

In our split output system, this is saved as:

```text
*_normal.json
```

### Why It Works

Normal MaskCut is strong when the target object is visually dominant at the full
image scale. Large central objects, people, animals, vehicles, and high-contrast
foreground regions tend to appear clearly in the DINO patch graph.

### Why It Fails For Small Objects

Small objects occupy too few ViT patches. Even with ViT-S/8, an object that is
20-30 pixels wide may only cover a few tokens after resizing. Normalized cuts then
prefer the dominant global partition, not the small local object.

It can also merge multiple nearby objects into one mask when the feature graph sees
them as one coherent foreground group.

### Pros

- Fastest and simplest strategy.
- Most stable masks.
- Less likely to produce many tiny noisy fragments.
- Best baseline for fair comparison.
- Good for preserving the original CutLER behavior.

### Cons

- Misses many small foreground objects.
- Often finds only the dominant object.
- Can group several objects into one large mask.
- Does not directly exploit local zoom.

### When It Is Better

Use normal MaskCut when you care about precision and stability more than small
object recall. It is also the correct baseline for measuring whether multiscale
actually helps.

### When It Is Worse

It is worse when the image contains multiple small tabletop, shelf, street, or
foreground objects where no single small object dominates the full image.

## Strategy 2: Legacy Dense-Grid Multi-Crop

### What It Does

The legacy multi-crop strategy creates a fixed set of sliding-window crops across
the image at several crop scales. MaskCut is run in every crop or in many crops.
The crop masks are then projected back to the full image and merged.

This is the older high-recall behavior preserved in:

```text
multiscale/multiscale_maskcut_legacy.py
```

and conceptually available through:

```text
--multi-crop --ms-preset legacy
```

or:

```text
--multi-crop --crop-mode grid
```

### Why It Works

Cropping makes small objects larger relative to the MaskCut input. An object that
was tiny in the full image can become a meaningful foreground region in a crop.
This directly attacks the patch-resolution problem.

### Main Problem

The grid has no idea where objects are. It spends compute on empty background,
large dominant objects, and repeated views of the same object. This can produce
many masks, but many are duplicates, fragments, or crop-border artifacts.

### Pros

- Strong small-object recall.
- Simple and easy to reason about.
- Does not depend on a proposal heatmap being correct.
- Can find side objects that heatmap modes may miss.
- Good as an upper-bound diagnostic for "can crop MaskCut ever see this object?"

### Cons

- Slow because it runs MaskCut on many crops.
- Repeats the same dominant object across overlapping crops.
- Produces many duplicate or partial masks.
- More sensitive to merging thresholds.
- Can flood training with noisy tiny masks.

### When It Is Better

It is better when the priority is exploratory recall: finding out whether local
zoom helps at all, even if output quality is messy.

### When It Is Worse

It is worse for pseudo-label training if the merged output contains too many
fragments. More annotations are not automatically better if many are unstable or
object parts.

## Strategy 3: Hybrid Heatmap-Guided Multi-Crop

### What It Does

This is the current main multiscale strategy. It is preserved as:

```text
multiscale/multiscale_maskcut_hybrid.py
```

and it is also the default behavior in the main file with:

```text
--multi-crop --ms-preset small
```

The strategy uses one full-image DINO feature pass to build a cheap crop proposal
heatmap. The heatmap is based mainly on local DINO feature contrast: patches whose
features differ from their neighbors are treated as possible object boundaries or
small-object islands.

The pipeline is:

1. Run full-image MaskCut.
2. Build a DINO feature-contrast heatmap.
3. Downweight regions already covered by full-image masks.
4. Select high-scoring crop windows around heatmap peaks.
5. Add a small spatial rescue quota so crop proposals are not only from the image
   center.
6. Run MaskCut on selected crops.
7. Score crop masks by area prior, compactness, crop score, CRF agreement, border
   touch, aspect ratio, and duplicate overlap.
8. Merge the best crop masks.

### Why It Works

It keeps the main benefit of cropping while avoiding a full dense grid. It tries to
spend MaskCut calls where DINO features suggest local structure and where normal
MaskCut did not already explain the image.

This is aligned with the project goal: add useful small masks, not simply add many
masks.

### Why It Can Fail

The heatmap can become biased toward the most textured or central region. If an
object is visually low-contrast, near the image border, or already partly covered
by a normal mask, it may not receive a crop.

Also, feature contrast is not the same as objectness. Textured background can look
interesting, while smooth small objects can be missed.

### Pros

- Much more efficient than dense grid.
- More targeted toward likely missed objects.
- Produces split outputs and candidate debug metadata.
- Better controlled than old high-recall grid output.
- Good current default for small-object pseudo-label experiments.

### Cons

- Can miss side objects if the heatmap peaks are concentrated elsewhere.
- Depends on hand-designed scoring weights.
- Still has many parameters, even with presets.
- Can reject real masks if filters are too strict.
- Can keep object parts if crop boundaries cut through an object.

### When It Is Better

It is better when you want a practical compromise between recall, speed, and mask
quality. This should be the first serious multiscale candidate to compare against
normal MaskCut.

### When It Is Worse

It is worse when the heatmap proposal stage misses the object entirely. If no crop
is proposed around an object, crop MaskCut never gets a chance to find it.

## Strategy 4: Hybrid Combined Output

### What It Does

The `combined` split unions two sources:

```text
normal masks + multiscale crop masks
```

It is saved as:

```text
*_combined.json
```

### Why It Seems Attractive

It preserves what already works from normal MaskCut while adding small-object
masks recovered from crops. This matches the intuitive project goal.

### Main Risk

The same physical object can appear as:

- one large normal mask,
- one crop mask for an object part,
- several crop fragments,
- or a smaller mask inside a larger mask.

That creates hierarchical overlaps. A detector trained on pseudo-labels may see
conflicting targets: should the full group be one object, or should each smaller
piece be a separate object?

### Pros

- Highest visual recall.
- Preserves normal MaskCut results.
- Useful for qualitative inspection.
- Helps see whether multiscale is adding anything new.

### Cons

- Risky for detector training before overlap hierarchy is handled.
- Can double-count objects.
- Can mix object-level and part-level masks.
- Needs stronger deduplication or containment logic.

### When It Is Better

It is better as a debugging view. It is also useful when manually inspecting
whether crop masks complement normal masks.

### When It Is Worse

It is worse as a training target if overlapping masks are not filtered carefully.
For now, do not use `combined` as the main pseudo-label JSON unless we explicitly
decide how to handle nested overlaps.

## Strategy 5: Raw Multiscale Output

### What It Does

The `raw_multiscale` split stores crop masks before the final merge/filter stage:

```text
*_raw_multiscale.json
```

It answers a different question from `multiscale`: not "what should we train on?",
but "what did the crop runs actually produce?"

### Why It Matters

If 30 crop candidates become 3 final masks, the raw output shows whether the
missing masks were never discovered or were discovered and then filtered out.

This is important because our failures can happen in two places:

1. Crop proposal failure: the object never gets cropped.
2. Mask filtering/merging failure: the object is found but rejected or merged away.

### Pros

- Best diagnostic for understanding lost masks.
- Helps tune filters without rerunning MaskCut.
- Shows duplicates and crop-border artifacts directly.
- Makes the pipeline less opaque.

### Cons

- Too noisy for training.
- Contains duplicates and unstable masks.
- Can overstate how good the actual final method is.

### When It Is Better

Use it when evaluating whether the crop stage is capable of finding small objects.

### When It Is Worse

Do not treat it as a final method. It is a debugging artifact.

## Strategy 6: MOST-Lite v2 Soft Token-Cluster Crop Proposals

### What It Does

MOST-lite is the new experimental architecture in:

```text
multiscale/multiscale_maskcut.py
```

Run the project candidate, MOST-lite v2 soft, with:

```text
--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45
```

The plain preset defaults to v2 strict behavior:

```text
--multi-crop --ms-preset mostlite
```

The only difference between v2 strict and v2 soft is the CRF agreement threshold:
v2 strict uses `--crf-iou-thresh 0.5`, while v2 soft uses `0.45`. We chose v2
soft as the MOST-lite candidate because it keeps the v2 cleanup behavior but is
slightly less likely to discard useful crop masks.

It is inspired by object proposal methods that use DINO token structure more
directly instead of selecting crops from a generic grid or contrast heatmap.

The current lightweight implementation does this:

1. Extract one full-image DINO feature grid.
2. Compute local feature contrast.
3. Estimate a border/background feature prototype.
4. Score tokens by contrast plus distance from the border/background prototype.
5. Pick high-objectness token seeds.
6. Grow connected token components using feature similarity.
7. Reject components that are too tiny or too large.
8. Convert compact token components into crop boxes.
9. Run MaskCut only on those selected crop boxes.
10. Reuse the existing crop-mask scorer and merge logic.

The current `mostlite` preset adds stricter crop post-processing than the first
prototype:

- full-image MaskCut keeps `--N 3`, while crop MaskCut defaults to `--crop-n 1`;
- only the best candidate is kept per crop proposal with `--crop-keep-per-window 1`;
- internally border-touching crop masks are retried on larger crops;
- crop-shaped masks are rejected when they fill the crop and touch internal crop
  borders;
- masks are scored against the DINO token cluster that proposed the crop;
- CRF agreement defaults to `--crf-iou-thresh 0.5`, closer to original MaskCut
  post-processing. For the project comparison, override this to `0.45` to run
  v2 soft.

### Why It Might Be Better

The hybrid heatmap chooses crop centers from local contrast peaks. MOST-lite tries
to form object-like token groups before choosing crop boxes. This can be better
because the proposal is not just "this point is interesting"; it is closer to
"this compact region of DINO tokens behaves like a foreground object."

That may reduce wasted crops and may reduce repeated crops around the same large
dominant object.

### Why It Might Be Worse

It is still not full MOST. It does not implement a full object discovery method,
tracking, learned objectness, or a mature proposal-ranking stage. It is a cheap
proposal generator bolted onto our existing MaskCut pipeline.

It can also fail if the background border prototype is wrong. For example, if the
object touches the image border, or the border contains foreground clutter, then
"different from border" is a weak objectness cue.

The token-cluster step also depends on similarity thresholds. Too strict means
fragmented crops. Too loose means large mixed regions.

### Pros

- More object-like crop proposals than plain heatmap peaks.
- Still uses only one full-image DINO feature pass before crop MaskCut.
- May reduce redundant crops around the same dominant object.
- Better conceptual fit for small-object discovery than dense grid.
- Keeps the current scoring, split outputs, and candidate debug system.

### Cons

- More experimental and less tested than the hybrid heatmap mode.
- Adds assumptions about border/background features.
- Can miss objects that are smooth, low-contrast, or border-touching.
- Can produce poor clusters if DINO features group object and background together.
- Still has threshold parameters, though presets hide most of them.

### When It Is Better

It may be better when the image has many compact objects and the DINO token space
separates them cleanly from the background.

### When It Is Worse

It may be worse on cluttered scenes where the border is not background, or when
the object is defined more by shape than by a distinct DINO feature cluster.

## Main Differences

| Dimension             | Normal       | Legacy grid           | Hybrid heatmap              | MOST-lite v2 soft             |
| --------------------- | ------------ | --------------------- | --------------------------- | ----------------------------- |
| Crop proposal         | None         | Dense sliding windows | DINO feature-contrast peaks | DINO token clusters           |
| Compute cost          | Low          | High                  | Medium                      | Medium                        |
| Small-object recall   | Low          | High                  | Medium-high                 | Unknown, likely medium-high   |
| Noise risk            | Low          | High                  | Medium                      | Medium                        |
| Duplicate risk        | Low          | High                  | Medium                      | Lower if clusters behave well |
| Side-object coverage  | Not targeted | Strong                | Depends on heatmap/rescue   | Depends on token seeds        |
| Parameter sensitivity | Low          | Medium                | High                        | High                          |
| Best role             | Baseline     | Recall diagnostic     | Main candidate              | Experimental comparison       |

## Why Presets Matter

There are many thresholds because the pipeline has several independent failure
points:

- which crops are selected,
- how many crop masks are produced,
- which masks are filtered,
- how duplicates are merged,
- and whether normal masks are allowed to suppress crop masks.

The presets are meant to reduce manual tuning:

| Preset       | Crop mode | Purpose                                                       |
| ------------ | --------- | ------------------------------------------------------------- |
| `small`    | heatmap   | Current default, favors small masks and score-first filtering |
| `balanced` | heatmap   | Less aggressive, useful if `small` is too strict            |
| `mostlite` | mostlite  | v2 strict by default; add `--crf-iou-thresh 0.45` for v2 soft |
| `legacy`   | grid      | Older dense-grid behavior for recall comparison               |

When comparing methods, prefer changing the preset first. Only tune individual
thresholds after identifying a specific failure mode in `candidate_debug`.

## Evaluation Questions

For each debug image, inspect the outputs in this order:

1. `normal`: What does full-image MaskCut already get right?
2. `raw_multiscale`: Did crop MaskCut find plausible small objects at all?
3. `multiscale`: Did filtering keep the good crop masks?
4. `combined`: Do crop masks add useful objects without creating harmful overlaps?
5. `candidate_debug`: Were missed objects caused by crop selection, filtering, or
   merging?

Useful stats to record:

- number of normal masks,
- number of raw crop masks,
- number of final multiscale masks,
- number of combined masks,
- masks touching crop border,
- duplicate masks removed by merge,
- visually good small-object masks,
- visually bad fragments.

## Practical Next Steps

1. Run the same images with baseline, hybrid heatmap, and MOST-lite v2 soft.
2. Convert all split outputs to overlays.
3. Compare `raw_multiscale` vs `multiscale` to see whether good masks are being
   filtered out.
4. If `raw_multiscale` misses objects, improve crop proposal.
5. If `raw_multiscale` finds objects but `multiscale` loses them, improve scoring
   and merging.
6. Only test detector training after the final multiscale masks look cleaner than
   the normal baseline on a small visual sample.

## Expected Outcome By Strategy

| Strategy                    | Expected output                                        |
| --------------------------- | ------------------------------------------------------ |
| Normal                      | Few masks, stable, misses small objects                |
| Legacy grid                 | Many masks, high recall, noisy                         |
| Hybrid heatmap small preset | Fewer masks, more targeted small-object additions      |
| Hybrid balanced preset      | More forgiving than `small`, possibly noisier        |
| MOST-lite v2 soft           | Fewer object-like crop proposals, experimental quality |
| Combined                    | Best visual coverage, risky training target            |

## Current Project Position

The best next scientific comparison is not "which output has the most masks?" It
is:

```text
Does multiscale add visually good small-object masks that normal MaskCut misses,
without adding too many fragments or duplicate overlaps?
```

Right now, the most defensible main candidate is the hybrid heatmap strategy with
`--ms-preset small`, because it is targeted and already has debug tooling. The
MOST-lite comparison should use v2 soft:

```text
--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45
```

Use v2 soft rather than v1 or v2 strict because v1 is too permissive/noisy, while
v2 strict can drop useful masks. V2 soft keeps the cleanup behavior and relaxes
only the CRF agreement threshold.
