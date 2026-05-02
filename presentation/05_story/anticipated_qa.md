# Anticipated Q&A

Likely questions from the audience / examiner, with prepared answers.

---

## Q1: Why didn't you use full ImageNet like the paper?

**Short answer:** Compute constraints. Multi-scale MaskCut is ~10× slower than single-scale.
At 6–7 sec/image on an A100, 1.3M images would take ~100 days. Our SLURM jobs have a
24-hour wall-time limit and we have one GPU.

**Better answer:** The 10-class TinyImageNet subset (500 images) is the *largest* subset
where both baseline and multiscale can complete within that limit. The comparison remains
valid because the two conditions see identical data — the absolute AP numbers are lower
than the paper's, but the *relative* improvement from multiscale is what we're measuring.

---

## Q2: Why these specific scales (1.0, 0.75, 0.5)?

**Short answer:** They cover a 1×–4× range of object sizes, and 0.5× is the coarsest scale
where a 480×480 crop still captures enough context for meaningful attention maps.

**Better answer:** The original MaskCut processes images at fixed 480×480. An object at
0.5× zoom in the original image occupies 240×240 px before resizing — large enough for
DINO's 8×8 patches to produce a useful attention signal. Going to 0.25× would mean
120×120 objects in the crop, which is below the patch resolution floor.
The 0.75× level fills the gap between full and half scale. This is a heuristic choice;
a proper ablation would test 2-scale vs 3-scale — out of scope for this project.

---

## Q3: How does NMS handle masks from different scales — couldn't a large mask from scale 1.0 suppress a small mask from scale 0.5?

**Short answer:** We use a graph-based merge, not greedy NMS. Two masks are connected if
IoU > 0.5 *or* one contains the other (intersection/smaller > 0.7). Small masks from
finer crops only get suppressed if they are *actually* inside a larger mask — which is
the correct behavior for a nested object (e.g., a wheel inside a car).

**Better answer:** The `--small-first` flag prioritizes keeping smaller masks during
connected-component resolution. If a small mask and a large mask overlap significantly,
we favor the small mask rather than the large one — precisely because we're trying to
recover small objects that single-scale would have suppressed anyway.

---

## Q4: Isn't COCO val2017 the wrong benchmark — shouldn't you use a small-object dataset like VisDrone?

**Short answer:** COCO val2017 is the standard benchmark used by CutLER, making our
results directly comparable to the paper. It also has APs/APm/APl breakdowns, which is
exactly what we need to measure small-object improvement.

**Better answer:** You're right that VisDrone or DOTA would be a more direct test of
small-object detection. We chose COCO for comparability with CutLER. Evaluating on
VisDrone would be the natural next step (and is listed as future work).

---

## Q5: Your training set is 500 images. Isn't that too small to draw conclusions?

**Short answer:** For a *relative* comparison between two pseudo-label strategies on the
same data, 500 images is sufficient. We're not claiming state-of-the-art AP; we're
testing whether multi-scale pseudo-labels are better than single-scale ones.

**Better answer:** The experiment is designed as a controlled comparison: identical
training config, identical evaluation, the only variable is the pseudo-label source.
A small training set means both methods have less signal to work with, which if anything
makes it *harder* to see improvements — so a positive result would be conservative.

---

## Q6: What happens if multi-scale generates too many false-positive small masks? Could AP actually go down?

**Short answer:** Yes, this is a real risk. More masks means more potential false positives.
If the detector learns from noisy small-object pseudo-labels, precision on small objects
could drop even if recall goes up, and the AP metric (which averages precision-recall)
might not improve or could worsen.

**Better answer:** This is why we track APs, APm, and APl separately rather than just
overall AP. The 5-class development run showed multi-scale generates 7.5× more annotations
(24k vs 3k), concentrated in small objects (94.6% vs 56.6% small). Whether that signal
is clean enough depends on the quality of the merge step. The graph-based merge with
containment and box-expansion suppression is our attempt to filter the worst duplicates,
but it's a heuristic — the results table will tell us whether it was enough.

---

## Q7: How is this different from standard multi-scale inference at test time?

**Short answer:** Multi-scale *inference* (image pyramid at test time) is applied to a
*trained* detector. Our multi-scale extension is at the *pseudo-label generation* stage,
before any detector exists. We're changing what training data the detector sees, not how
it runs at test time.

**Better answer:** Test-time multi-scale uses a pre-trained model with learned features.
Our approach generates richer pseudo-labels so the model can *learn* small-object features
in the first place. They are complementary — you could combine both. We don't use
test-time multi-scale in our evaluation (to keep it a fair single-variable comparison).

---

## Q8: Why Normalized Cuts and not something simpler, like thresholding the attention map?

**Short answer:** We didn't design MaskCut — it's the upstream CutLER method. Our
contribution is the multi-scale extension, not the core segmentation algorithm.

**Better answer:** Normalized Cuts on the attention affinity matrix tends to produce
coherent, connected masks even when the attention is noisy, because it considers pairwise
patch relationships rather than thresholding each patch independently. Simple thresholding
often produces fragmented, scattered masks. That said, NCut is computationally expensive
(O(n³) for dense graphs) — one reason why running it on many crop windows is costly.
