# Elevator Pitch

*30-second version — use for the opening of the talk or when someone asks "what's your project about?"*

---

Modern object detectors need thousands of labeled bounding boxes to train, which is
expensive and doesn't generalize across domains. CutLER showed that DINO's self-attention
maps can generate useful pseudo-labels without any human annotation — but the method has
a blind spot: it processes images at a single fixed scale, so small objects get lost.
We extended the pseudo-label generation step with a multi-scale cropping strategy: we crop
the image at multiple zoom levels, run the same algorithm on each crop, and merge the
proposals back. The goal is to recover small-object masks that the single-scale method
misses, and see whether those better pseudo-labels translate into a measurable improvement
on COCO's small-object AP metric.

---

*Longer version (60 sec) — for a project demo or poster session:*

Object detection normally requires annotating thousands of images with bounding boxes —
a slow, expensive process that has to be repeated for every new domain.
CutLER is a fully unsupervised approach: it uses DINO's self-attention maps as a signal
for where objects are, then applies Normalized Cuts to produce binary pseudo-masks.
A detector trained on these noisy pseudo-labels still achieves surprisingly competitive
results on COCO, with zero human annotation.

The catch is that single-scale processing is biased toward large, dominant objects.
Small objects occupy just a few attention patches and rarely survive the graph partitioning.

Our contribution is a multi-scale extension of the MaskCut stage: we slide crop windows
at 0.75× and 0.5× zoom over the image, run MaskCut on each crop, project masks back to
original coordinates, and merge them with a graph-based IoU filter.
This costs about 10× more compute at the pseudo-label stage, but the detector training
and inference are unchanged.

We evaluate on a controlled 10-class TinyImageNet subset — identical data, identical
training, the only variable is the pseudo-label generation strategy — and measure whether
multi-scale masks improve COCO APs without degrading overall AP.
