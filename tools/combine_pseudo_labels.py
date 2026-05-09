#!/usr/bin/env python3
"""Combine baseline and multiscale COCO pseudo labels with simple dedup rules."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from pycocotools import mask as mask_utils


def decode_mask(segmentation):
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("utf-8")
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return decoded.astype(bool)
    raise TypeError(
        f"Unsupported segmentation type: {type(segmentation)!r}. "
        "Expected COCO RLE dict pseudo labels."
    )


def ann_area(ann):
    area = ann.get("area")
    if area is not None:
        return float(area)
    return float(mask_utils.area(ann_to_rle(ann)))


def ann_to_rle(ann):
    rle = dict(ann["segmentation"])
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    return rle


def iou_and_inside(mask_a, mask_b):
    inter = float((mask_a & mask_b).sum())
    if inter <= 0.0:
        return 0.0, 0.0, 0.0
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    inside_a = inter / area_a if area_a > 0 else 0.0
    inside_b = inter / area_b if area_b > 0 else 0.0
    return iou, inside_a, inside_b


def build_mask_cache(annotations):
    cache = {}
    for ann in annotations:
        cache[ann["id"]] = decode_mask(ann["segmentation"])
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--multiscale-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--inside-thresh", type=float, default=0.7)
    parser.add_argument("--sort-multiscale-by-area", action="store_true")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    multiscale_path = Path(args.multiscale_json)
    output_path = Path(args.output_json)

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)
    with multiscale_path.open("r", encoding="utf-8") as f:
        multiscale = json.load(f)

    combined = {
        "info": deepcopy(baseline.get("info", {})),
        "licenses": deepcopy(baseline.get("licenses", [])),
        "images": deepcopy(baseline["images"]),
        "categories": deepcopy(baseline["categories"]),
        "annotations": [],
    }

    baseline_anns = deepcopy(baseline["annotations"])
    multiscale_anns = deepcopy(multiscale["annotations"])

    baseline_masks = build_mask_cache(baseline_anns)
    multiscale_masks = build_mask_cache(multiscale_anns)

    baseline_by_image = defaultdict(list)
    for ann in baseline_anns:
        baseline_by_image[ann["image_id"]].append(ann)

    if args.sort_multiscale_by_area:
        multiscale_anns.sort(key=ann_area, reverse=True)

    added_by_image = defaultdict(list)
    kept_multiscale = []
    skipped = 0

    for ann in multiscale_anns:
        candidate_mask = multiscale_masks[ann["id"]]
        image_id = ann["image_id"]
        duplicate = False

        for kept in baseline_by_image[image_id]:
            iou, inside_candidate, inside_kept = iou_and_inside(candidate_mask, baseline_masks[kept["id"]])
            if iou >= args.iou_thresh or inside_candidate >= args.inside_thresh or inside_kept >= args.inside_thresh:
                duplicate = True
                break

        if duplicate:
            skipped += 1
            continue

        for kept in added_by_image[image_id]:
            iou, inside_candidate, inside_kept = iou_and_inside(candidate_mask, multiscale_masks[kept["id"]])
            if iou >= args.iou_thresh or inside_candidate >= args.inside_thresh or inside_kept >= args.inside_thresh:
                duplicate = True
                break

        if duplicate:
            skipped += 1
            continue

        kept_multiscale.append(ann)
        added_by_image[image_id].append(ann)

    next_ann_id = 1
    for ann in baseline_anns + kept_multiscale:
        ann["id"] = next_ann_id
        next_ann_id += 1
        combined["annotations"].append(ann)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f)

    print(f"images: {len(combined['images'])}")
    print(f"baseline anns: {len(baseline_anns)}")
    print(f"multiscale anns: {len(multiscale_anns)}")
    print(f"added multiscale anns: {len(kept_multiscale)}")
    print(f"skipped duplicate/inside: {skipped}")
    print(f"combined anns: {len(combined['annotations'])}")


if __name__ == "__main__":
    main()
