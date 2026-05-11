#!/usr/bin/env python
"""
Visualize the same sampled image IDs across multiple hybrid ablation outputs.

Usage:
    python tools/visualize_hybrid_ablations.py \
        --ablation-root data/tiny-imagenet-5/annotations/hybrid_ablations \
        --image-root data/tiny-imagenet-5/train_flat \
        --output-root experiments/visualizations/hybrid_ablations \
        --variants baseline_100 topk8_100 tightcrop_100 \
        --num-samples 12
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from pycocotools.coco import COCO

from visualize_pseudo_masks import visualize_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-root", required=True, help="Root folder containing one subfolder per ablation variant")
    p.add_argument("--image-root", required=True, help="Root directory of images")
    p.add_argument("--output-root", required=True, help="Where to save per-variant visualization folders")
    p.add_argument("--variants", nargs="+", required=True, help="Variant folder names to compare")
    p.add_argument("--num-samples", type=int, default=12, help="Number of shared images to visualize")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--sample-from-variant",
        default=None,
        help="Variant used to choose the shared sampled image IDs (defaults to first variant)",
    )
    return p.parse_args()


def resolve_primary_json(variant_dir: Path) -> Path:
    candidates = []
    for path in sorted(variant_dir.glob("*.json")):
        name = path.name
        if name.endswith("_normal.json"):
            continue
        if name.endswith("_raw_multiscale.json"):
            continue
        if name.endswith("_multiscale.json"):
            continue
        if name.endswith("_combined.json"):
            continue
        if name.endswith("_candidate_debug.json"):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not find primary JSON in {variant_dir}")
    return candidates[0]


def main():
    args = parse_args()
    random.seed(args.seed)

    ablation_root = Path(os.path.expanduser(args.ablation_root))
    output_root = Path(os.path.expanduser(args.output_root))
    image_root = os.path.expanduser(args.image_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_variant = args.sample_from_variant or args.variants[0]
    sample_json = resolve_primary_json(ablation_root / sample_variant)
    sample_coco = COCO(str(sample_json))
    sample_ids = list({ann["image_id"] for ann in sample_coco.dataset.get("annotations", [])})
    sample_count = min(args.num_samples, len(sample_ids))
    sampled_ids = random.sample(sample_ids, sample_count)

    print(f"Sampling {sample_count} shared image IDs from variant '{sample_variant}'")

    for variant in args.variants:
        variant_dir = ablation_root / variant
        variant_json = resolve_primary_json(variant_dir)
        variant_out = output_root / variant
        variant_out.mkdir(parents=True, exist_ok=True)

        coco = COCO(str(variant_json))
        visualized = 0
        for image_id in sampled_ids:
            if image_id not in coco.imgs:
                continue
            img_info = coco.imgs[image_id]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
            ok = visualize_image(image_id, img_info, anns, image_root, str(variant_out))
            if ok:
                visualized += 1

        print(f"{variant}: visualized {visualized} images -> {variant_out}")


if __name__ == "__main__":
    main()
