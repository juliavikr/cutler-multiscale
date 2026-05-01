#!/usr/bin/env python
"""
Visualize pseudo-label masks from a COCO-format JSON.

Picks N random images, overlays predicted masks and bounding boxes, and saves
one PNG per image to the output directory.

Usage:
    python tools/visualize_pseudo_masks.py \
        --json ~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json \
        --image-root ~/data/tiny-imagenet-10classes/train \
        --output-dir experiments/visualizations/baseline \
        --num-samples 20
"""

import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True, help="Path to COCO-format pseudo-label JSON")
    p.add_argument("--image-root", required=True, help="Root directory of images (class_name/image.JPEG)")
    p.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    p.add_argument("--num-samples", type=int, default=20, help="Number of images to visualize (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def find_image_path(image_root, file_name):
    """Resolve image path, handling TinyImageNet's class/images/file.JPEG layout."""
    direct = os.path.join(image_root, file_name)
    if os.path.exists(direct):
        return direct
    # TinyImageNet stores images one level deeper: class/images/filename
    parts = file_name.replace("\\", "/").split("/")
    if len(parts) == 2:
        candidate = os.path.join(image_root, parts[0], "images", parts[1])
        if os.path.exists(candidate):
            return candidate
    # Last resort: search by basename
    basename = os.path.basename(file_name)
    for root, _, files in os.walk(image_root):
        if basename in files:
            return os.path.join(root, basename)
    return None


def decode_mask(ann, height, width):
    """Return a binary (H, W) numpy array from a COCO annotation's segmentation."""
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        # Compressed or uncompressed RLE
        rle = seg
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(bool)
    elif isinstance(seg, list) and len(seg) > 0:
        # Polygon(s)
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(bool)
    return np.zeros((height, width), dtype=bool)


def visualize_image(image_id, img_info, anns, image_root, output_dir):
    img_path = find_image_path(image_root, img_info["file_name"])
    if img_path is None:
        print(f"  WARNING: image not found for id={image_id} ({img_info['file_name']}), skipping")
        return False

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    n_masks = len(anns)

    # Build a single RGBA overlay for all masks
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    cmap = plt.cm.get_cmap("tab20", max(n_masks, 1))
    colors = [cmap(i % 20) for i in range(n_masks)]

    for ann, color in zip(anns, colors):
        binary = decode_mask(ann, h, w)
        overlay[binary, :3] = color[:3]
        overlay[binary, 3] = 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(overlay, interpolation="none")

    # Bounding boxes
    for ann, color in zip(anns, colors):
        x, y, bw, bh = ann["bbox"]
        rect = mpatches.Rectangle(
            (x, y), bw, bh,
            linewidth=1.5, edgecolor=color[:3], facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"image_id={image_id}  |  {n_masks} mask{'s' if n_masks != 1 else ''}", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    out_path = os.path.join(output_dir, f"{image_id}_{n_masks}masks.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    random.seed(args.seed)

    image_root = os.path.expanduser(args.image_root)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading JSON: {os.path.expanduser(args.json)}")
    coco = COCO(os.path.expanduser(args.json))

    # Only sample images that have at least one annotation
    img_ids_with_anns = list({ann["image_id"] for ann in coco.dataset.get("annotations", [])})
    if not img_ids_with_anns:
        print("ERROR: no annotations found in JSON.", file=sys.stderr)
        sys.exit(1)

    n = min(args.num_samples, len(img_ids_with_anns))
    sampled_ids = random.sample(img_ids_with_anns, n)
    print(f"Sampling {n} of {len(img_ids_with_anns)} annotated images (seed={args.seed})")

    visualized = 0
    mask_counts = []

    for image_id in sampled_ids:
        img_info = coco.imgs[image_id]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        ok = visualize_image(image_id, img_info, anns, image_root, output_dir)
        if ok:
            visualized += 1
            mask_counts.append(len(anns))

    # Summary
    print(f"\n--- Summary ---")
    print(f"Visualized: {visualized} images → {output_dir}")
    dist = Counter(mask_counts)
    for count in sorted(dist):
        label = "mask" if count == 1 else "masks"
        print(f"  Images with {count} {label}: {dist[count]}")
    if mask_counts:
        print(f"  Mean masks/image: {sum(mask_counts)/len(mask_counts):.2f}")


if __name__ == "__main__":
    main()
