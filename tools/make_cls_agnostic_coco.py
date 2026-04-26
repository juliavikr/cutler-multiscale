#!/usr/bin/env python3
"""
Convert a standard COCO instances JSON to a class-agnostic version.

All annotation category_ids are remapped to 1 and the categories list is
replaced with a single 'object' category. Required because CutLER's
Cascade Mask R-CNN head is class-agnostic (NUM_CLASSES=1), so evaluating
against the 80-class COCO ground truth produces zero matches.

Usage:
    python tools/make_cls_agnostic_coco.py \
        --input  ~/data/coco/annotations/instances_val2017.json \
        --output ~/data/coco/annotations/instances_val2017_cls_agnostic.json
"""

import argparse
import json
from pathlib import Path


def make_cls_agnostic(input_path: str, output_path: str) -> None:
    print(f"Reading {input_path} ...")
    with open(input_path) as f:
        data = json.load(f)

    n_before = len({a["category_id"] for a in data["annotations"]})

    for ann in data["annotations"]:
        ann["category_id"] = 1

    data["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(
        f"Done. Collapsed {n_before} categories → 1. "
        f"{len(data['annotations'])} annotations, {len(data['images'])} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Path to input COCO instances JSON")
    parser.add_argument("--output", required=True, help="Path to write class-agnostic JSON")
    args = parser.parse_args()
    make_cls_agnostic(args.input, args.output)
