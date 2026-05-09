"""
Entry point for CutLER training with a dynamically registered COCO dataset.

This keeps the upstream CutLER code untouched while letting our reproducible
pipeline train from run-specific pseudo-label snapshots instead of mutable
shared filenames.
"""
import argparse
import os
import runpy
import sys

from detectron2.data.datasets import register_coco_instances


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-name", required=True, help="Detectron2 dataset name to register")
    parser.add_argument("--json-path", required=True, help="COCO-format pseudo-label JSON")
    parser.add_argument("--image-root", required=True, help="Image root for the pseudo-label JSON")
    args, remaining = parser.parse_known_args()

    register_coco_instances(
        args.dataset_name,
        {},
        os.path.abspath(args.json_path),
        os.path.abspath(args.image_root),
    )

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    train_net = os.path.abspath(os.path.join(tools_dir, "../CutLER/cutler/train_net.py"))
    cutler_dir = os.path.dirname(train_net)
    if cutler_dir not in sys.path:
        sys.path.insert(0, cutler_dir)

    sys.argv = [train_net] + remaining
    runpy.run_path(train_net, run_name="__main__")


if __name__ == "__main__":
    main()
