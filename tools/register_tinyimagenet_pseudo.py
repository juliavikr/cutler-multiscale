import os
from detectron2.data.datasets import register_coco_instances

_ANNO_DIR = os.path.expanduser(
    "~/data/tiny-imagenet-5classes/annotations"
)
_IMAGE_ROOT = os.path.expanduser(
    "~/data/tiny-imagenet-5classes/train"
)

register_coco_instances(
    "tinyimagenet_baseline_pseudo",
    {},
    os.path.join(_ANNO_DIR, "tinyimagenet_5c_baseline_pseudo.json"),
    _IMAGE_ROOT,
)

register_coco_instances(
    "tinyimagenet_hybrid_pseudo",
    {},
    os.path.join(_ANNO_DIR, "tinyimagenet_5c_hybrid_pseudo.json"),
    _IMAGE_ROOT,
)
