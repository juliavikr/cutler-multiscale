import os
from detectron2.data.datasets import register_coco_instances

_ANNO_DIR = os.path.expanduser(
    "~/data/tiny-imagenet-10classes/annotations"
)
_IMAGE_ROOT = os.path.expanduser(
    "~/data/tiny-imagenet-10classes/train"
)

register_coco_instances(
    "tinyimagenet_baseline_pseudo",
    {},
    os.path.join(_ANNO_DIR, "tinyimagenet_10c_baseline_pseudo.json"),
    _IMAGE_ROOT,
)

register_coco_instances(
    "tinyimagenet_multiscale_pseudo",
    {},
    os.path.join(_ANNO_DIR, "tinyimagenet_10c_multiscale_pseudo.json"),
    _IMAGE_ROOT,
)
