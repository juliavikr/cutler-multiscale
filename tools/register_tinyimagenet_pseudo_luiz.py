import os
from detectron2.data.datasets import register_coco_instances


_DATA_ROOT = os.path.expanduser(
    os.environ.get("DATA_ROOT", os.path.join(os.getcwd(), "../../data"))
)
_ANNO_DIR_5C = os.path.join(_DATA_ROOT, "tiny-imagenet-5", "annotations")
_IMAGE_ROOT_5C = os.path.join(_DATA_ROOT, "tiny-imagenet-5", "train_flat")

register_coco_instances(
    "tinyimagenet_5c_baseline_pseudo",
    {},
    os.path.join(_ANNO_DIR_5C, "v1_baseline_pseudo.json"),
    _IMAGE_ROOT_5C,
)

register_coco_instances(
    "tinyimagenet_5c_multiscale_pseudo",
    {},
    os.path.join(_ANNO_DIR_5C, "v1_multiscale_pseudo.json"),
    _IMAGE_ROOT_5C,
)
