"""
Entry point for CutLER training with TinyImageNet pseudo-label datasets pre-registered.
Run from the CutLER/cutler/ directory, same as you would run train_net.py directly.

Example (from CutLER/cutler/):
    python ~/cutler-multiscale/tools/train_wrapper.py --num-gpus 1 \
        --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
        DATASETS.TRAIN '("tinyimagenet_baseline_pseudo",)' ...
"""
import os
import sys
import runpy

# Register our TinyImageNet pseudo-label datasets before Detectron2 initializes
_tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _tools_dir)
import register_tinyimagenet_pseudo  # noqa: F401

# Run the upstream train_net.py as __main__ (runpy prepends its dir to sys.path,
# so CutLER's local imports like `from config import ...` continue to work).
_train_net = os.path.abspath(os.path.join(_tools_dir, "../CutLER/cutler/train_net.py"))
_cutler_dir = os.path.dirname(_train_net)
if _cutler_dir not in sys.path:
    sys.path.insert(0, _cutler_dir)
runpy.run_path(_train_net, run_name="__main__")
