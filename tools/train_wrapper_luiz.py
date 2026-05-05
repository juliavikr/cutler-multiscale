"""
Entry point for Luiz's 5-class TinyImageNet CutLER training runs.
Run from CutLER/cutler/, same as train_net.py.
"""
import os
import runpy
import sys


_tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _tools_dir)
import register_tinyimagenet_pseudo_luiz  # noqa: F401

_train_net = os.path.abspath(os.path.join(_tools_dir, "../CutLER/cutler/train_net.py"))
_cutler_dir = os.path.dirname(_train_net)
if _cutler_dir not in sys.path:
    sys.path.insert(0, _cutler_dir)
runpy.run_path(_train_net, run_name="__main__")
