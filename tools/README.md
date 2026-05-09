# Tools

Utility scripts for visualization, dataset registration, training orchestration, and result analysis.

---

## `visualize_pseudo_masks.py`

Overlays COCO-format pseudo-masks on images and saves one PNG per image.

```bash
python tools/visualize_pseudo_masks.py \
    --json ~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json \
    --image-root ~/data/tiny-imagenet-10classes/train \
    --output-dir experiments/visualizations/baseline \
    --num-samples 20 \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--json` | required | Path to COCO-format pseudo-label JSON |
| `--image-root` | required | Root directory of the image dataset |
| `--output-dir` | required | Where to save PNG overlays |
| `--num-samples` | 20 | Number of random images to visualize |
| `--seed` | 42 | Random seed for reproducibility |

Output: `<output-dir>/<image_id>.png` files. On the cluster, use `sbatch slurm/run_visualize.sh` instead.

---

## `register_tinyimagenet_pseudo.py` (legacy)

Registers TinyImageNet pseudo-label datasets with Detectron2's dataset catalog. Not called directly — imported automatically by the older `train_wrapper.py` training path. Registers two datasets:
- `tinyimagenet_baseline_pseudo`
- `tinyimagenet_multiscale_pseudo`

Paths are resolved from `~/data/tiny-imagenet-10classes/`.

---

## `train_wrapper.py` (legacy)

Wraps `CutLER/cutler/train_net.py` by pre-registering TinyImageNet datasets before Detectron2 initialises. Called automatically by `slurm/run_training.sh`. Run from `CutLER/cutler/` in the same way as `train_net.py` directly:

```bash
cd ~/cutler-multiscale/CutLER/cutler
python ~/cutler-multiscale/tools/train_wrapper.py \
    --num-gpus 1 \
    --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
    DATASETS.TRAIN '("tinyimagenet_baseline_pseudo",)' \
    ...
```

---

## `make_cls_agnostic_coco.py`

Converts a standard COCO annotation JSON to class-agnostic format (all category IDs replaced with a single "object" category). Required before evaluating a class-agnostic detector on COCO val2017.

```bash
python tools/make_cls_agnostic_coco.py \
    --input ~/data/coco/annotations/instances_val2017.json \
    --output ~/data/coco/annotations/instances_val2017_cls_agnostic.json
```

---

## `run_latest_pipeline.py`

Reproducible end-to-end runner for the final 5-class study. It downloads the
required datasets and DINO weights, prepares the locked TinyImageNet 5-class
subset, generates baseline and refined hybrid pseudo-labels, merges them,
trains one or more detector variants, and evaluates them on class-agnostic
COCO.

Example: run the final combined pipeline only

```bash
python tools/run_latest_pipeline.py \
    --data-root ~/data \
    --run-name latest_hybrid_pipeline \
    --variants combined
```

Example: reproduce the baseline / hybrid / combined comparison

```bash
python tools/run_latest_pipeline.py \
    --data-root ~/data \
    --run-name latest_hybrid_comparison \
    --variants baseline,hybrid,combined
```

Outputs:

- `experiments/repro_runs/<run-name>/manifest.json`
- `experiments/repro_runs/<run-name>/summary.md`
- `experiments/repro_runs/<run-name>/summary.csv`
- per-step logs under `experiments/repro_runs/<run-name>/logs/`

---

## `train_wrapper_dynamic.py`

Registers a single COCO-format pseudo-label dataset from explicit `--json-path`
and `--image-root` arguments, then delegates to `CutLER/cutler/train_net.py`.
This is used by `run_latest_pipeline.py` to avoid relying on mutable shared
annotation filenames.

---

## `experiments/rank_small_ap.py`

Scans training output directories for `eval.log` files, extracts COCO AP metrics, and prints runs ranked by APs (small-object AP). Optionally exports a CSV.

```bash
python experiments/rank_small_ap.py \
    --root ~/cutler-multiscale/experiments \
    --csv-out results/ap_ranking.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | `output` | Directory to scan for `eval.log` files |
| `--csv-out` | — | Optional path to save a CSV with all metrics |
