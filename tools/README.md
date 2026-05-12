# Tools

Utility scripts for pseudo-label merging, visualization, dataset registration, training orchestration, and result analysis. Scripts are listed in order of importance to the final project pipeline.

---

## Active scripts

### `run_latest_pipeline.py`

Reproducible end-to-end runner for the final 5-class study. Downloads required
datasets and DINO weights, prepares the locked TinyImageNet 5-class subset,
generates baseline and refined hybrid pseudo-labels, merges them, trains one or
more detector variants, and evaluates on class-agnostic COCO. This is the
recommended reproduction path.

```bash
# Reproduce the final combined result only
python tools/run_latest_pipeline.py \
    --data-root ~/data \
    --run-root ~/cutler-multiscale/experiments/repro_runs \
    --run-name latest_hybrid_pipeline \
    --variants combined

# Reproduce the full baseline / hybrid / combined comparison
python tools/run_latest_pipeline.py \
    --data-root ~/data \
    --run-root ~/cutler-multiscale/experiments/repro_runs \
    --run-name latest_hybrid_comparison \
    --variants baseline,hybrid,combined
```

Outputs written to `experiments/repro_runs/<run-name>/`:

| File | Contents |
|------|----------|
| `manifest.json` | exact run configuration and artifact paths |
| `summary.json` | machine-readable pseudo-label and eval summary |
| `summary.md` | human-readable result summary |
| `summary.csv` | compact results table |
| `logs/` | per-step logs |
| `pseudo_labels/` | run-specific JSON snapshots used for training |
| `training/` | detector checkpoints |
| `eval/` | COCO evaluation outputs |

---

### `combine_pseudo_labels.py`

Merges baseline single-scale and refined hybrid crop pseudo-label JSONs into a
single COCO-format JSON. Deduplicates by IoU and containment so that crop masks
that substantially overlap an existing baseline mask are dropped.

```bash
python tools/combine_pseudo_labels.py \
    --baseline-json ~/data/tiny-imagenet-5/annotations/v1_baseline_pseudo.json \
    --multiscale-json ~/data/tiny-imagenet-5/annotations/v1_multiscale_pseudo.json \
    --output-json ~/data/tiny-imagenet-5/annotations/v1_combined_pseudo.json \
    --iou-thresh 0.5 \
    --inside-thresh 0.7
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--baseline-json` | required | Path to baseline pseudo-label JSON |
| `--multiscale-json` | required | Path to hybrid crop pseudo-label JSON |
| `--output-json` | required | Output path for the merged JSON |
| `--iou-thresh` | `0.5` | IoU threshold above which a crop mask is treated as a duplicate |
| `--inside-thresh` | `0.7` | Containment fraction above which a crop mask is treated as a duplicate |
| `--sort-multiscale-by-area` | off | Process crop masks largest-first before dedup |

---

### `train_wrapper_dynamic.py`

Registers a single COCO-format pseudo-label dataset from explicit `--json-path`
and `--image-root` arguments, then delegates to `CutLER/cutler/train_net.py`.
Used by `run_latest_pipeline.py` to avoid relying on mutable shared annotation
filenames. This is the correct training entry point for the final pipeline.

```bash
cd ~/cutler-multiscale/CutLER/cutler
python ~/cutler-multiscale/tools/train_wrapper_dynamic.py \
    --dataset-name my_pseudo_labels \
    --json-path ~/data/tiny-imagenet-5/annotations/v1_combined_pseudo.json \
    --image-root ~/data/tiny-imagenet-5/train \
    --num-gpus 1 \
    --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
    DATASETS.TRAIN '("my_pseudo_labels",)'
```

---

### `visualize_pseudo_masks.py`

Overlays COCO-format pseudo-masks on images and saves one PNG per image.

```bash
python tools/visualize_pseudo_masks.py \
    --json ~/data/tiny-imagenet-5/annotations/v1_baseline_pseudo.json \
    --image-root ~/data/tiny-imagenet-5/train \
    --output-dir experiments/visualizations/baseline \
    --num-samples 20 \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--json` | required | Path to COCO-format pseudo-label JSON |
| `--image-root` | required | Root directory of the image dataset |
| `--output-dir` | required | Where to save PNG overlays |
| `--num-samples` | `20` | Number of random images to visualize |
| `--seed` | `42` | Random seed for reproducibility |

On the cluster, use `sbatch slurm/run_visualize.sh` instead.

---

### `visualize_hybrid_ablations.py`

Visualizes the same sampled image IDs across multiple hybrid ablation output
folders so that variants can be compared side by side.

```bash
python tools/visualize_hybrid_ablations.py \
    --ablation-root ~/data/tiny-imagenet-5/annotations/hybrid_ablations \
    --image-root ~/data/tiny-imagenet-5/train_flat \
    --output-root experiments/visualizations/hybrid_ablations \
    --variants baseline_100 topk8_100 tightcrop_100 \
    --num-samples 12
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ablation-root` | required | Root folder with one subfolder per ablation variant |
| `--image-root` | required | Root directory of images |
| `--output-root` | required | Where to save per-variant visualization folders |
| `--variants` | required | Variant folder names to compare |
| `--num-samples` | `12` | Number of shared images to visualize |
| `--seed` | `42` | Random seed |
| `--sample-from-variant` | first variant | Which variant's image IDs to use as the shared sample |

---

### `make_cls_agnostic_coco.py`

Converts a standard COCO annotation JSON to class-agnostic format (all category
IDs replaced with a single "object" category). Required before evaluating a
class-agnostic detector on COCO val2017.

```bash
python tools/make_cls_agnostic_coco.py \
    --input ~/data/coco/annotations/instances_val2017.json \
    --output ~/data/coco/annotations/instances_val2017_cls_agnostic.json
```

---

### `compare_pseudo_label_stats.py`

Prints descriptive statistics (mask count, area distribution, masks per image)
for two COCO-format pseudo-label JSONs side by side. Useful for sanity-checking
a merged or filtered JSON before running detector training.

```bash
python tools/compare_pseudo_label_stats.py \
    --a ~/data/tiny-imagenet-5/annotations/v1_baseline_pseudo.json \
    --b ~/data/tiny-imagenet-5/annotations/v1_combined_pseudo.json
```

---

## Experiments analysis scripts

These live in `experiments/` but are documented here for completeness.

### `experiments/plot_detector_results.py`

Generates the main 5-class detector comparison PNG from
`results/detector_results_5class.csv`.

```bash
python experiments/plot_detector_results.py \
    --csv results/detector_results_5class.csv \
    --output results/figures/detector_results_5class.png
```

### `experiments/plot_hybrid_ablation_results.py`

Generates the ablation summary PNG panels from
`results/hybrid_ablation_100_summary.csv`.

```bash
python experiments/plot_hybrid_ablation_results.py \
    --csv results/hybrid_ablation_100_summary.csv \
    --output-dir experiments/visualizations/hybrid_ablation_summary
```

### `experiments/plot_training_losses.py`

Plots Detectron2 training loss curves from `metrics.json` files written during
detector training.

```bash
python experiments/plot_training_losses.py \
    --metrics-root ~/cutler-multiscale/experiments/repro_runs \
    --output-dir results/figures/training_losses
```

### `experiments/rank_small_ap.py`

Scans training output directories for `eval.log` files, extracts COCO AP
metrics, and prints runs ranked by APs. Useful for identifying which training
run produced the best small-object AP.

```bash
python experiments/rank_small_ap.py \
    --root ~/cutler-multiscale/experiments \
    --csv-out results/ap_ranking.csv
```

---

## Legacy scripts

These are kept for historical reference. They are not part of the final pipeline.

| Script | Superseded by |
|--------|---------------|
| `train_wrapper.py` | `train_wrapper_dynamic.py` (fixed annotation filenames) |
| `train_wrapper_luiz.py` | `train_wrapper_dynamic.py` |
| `register_tinyimagenet_pseudo.py` | `train_wrapper_dynamic.py` (dynamic registration) |
| `register_tinyimagenet_pseudo_luiz.py` | `train_wrapper_dynamic.py` |
