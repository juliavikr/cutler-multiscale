# cutler-multiscale

Improving small-object recovery in CutLER by adding **heatmap-guided multi-scale rescue masks** to the original single-scale MaskCut pseudo-labels.

Important boundary: all project-specific work lives outside `CutLER/`. The upstream `CutLER/` folder is treated as read-only vendor code.

## Final project story

The repository is organized around one main idea:

1. run the original **single-scale MaskCut** baseline
2. generate **additional crop-level masks** with our refined hybrid method
3. **merge** those extra masks with the original MaskCut masks
4. train the detector on the merged pseudo-label set

The refined hybrid masks were **not** successful as a standalone pseudo-label source. The best result comes from using them as a **supplement** to the original MaskCut masks.

## Main result

Class-agnostic COCO evaluation, main 5-class study:

| Training pseudo-labels | BBOX AP | SEGM AP | Notes |
|---|---:|---:|---|
| Baseline single-scale | 2.1447 | 0.4792 | original MaskCut masks only |
| New hybrid-only | 0.3026 | 0.1946 | refined hybrid masks only |
| New combined hybrid-best | 2.2557 | 1.0814 | baseline + refined hybrid masks |

Takeaway: the refined hybrid method helps **after merging with baseline**, not by replacing it.

## Reproducing the results

### Recommended path

The most reproducible path is the new Python pipeline:

- [tools/run_latest_pipeline.py](C:/Users/Luiz%20Venosa/Documents/Bocconi/Master/2nd%20Semester/Computer%20VIsion/project/cutler-multiscale/tools/run_latest_pipeline.py)

It does the full final workflow:

1. download TinyImageNet, COCO val2017, annotations, and DINO weights
2. prepare the locked 5-class TinyImageNet subset
3. generate baseline single-scale MaskCut pseudo-labels
4. generate the latest refined hybrid multi-scale pseudo-labels
5. merge baseline + refined hybrid crop masks
6. train the detector
7. evaluate on class-agnostic COCO
8. save a manifest and summary tables for provenance

### Environment

Activate the same environment used for the main experiments:

```bash
conda activate cutler
```

The pipeline expects:

- Detectron2 / CutLER dependencies already installed
- one visible GPU
- permission to download datasets and DINO weights into `DATA_ROOT`

### Final result only

This reproduces the **main grading-facing result**: baseline + refined hybrid merged.

```bash
cd /home/3191856/cv_project/cutler-multiscale

python tools/run_latest_pipeline.py \
  --data-root /home/3191856/data \
  --run-root /home/3191856/cv_project/cutler-multiscale/experiments/repro_runs \
  --run-name latest_hybrid_pipeline \
  --variants combined
```

### Full comparison

This reruns the three most important 5-class detector variants:

- baseline single-scale
- refined hybrid-only
- baseline + refined hybrid combined

```bash
cd /home/3191856/cv_project/cutler-multiscale

python tools/run_latest_pipeline.py \
  --data-root /home/3191856/data \
  --run-root /home/3191856/cv_project/cutler-multiscale/experiments/repro_runs \
  --run-name latest_hybrid_comparison \
  --variants baseline,hybrid,combined
```

### What gets written

For a run named `<run-name>`, the pipeline writes to:

```text
experiments/repro_runs/<run-name>/
```

Important outputs:

- `manifest.json` - exact run configuration and artifact paths
- `summary.json` - machine-readable pseudo-label and eval summary
- `summary.md` - human-readable result summary
- `summary.csv` - compact results table
- `logs/` - per-step logs
- `pseudo_labels/` - run-specific JSON snapshots used for training
- `training/` - detector checkpoints and training outputs
- `eval/` - COCO evaluation outputs

### Why this is the preferred path

This runner is more reproducible than the older SLURM aliases because it:

- uses **run-specific pseudo-label snapshots** instead of mutable shared `v1_*.json` filenames
- fixes the 5-class subset explicitly in code
- writes a provenance manifest for the generated JSONs, checkpoints, and summaries
- keeps the final project story aligned with the actual successful pipeline

## Manual pipeline (advanced)

If you want to run individual stages manually, the core project-owned pieces are:

### 1. Baseline pseudo-label generation

- implementation: `multiscale/multiscale_maskcut.py`
- historical cluster runner: `slurm/run_singlescale_maskcut.sh`

### 2. Refined hybrid pseudo-label generation

- implementation: `multiscale/multiscale_maskcut.py`
- historical cluster runner: `slurm/run_multiscale_maskcut.sh`

### 3. Merge baseline + refined hybrid masks

- helper: `tools/combine_pseudo_labels.py`

### 4. Detector training

- dynamic wrapper used by the reproducible pipeline: `tools/train_wrapper_dynamic.py`
- older fixed-name cluster runner: `slurm/run_training_luiz.sh`

### 5. COCO evaluation

- direct CutLER eval call inside the reproducible pipeline
- older cluster runner: `slurm/run_eval.sh`

## Repository map

| Path | Purpose |
|---|---|
| `README.md` | top-level reproduction and final contribution summary |
| `PROJECT_OVERVIEW.md` | plain-English explanation of the method |
| `PROJECT_NOTES.md` | experiment ledger, final results, ablations, runtime notes |
| `multiscale/` | our custom multi-scale pseudo-label generation code and method docs |
| `slurm/` | cluster runners for generation, training, eval, and ablations |
| `tools/` | utility scripts for merging, plotting, and visualization |
| `results/` | committed result summaries |
| `experiments/visualizations/` | generated charts and figure assets |
| `CutLER/` | upstream vendor code, not modified here |

## Key project-owned files

### Core method

- `multiscale/multiscale_maskcut.py` - current main implementation
- `multiscale/MULTISCALE_MASKCUT.md` - code guide and CLI reference
- `multiscale/STRATEGY_COMPARISON.md` - comparison of proposal strategies
- `multiscale/EVALUATION_PROCESS.md` - evaluation methodology

### Utilities

- `tools/combine_pseudo_labels.py` - merge baseline and refined hybrid masks
- `experiments/plot_hybrid_ablation_results.py` - generate ablation charts
- `experiments/plot_training_losses.py` - generate detector loss charts

## Historical snapshots

These remain in the repo only as references to earlier design stages and should not be treated as the primary path:

- `multiscale/multiscale_maskcut_hybrid.py`
- `multiscale/multiscale_maskcut_legacy.py`
- `multiscale/final_multiscale.py`
- `slurm/run_hybrid_maskcut_tinyimagenet.sh`
- `slurm/run_multiscale_maskcut_tinyimagenet.sh`

## Reproduction notes

All heavy runs were executed on the Bocconi HPC cluster with a single A100 GPU. The final 5-class study used the locked class subset in [PROJECT_NOTES.md](C:/Users/Luiz%20Venosa/Documents/Bocconi/Master/2nd%20Semester/Computer%20VIsion/project/cutler-multiscale/PROJECT_NOTES.md) and the refined hybrid defaults described there.

The project is intentionally organized so that:

- `CutLER/` remains untouched vendor code
- all project-specific logic lives in `multiscale/`, `tools/`, `slurm/`, and the project docs
- the final successful result is the **combined** pipeline, not hybrid-only

## Where to look next

- final numbers and ablations: `PROJECT_NOTES.md`
- plain-English method story: `PROJECT_OVERVIEW.md`
- cluster commands: `slurm/README.md`
- result summaries: `results/README.md`
