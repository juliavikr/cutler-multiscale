# cutler-multiscale
Improving small-object detection in CutLER via Multi-Scale Pseudo-Label Generation

Custom multi-scale pseudo-label generation code lives in [`multiscale/`](./multiscale/), not inside the upstream `CutLER/` submodule.

## Setup & Reproduction

### Prerequisites

- Python 3.9, CUDA 12.1
- Full package list: [`experiments/environment.yml`](./experiments/environment.yml)
- DINO ViT-S/8 pretrained weights (downloaded via `slurm/download_checkpoint.sh`)
- COCO val2017 images and annotations (downloaded via `slurm/download_data.sh`)

All heavy computation runs on the **Bocconi HPC cluster** (A100 80GB). Do not run training or MaskCut locally.

### Environment

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/juliavikr/cutler-multiscale
cd cutler-multiscale

# 2. Create conda environment
conda env create -f experiments/environment.yml
conda activate cutler

# 3. Install Detectron2 (must use miropsota pre-built wheels — source build fails against PyTorch 2.x)
bash slurm/install_detectron2.sh
```

### Data and Weights (run on cluster)

```bash
bash slurm/download_checkpoint.sh   # CutLER cascade checkpoint + pre-generated MaskCut annotations → ~/cutler-multiscale/checkpoints/
bash slurm/download_data.sh         # COCO val2017 → ~/data/coco/
bash slurm/download_tinyimagenet.sh # TinyImageNet 10-class subset → ~/data/tiny-imagenet-10classes/
```

DINO ViT-S/8 weights (`dino_deitsmall8_pretrain.pth`) must be downloaded separately from the [DINO release](https://github.com/facebookresearch/dino) and placed at `~/data/weights/dino_deitsmall8_pretrain.pth`.

### Cluster Paths

| Variable | Value |
|----------|-------|
| `PROJECT_ROOT` | `~/cutler-multiscale` |
| `DATA_ROOT` | `~/data` |

All SLURM scripts use `DATA_ROOT` as an environment variable. Override if your paths differ.

### End-to-End Reproduction

```
1. Clone + env setup (above)
2. Download data and weights (above)
3. Generate pseudo-labels:  sbatch slurm/run_maskcut_baseline.sh
4. Train detector:          PSEUDO_LABEL_NAME=baseline sbatch slurm/run_training.sh
5. Evaluate:                sbatch slurm/run_eval.sh
```

See **How To Run The Three Methods** below for the full multi-scale experiment commands. See [`slurm/README.md`](./slurm/README.md) for a complete script index.

---

## Current Experiment Focus

Focus the report and final experiments on three methods:

| Method | What it tests | Main command/flags |
| --- | --- | --- |
| Baseline MaskCut | Original full-image CutLER pseudo-label generation | no `--multi-crop` |
| Hybrid heatmap | Our main multiscale candidate: DINO heatmap crop proposals | `--multi-crop --ms-preset small` |
| MOST-lite v2 soft | Experimental token-cluster crop proposals, slightly softer than v2 strict | `--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45` |

Use `multiscale` as the training/evaluation output for the two multiscale
methods. Use `normal`, `raw_multiscale`, `combined`, and `candidate_debug` only
for diagnosis and figures unless the experiment explicitly says otherwise.

## How To Run The Three Methods

These commands keep the shared comparison settings fixed:

```text
ViT: small
feature: k
patch size: 8
tau: 0.15
N: 3
fixed_size: 480
```

### 1. Baseline MaskCut

On the cluster:

```bash
sbatch slurm/run_maskcut_baseline.sh
```

This writes:

```text
~/data/tiny-imagenet-10classes/annotations/tinyimagenet_10c_baseline_pseudo.json
```

### 2. Hybrid Heatmap Multiscale

On the cluster:

```bash
DATASET_PATH="${HOME}/data/tiny-imagenet-10classes/train" \
OUT_DIR="${HOME}/data/tiny-imagenet-10classes/annotations" \
DINO_WEIGHTS="${HOME}/data/weights/dino_deitsmall8_pretrain.pth" \
TAU=0.15 \
N_MASKS=3 \
FIXED_SIZE=480 \
NUM_FOLDER_PER_JOB=10 \
MS_PRESET=small \
PRIMARY_OUTPUT=multiscale \
sbatch slurm/run_multiscale_maskcut.sh
```

Equivalent direct flags:

```text
--multi-crop --ms-preset small --primary-output multiscale
```

### 3. MOST-lite v2 Soft

On the cluster:

```bash
DATASET_PATH="${HOME}/data/tiny-imagenet-10classes/train" \
OUT_DIR="${HOME}/data/tiny-imagenet-10classes/annotations" \
DINO_WEIGHTS="${HOME}/data/weights/dino_deitsmall8_pretrain.pth" \
TAU=0.15 \
N_MASKS=3 \
FIXED_SIZE=480 \
NUM_FOLDER_PER_JOB=10 \
MS_PRESET=mostlite \
CRF_IOU_THRESH=0.45 \
PRIMARY_OUTPUT=multiscale \
sbatch slurm/run_multiscale_maskcut.sh
```

Equivalent direct flags:

```text
--multi-crop --ms-preset mostlite --crf-iou-thresh 0.45 --primary-output multiscale
```

Why v2 soft: v1 is too permissive/noisy, while v2 strict is cleaner but may drop
useful masks. v2 soft keeps the v2 cleanup steps and only relaxes CRF agreement
from `0.50` to `0.45`, making it the best experimental MOST-lite candidate to
compare against the hybrid heatmap method.

## Results

Evaluated on COCO val2017, class-agnostic, fully unsupervised (no labels used at any stage).

### Bounding Box Detection

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (paper) | 8.3 | 13.8 | 8.0 | — | — | — |
| CutLER (ours, reproduced) | 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 |
| MS-MaskCut (ours) | — | — | — | — | — | — |

### Instance Segmentation

| Method | AP | AP50 | AP75 | APs | APm | APl |
|--------|----|------|------|-----|-----|-----|
| CutLER (ours, reproduced) | 9.78 | 18.92 | 9.19 | 2.44 | 8.77 | 24.29 |
| MS-MaskCut (ours) | — | — | — | — | — | — |

Baseline uses the pre-trained `cutler_cascade_final.pth` checkpoint released by Facebook Research.
Results match the original CutLER paper, confirming reproducibility.
