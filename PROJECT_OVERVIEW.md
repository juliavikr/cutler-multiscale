# Project Overview: cutler-multiscale

This document explains everything about this project from scratch — no prior knowledge assumed.

---

## 1. What Is This Project?

This is a computer vision course project at Bocconi University. The goal is to **detect objects in images automatically, without ever showing the computer a single hand-labeled example**. We start from an existing system called **CutLER** (published by Facebook Research) that already does this reasonably well, and we extend it to be better at finding **small objects** — things like distant pedestrians, small animals, or vehicles far away that the original system tends to miss.

---

## 2. The Core Problem: What Is Unsupervised Object Detection?

### The usual way (supervised)

Normally, training an object detector requires a huge dataset where humans have manually drawn boxes around every object in thousands of images and labeled what each object is ("car", "dog", "person", etc.). This is called **supervised learning**. The COCO dataset, for example, took tens of thousands of person-hours to annotate.

### The problem with labels

- Labeling is **expensive** (requires human annotators)
- Labels are **slow** to produce
- Labels are **domain-specific** — if you want to detect objects in a new setting (aerial images, medical scans, underwater footage), you need to start over

### The unsupervised alternative

**Unsupervised object detection** means: can a computer figure out where objects are *just by looking at a lot of unlabeled images*? No human boxes, no category names — just raw pixels.

CutLER answers: yes, to a useful degree. It exploits the fact that objects in images have consistent visual patterns (textures, edges, colors) that distinguish them from backgrounds — and a powerful enough visual model can discover these patterns on its own.

---

## 3. The Key Technology: DINO and Vision Transformers

### What is a Vision Transformer (ViT)?

A standard neural network for images processes pixels through many layers of convolutions. A **Vision Transformer** takes a different approach:

1. Chop the image into a grid of small square tiles called **patches** (e.g., 8×8 pixels each)
2. Treat each patch as a "token" — like a word in a sentence
3. Use the **Transformer** architecture (the same kind used in language models like GPT) to let every patch attend to every other patch

This gives the model a global view of the image: a patch in the top-left corner can directly influence how a patch in the bottom-right is interpreted.

### What is DINO?

**DINO** (Self-**Di**stillation with **No** labels) is a method for training a Vision Transformer *without any labels*. It trains the model by having it compare two differently-cropped views of the same image and trying to produce consistent representations. After training on ImageNet (1.2 million images, no labels used), DINO learns surprisingly rich visual features.

### Why DINO is useful for us

DINO's attention maps — the internal attention weights the model uses when looking at a patch — **naturally highlight objects**. Even though DINO was never told what an "object" is, its attention heads tend to focus on the foreground subject and ignore the background. This is the foundational observation that makes CutLER possible.

---

## 4. The CutLER Pipeline, Step by Step

CutLER works in three stages. Think of it like this: first generate rough labels, then train a detector on those labels, then optionally refine.

```
Stage 1: MaskCut        →   "Pseudo-masks" (rough object masks, no human input)
Stage 2: Train detector →   Mask R-CNN trained on pseudo-masks
Stage 3: Self-training  →   Use the detector to improve its own training data (optional)
```

---

### Stage 1: MaskCut — Generating Pseudo-Masks

**MaskCut** is an algorithm that takes an unlabeled image and produces binary masks (black-and-white images where white = "object", black = "background") for each object it finds.

**How it works:**

1. **Divide the image into patches** (e.g., an 480×480 image → 3,600 patches of 8×8 pixels each)

2. **Run DINO** to get a feature vector for each patch. Similar-looking patches get similar vectors.

3. **Build a similarity graph**: treat each patch as a node. Draw edges between patches, weighted by how similar their DINO features are.

4. **Spectral clustering (Normalized Cuts)**: find the mathematical split that divides the patches into two groups — "object" vs "background" — by minimizing the total similarity across the cut while maximizing similarity within each group. Concretely, this means computing the second-smallest eigenvector of a matrix derived from the graph, which naturally separates the two groups.

5. **Threshold the eigenvector** to get a binary mask (positive values = object, negative = background).

6. **CRF post-processing**: Conditional Random Field is a classic technique that sharpens the mask boundary — it pushes the mask to snap to actual image edges (color boundaries, texture changes) rather than sitting at patch centers.

7. **Repeat**: Mask out the object just found, then run steps 3–6 again to find the *next* object in the same image. Repeat N times (typically N=3) to get multiple masks per image.

**Output:** A JSON file mapping each image to a list of instance masks (one per detected object). These are called **pseudo-labels** — they were generated automatically, not by humans.

---

### Stage 2: Training the Detector

With pseudo-labels in hand, we now train a conventional object detector as if those pseudo-labels were real ground truth.

**The architecture: Cascade Mask R-CNN**

- **R-CNN** family: Region-based CNN. The model first proposes candidate regions of the image that might contain objects, then classifies and refines each region.
- **Cascade**: Multiple stages of refinement. Each stage takes the previous stage's output and improves it — better bounding boxes, more accurate masks.
- **Mask head**: In addition to bounding boxes, predicts a pixel-level mask for each detected instance.
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN). ResNet-50 is a standard deep CNN for feature extraction; FPN adds a multi-resolution feature hierarchy, helping detect objects at different sizes.
- **Initialization**: The backbone is initialized with DINO pre-trained weights (the same model used in Stage 1), giving it a head start.

**Class-agnostic training**: There is only one class — "object". The detector does not try to distinguish cats from cars. It just learns to find *things* in general. This is intentional: the pseudo-masks have no category information.

**Copy-paste augmentation**: During training, objects (masked regions) from other images are randomly pasted into the current image. This cheap trick significantly improves robustness and is standard in mask-based detection systems.

**Output:** A trained `.pth` checkpoint file — the trained model weights.

---

### Stage 3: Self-Training (Optional Refinement)

The detector from Stage 2 is imperfect, and so are the pseudo-masks from Stage 1. Self-training tries to break this chicken-and-egg problem:

1. Run the trained detector on the training images
2. Keep only high-confidence predictions (above a score threshold)
3. Treat these as the new pseudo-labels
4. Retrain the detector from scratch on the new labels
5. Repeat

Each iteration produces better labels → better detector → better labels. In practice, 1–2 rounds of self-training give meaningful gains.

---

### Evaluation

The trained detector is evaluated on **COCO val2017** — a standard benchmark with 5,000 images and ~36,000 annotated object instances spanning 80 categories. Since our detector is class-agnostic, evaluation is also class-agnostic (we just check whether the predicted boxes/masks overlap with ground-truth ones, regardless of category).

**Key metrics:**

| Metric | Meaning |
|--------|---------|
| **AP** | Average Precision (averaged over IoU thresholds 0.5–0.95) — main metric |
| **AP50** | Precision at IoU ≥ 0.50 (lenient) |
| **AP75** | Precision at IoU ≥ 0.75 (strict) |
| **APs** | AP for small objects (area < 32² pixels) |
| **APm** | AP for medium objects (32²–96² pixels) |
| **APl** | AP for large objects (> 96² pixels) |

**IoU** (Intersection over Union) measures how much a predicted box/mask overlaps with the ground-truth one. IoU = 1.0 means perfect overlap; IoU = 0 means no overlap. A prediction "counts" as correct if IoU exceeds the threshold.

**Our reproduced baseline results:**

| AP | AP50 | AP75 | APs | APm | APl |
|----|------|------|-----|-----|-----|
| 12.33 | 21.98 | 11.90 | 3.66 | 12.72 | 29.60 |

These match the CutLER paper, confirming our reproduction is correct.

---

## 5. The Problem: Small Objects Are Hard

Notice **APs = 3.66** — tiny compared to APl = 29.60. The detector finds large objects well but small ones poorly.

**Why?**

At native resolution (e.g., 480×480 input), a small object (say, a 20×20 pixel bird in the distance) maps to only **~6 patches** of size 8×8. With just 6 data points, the similarity graph is sparse, spectral clustering is unstable, and the eigenvector signal is weak and noisy. The object gets missed or produces a garbage mask.

**The intuition for the fix:** If we **zoom in** on that region of the image (crop it and resize to 480×480), the same bird now spans ~60 patches — 10× more signal. Spectral clustering becomes reliable and the object is found.

This is exactly what our multi-scale extension does.

---

## 6. Our Extension: Multi-Scale MaskCut

### The image pyramid concept

Instead of running MaskCut once at native resolution, we build an **image pyramid**: multiple versions of the image at different scales and crop positions.

```
Full image (1×)        → run MaskCut → masks at full scale
75% crops (0.75×)      → run MaskCut → masks at 75% scale, back-projected
50% crops (0.5×)       → run MaskCut → masks at 50% scale, back-projected
```

Each "crop" is a rectangular sub-region of the original image, resized to the standard input size (480×480). Crops overlap by 30% to avoid missing objects near crop edges.

### The pipeline

1. **Generate crop windows**: For each scale (1.0, 0.75, 0.5), create a grid of overlapping windows covering the full image
2. **Run MaskCut on each crop independently**: Each crop is treated as if it were a standalone image
3. **Back-project masks to full-image coordinates**: A mask in a 480×480 crop → scaled and shifted to the original image's coordinate system
4. **Merge all candidate masks**: Collect masks from all crops at all scales, then remove duplicates
   - Sort by mask quality/size
   - If two masks overlap by more than 50% IoU, keep only the better one (Non-Maximum Suppression / NMS)
5. **Output**: Final set of masks per image — same JSON format as original MaskCut

### Expected improvement

By running at finer scales, we recover small objects that were previously invisible at 1× resolution. The expected improvement is in **APs** — small-object AP — with possible minor decreases in APl (large objects are already found at 1× and don't need the zoom, so the extra crops mainly add noise there).

---

## 7. Repository: Every File Explained

### Top-level files

| File | What it is |
|------|-----------|
| `README.md` | Short project description and results table — the public face of the repo |
| `PROJECT_NOTES.md` | Running log: phase status, what worked, what broke, all results |
| `CLAUDE.md` | Instructions for Claude Code (the AI coding assistant used to write code) |
| `LICENSE` | MIT open-source license |
| `.gitignore` | Tells git which files NOT to track (large data files, model weights, logs) |
| `.gitmodules` | Tells git that `CutLER/` is a submodule (a link to another repo) |

---

### `CutLER/` — The Upstream Code (Do Not Edit)

This is a **git submodule**: a pointer to the original CutLER repository from Facebook Research. We use it as-is; our modifications live in `multiscale/` instead.

```
CutLER/
├── maskcut/
│   ├── maskcut.py      # Core MaskCut algorithm (Stage 1)
│   ├── dino.py         # Loads the DINO ViT model and extracts features
│   ├── crf.py          # CRF post-processing to sharpen mask edges
│   ├── merge_jsons.py  # Merges partial JSON outputs from parallel jobs
│   ├── demo.py         # Interactive demo: visualize masks on an image
│   └── colormap.py     # Color utilities for visualization
│
├── cutler/
│   ├── train_net.py              # Main training/evaluation script (Stage 2)
│   ├── config/                   # Config system (builds on Detectron2)
│   ├── data/                     # Dataset loading and registration
│   ├── modeling/                 # Neural network architecture definitions
│   ├── engine/                   # Training loop, hooks, checkpointing
│   ├── evaluation/               # COCO evaluation code
│   ├── tools/                    # Utility scripts (e.g., convert pseudo-labels)
│   └── model_zoo/configs/        # YAML configs for each experiment variant
│       ├── CutLER-ImageNet/      # Main unsupervised training configs
│       └── COCO-Semisupervised/  # Fine-tuning configs (1%/2%/5%/... labeled)
│
└── videocutler/                  # Video extension (not used in this project)
```

**Key config file — `cascade_mask_rcnn_R_50_FPN.yaml`**: Controls everything about the detector training: learning rate, batch size, number of training iterations, input image sizes, augmentation settings, etc.

---

### `multiscale/` — Our Custom Code

```
multiscale/
├── multiscale_maskcut.py    # The full multi-scale MaskCut implementation
└── MULTISCALE_MASKCUT.md    # Technical documentation for the script
```

**`multiscale_maskcut.py`** — This is the heart of our contribution. Key functions:

| Function | What it does |
|----------|-------------|
| `get_affinity_matrix()` | Builds the patch similarity graph from DINO features |
| `second_smallest_eigenvector()` | Runs spectral clustering (the math core of MaskCut) |
| `maskcut_single()` | Runs MaskCut on one image/crop, returns up to N masks |
| `get_crop_windows()` | Generates the sliding window grid at each scale |
| `filter_and_refine_masks()` | CRF refinement + quality filtering (rejects bad masks) |
| `merge_masks()` | NMS-style deduplication across all scales/crops |
| `maskcut_multicrop()` | Orchestrates everything for a single image |
| `save_to_coco_format()` | Converts masks to COCO JSON format for training |
| `main()` | CLI entry point: loops over a dataset, saves JSON output |

**Key command-line arguments you might tune:**

| Argument | Default | Meaning |
|----------|---------|---------|
| `--crop-scales` | `1.0,0.75,0.5` | Which zoom levels to use |
| `--crop-overlap` | `0.3` | Fraction of overlap between adjacent crops |
| `--merge-iou-thresh` | `0.5` | IoU above which two masks are considered duplicates |
| `--tau` | `0.2` | Sensitivity for the affinity graph |
| `--N` | `3` | Max masks per image/crop |
| `--small-first` | flag | Prefer keeping small masks when deduplicating |

---

### `slurm/` — HPC Job Scripts

All heavy computation runs on the Bocconi University HPC cluster via SLURM — a job scheduler that queues GPU jobs. These shell scripts are the interface between our code and the cluster.

| Script | Purpose | GPU time |
|--------|---------|----------|
| `run_eval.sh` | Evaluate pre-trained CutLER on COCO val2017 | ~2 hours |
| `run_maskcut.sh` | Generate pseudo-masks with original MaskCut | ~8 hours |
| `run_multiscale_maskcut.sh` | Generate pseudo-masks with our extension | ~16+ hours |
| `run_training.sh` | Train the Mask R-CNN detector | ~24 hours |
| `install_detectron2.sh` | One-time cluster setup: install Detectron2 | N/A |
| `download_checkpoint.sh` | Download pre-trained CutLER model weights | N/A |
| `download_data.sh` | Download COCO val2017 images and annotations | N/A |
| `fix_pillow.sh` | Fix a Pillow version conflict in the environment | N/A |

**How to submit a job:**
```bash
sbatch slurm/run_eval.sh        # submit to queue
squeue -u 3355142               # check status
tail -f logs/eval_<jobid>.out   # watch live output
```

---

### `experiments/` — Analysis and Config

| File | What it is |
|------|-----------|
| `environment.yml` | Conda environment specification — lists all Python packages and versions |
| `rank_small_ap.py` | Analysis script: loads evaluation results and ranks/filters by APs metric |

---

### `tools/` — Utility Scripts

| File | What it is |
|------|-----------|
| `make_cls_agnostic_coco.py` | Converts a standard COCO annotation JSON to class-agnostic format (replaces all category IDs with a single "object" category). Required before evaluating a class-agnostic detector on COCO. |

---

### `logs/` — Job Output

SLURM writes stdout and stderr from cluster jobs here. The directory is tracked by git (via `.gitkeep`) but the log files themselves are gitignored — they're often hundreds of MB. After a job runs, check here for errors or results.

---

## 8. Project Status (as of 2026-04-28)

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 1: Setup** | Done | Conda env, SLURM scripts, cluster paths, data downloaded |
| **Phase 2: Baseline** | Done | CutLER reproduced — AP=12.33 matches paper |
| **Phase 3: Multi-Scale** | In progress | Code written (`multiscale_maskcut.py`), not yet run on cluster |
| **Phase 4: Analysis** | Upcoming | Ablation studies, visualization, course report |

The immediate next step is to run `run_multiscale_maskcut.sh` on the cluster to generate multi-scale pseudo-labels, then retrain the detector and evaluate.

---

## 9. Development Workflow

Because training requires an A100 GPU (not available on a laptop), all heavy computation runs remotely.

```
1. Write / edit code here on Mac using Claude Code
        │
        ▼
2. git push origin <branch>     (send to GitHub)
        │
        ▼
3. SSH into cluster:  ssh 3355142@slogin.hpc.unibocconi.it
        │
        ▼
4. git pull                     (fetch latest code)
        │
        ▼
5. sbatch slurm/run_*.sh        (submit GPU job to queue)
        │
        ▼
6. squeue -u 3355142            (monitor job status)
        │
        ▼
7. Fetch results/logs back to Mac for analysis
```

The cluster uses **SLURM** (Simple Linux Utility for Resource Management) — a standard HPC job scheduler. You don't run code directly on the GPU machine; you submit a script to a queue and SLURM runs it when a GPU is free.

---

## 10. Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **COCO** | Common Objects in COntext — the standard benchmark dataset for object detection, with 80 categories and ~330k images |
| **AP** | Average Precision — the main detection accuracy metric, from 0 to 100 (higher = better) |
| **APs / APm / APl** | AP broken down by object size: small (< 32² px), medium, large |
| **IoU** | Intersection over Union — measures how much two boxes/masks overlap; 1.0 = perfect, 0 = no overlap |
| **ViT** | Vision Transformer — a neural network that processes images as sequences of patch tokens using the Transformer architecture |
| **DINO** | Self-supervised ViT pre-training method; learns visual features without any labels |
| **Pseudo-labels** | Automatically generated (noisy) training annotations, as opposed to human-annotated ground truth |
| **MaskCut** | The spectral clustering algorithm that generates pseudo-masks from DINO features |
| **Normalized Cuts** | The spectral graph partitioning algorithm underlying MaskCut; finds the best split of a graph into two parts |
| **Eigenvector** | A mathematical vector that captures the dominant structure of a matrix; in MaskCut, it encodes the object vs. background partition |
| **CRF** | Conditional Random Field — a post-processing technique that sharpens segmentation mask edges to align with image edges |
| **Mask R-CNN** | A widely-used architecture for instance segmentation that combines object detection with pixel-level masks |
| **Cascade** | A multi-stage refinement approach where each stage improves the previous stage's predictions |
| **Detectron2** | Facebook Research's object detection framework, which CutLER is built on top of |
| **Self-training** | Using a model's own predictions as training data to iteratively improve itself |
| **Image pyramid** | A set of copies of an image at different scales, used to detect objects at multiple resolutions |
| **NMS** | Non-Maximum Suppression — removes duplicate detections by keeping only the best-scoring one when two overlap significantly |
| **RLE** | Run-Length Encoding — a compact format for binary masks used in COCO annotation files |
| **SLURM** | The job scheduler used by the Bocconi HPC cluster; you submit shell scripts to a queue and it runs them on GPUs |
| **sbatch** | The SLURM command to submit a job script to the queue |
| **Conda env** | An isolated Python environment (`cutler`) with all dependencies installed — avoids version conflicts |
| **Submodule** | A git feature that lets one repository embed another (here, `CutLER/` is a submodule pointing to the Facebook Research repo) |
| **FPN** | Feature Pyramid Network — adds multi-scale feature maps to the backbone, helping detect objects at different sizes |
| **Class-agnostic** | Treating all objects as a single "object" category, ignoring what type of object they are |
| **WBF** | Weighted Box Fusion — an alternative to NMS for merging overlapping proposals by averaging rather than discarding |
