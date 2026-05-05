# SLURM Scripts

All heavy computation runs on the Bocconi HPC cluster via SLURM. Submit jobs with `sbatch`, monitor with `squeue -u <username>`, tail logs with `tail -f logs/<jobid>.out`.

## One-Time Setup (run once on first cluster login)

| Script | Purpose |
|--------|---------|
| `install_detectron2.sh` | Install Detectron2 via miropsota pre-built wheels. Required before any training run. |
| `download_checkpoint.sh` | Download `cutler_cascade_final.pth` (pretrained CutLER) and pre-generated MaskCut annotations. DINO weights must be downloaded separately — see `README.md` Setup section. |
| `download_data.sh` | Download COCO val2017 images and annotations to `~/data/coco/`. |
| `download_tinyimagenet.sh` | Download TinyImageNet-200 and create the 10-class subset at `~/data/tiny-imagenet-10classes/`. |
| `fix_pillow.sh` | Fix a Pillow version conflict. Run only if you see Pillow import errors. |

## Pseudo-Label Generation

| Script | Status | Purpose |
|--------|--------|---------|
| `run_maskcut_baseline.sh` | **Primary** | Single-scale baseline MaskCut on 10-class TinyImageNet. Use this for the controlled comparison. |
| `run_multiscale_maskcut.sh` | **Primary** | Multi-scale MaskCut. Requires env vars — see `README.md` for the exact invocations for hybrid and MOST-lite methods. |
| `run_singlescale_maskcut.sh` | Diagnostic | Runs without `--multi-crop` via `multiscale_maskcut.py`. For ablation only. |
| `run_speedtest.sh` | Diagnostic | Benchmarks per-image throughput on 1-class subset. Use to profile the ~48 s/image speed regression. |
| `run_maskcut.sh` | Legacy | Original CutLER-scale MaskCut (full TinyImageNet-200). Not used in the current 10-class comparison. |
| `run_multiscale_maskcut_tinyimagenet.sh` | Legacy | Older hardcoded multi-scale variant. Superseded by `run_multiscale_maskcut.sh`. |

## Training and Evaluation

| Script | Purpose |
|--------|---------|
| `run_training.sh` | Train Cascade Mask R-CNN. Set `PSEUDO_LABEL_NAME=baseline` or `multiscale` before submitting. |
| `run_eval.sh` | Evaluate a checkpoint on COCO val2017 (class-agnostic). |
| `run_visualize.sh` | Run `tools/visualize_pseudo_masks.py` on the cluster and save PNGs to `experiments/visualizations/`. |

## Setting Your SLURM Account

No script hardcodes an account number. Before submitting any job, set your own account one of two ways:

```bash
# Option A — set once in your shell session or ~/.bashrc:
export SBATCH_ACCOUNT=<your_student_number>

# Option B — pass it per submission:
sbatch --account=<your_student_number> slurm/run_maskcut_baseline.sh
```

SLURM picks up `SBATCH_ACCOUNT` automatically when the `#SBATCH --account` line is absent from the script.
