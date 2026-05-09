# SLURM Scripts

All heavy computation runs on the Bocconi HPC cluster through SLURM.

## Canonical scripts

These are the scripts that matter for the final project story.

### Pseudo-label generation

| Script | Role |
|---|---|
| `run_maskcut_baseline.sh` | baseline single-scale MaskCut generation |
| `run_multiscale_maskcut.sh` | refined hybrid multi-scale generation through `multiscale/multiscale_maskcut.py` |
| `run_hybrid_ablation_100.sh` | one named 100-image hybrid ablation |
| `run_hybrid_ablations_100_onejob.sh` | full 100-image hybrid ablation suite in a single job |

### Training and evaluation

| Script | Role |
|---|---|
| `run_training_luiz.sh` | main detector training script used in the 5-class study |
| `run_eval.sh` | class-agnostic COCO evaluation |

### Utilities

| Script | Role |
|---|---|
| `run_singlescale_maskcut.sh` | diagnostic single-scale run through `multiscale_maskcut.py` |
| `run_speedtest.sh` | throughput profiling |
| `run_visualize.sh` | cluster-side pseudo-mask visualization |

## Historical or non-primary scripts

These are kept for reference, but they are not the recommended path for the final project:

| Script | Status |
|---|---|
| `run_maskcut.sh` | legacy |
| `run_multiscale_maskcut_tinyimagenet.sh` | older hardcoded path |
| `run_hybrid_maskcut_tinyimagenet.sh` | older hybrid snapshot runner |
| `run_maskcut_baseline_coco.sh` | older COCO-oriented baseline helper |
| `run_multiscale_maskcut_coco.sh` | older COCO-oriented multiscale helper |
| `run_training.sh` | older generic training runner |
| `submit_hybrid_ablations_100.sh` | separate-job submission path, superseded by one-job ablation for the final study |
| `run_final_multiscale_luiz.sh` | one-off historical fork runner |

## One-time setup helpers

| Script | Role |
|---|---|
| `install_detectron2.sh` | install Detectron2 in the cluster environment |
| `download_checkpoint.sh` | download checkpoints and helper assets |
| `download_data.sh` | download COCO data |
| `download_tinyimagenet.sh` | download TinyImageNet subset |
| `fix_pillow.sh` | fix Pillow version issues if needed |

## Final workflow

```bash
# 1. Baseline masks
sbatch slurm/run_maskcut_baseline.sh

# 2. Refined hybrid masks
sbatch slurm/run_multiscale_maskcut.sh

# 3. Merge baseline + refined hybrid
python tools/combine_pseudo_labels.py ...

# 4. Train detector
sbatch slurm/run_training_luiz.sh

# 5. Evaluate
sbatch slurm/run_eval.sh
```

## Notes

- The final project result is based on **combined pseudo-labels**, not hybrid-only pseudo-labels.
- The authoritative method implementation is `multiscale/multiscale_maskcut.py`.
