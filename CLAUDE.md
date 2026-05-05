# Instructions for Claude Code

## Project context

This is a Bocconi University computer vision course project on unsupervised small-object detection. We extend Facebook Research's CutLER system by replacing its single-scale MaskCut pseudo-label generator with a hybrid heatmap-guided multi-scale variant. The goal is to improve small-object AP (APs) on COCO val2017. All heavy computation — pseudo-label generation, detector training, evaluation — runs on the Bocconi HPC cluster via SLURM. The local Mac is for code editing only; never suggest running training or MaskCut locally.

---

## Repo layout

- `CutLER/` — upstream Facebook Research submodule; do not edit directly
- `multiscale/` — our multi-scale MaskCut implementation and documentation
- `slurm/` — SLURM job scripts for all cluster jobs
- `tools/` — utility scripts (visualization, evaluation, dataset registration)
- `experiments/` — `environment.yml`, analysis scripts, committed eval outputs
- `presentation/` — report draft, slide planning, design decision docs
- `logs/` — gitignored; SLURM job stdout/stderr (check here after jobs finish)
- `debug/` — gitignored; single-image diagnostic outputs (contact sheets, overlays)

---

## When user asks for SLURM scripts, always:

- Use `#SBATCH --account=3355142`, `--partition=stud`, `--qos=stud`
- Activate the environment with: `module load miniconda3 && eval "$(conda shell.bash hook)" && conda activate cutler`
- Write output to `logs/` with a `%j` job ID suffix (e.g., `logs/myjob_%j.out`)
- Run `mkdir -p` on any output directory before writing to it
- Match the locked parameters from `PROJECT_NOTES.md` (ViT-S/8, tau=0.15, N=3, fixed_size=480, DINO weights) — any deviation invalidates the comparison

---

## When user asks to modify the upstream CutLER submodule:

- Don't. Use a wrapper or local override pattern instead. Example: `tools/train_wrapper.py` uses `runpy.run_path` to inject custom dataset registration without editing `CutLER/cutler/train_net.py`.
- Avoid heredoc `cat << 'EOF'` blocks in chat — markdown auto-conversion in some clients corrupts the quoting. Write the file via Claude Code locally then `git push`, or use `nano` directly on the cluster.

---

## Conventions

- Pseudo-label JSONs live on the cluster at `~/data/tiny-imagenet-10classes/annotations/` and are gitignored (regenerate from SLURM scripts)
- Large model checkpoints live at `~/cutler-multiscale/checkpoints/` on the cluster and are gitignored
- Final eval metrics (COCO AP tables) go in `experiments/` and are committed
- SLURM scripts use `${HOME}` for all paths — never hardcode `/home/3355142/`
- The `multiscale` split (not `combined`, not `raw_multiscale`) is the canonical training-ready pseudo-label output from all multi-scale runs
- Prefer small, focused commits tied to experiment milestones; keep `PROJECT_NOTES.md` updated after each phase transition
