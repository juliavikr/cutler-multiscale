# presentation/

This folder contains everything needed to produce the course report and slides for the
**cutler-multiscale** project. Keep it updated as results come in.

## Folder map

| Folder / File | Purpose |
|---|---|
| `01_results/` | Numbers, tables, runtime stats — the raw material |
| `02_visualizations/` | Hand-picked mask overlay images for slides/report figures |
| `03_diagrams/` | Text descriptions of figures to draw (pipeline, crop strategy) |
| `04_design_decisions/` | The *why* behind each choice — useful for the discussion section |
| `05_story/` | Narrative: slide outline, elevator pitch, anticipated Q&A |
| `report_draft.md` | IMRaD skeleton for the written report |

## How to use this for the report

1. Start with `05_story/outline.md` to agree on the story arc.
2. Fill `01_results/comparison_table_template.md` once multiscale results are in.
3. Pick 2–4 images from `02_visualizations/` to include as figures.
4. Use `03_diagrams/` descriptions to draw figures in Excalidraw / draw.io / Mermaid.
5. Copy numbers and rationale into `report_draft.md` and write.

## Status

- [x] Baseline results available
- [ ] Multiscale pseudo-labels — pending Luiz's finalization
- [ ] Multiscale training + evaluation — pending
- [ ] Visualizations selected — pending (run `sbatch slurm/run_visualize.sh` on cluster first)
