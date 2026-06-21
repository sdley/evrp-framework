# Results Directory

This directory stores generated experiment artifacts to keep the repository root clean.

## Structure

- `figures/`: standalone plots and paper figures (`.pdf`, `.png`)
- `data/`: exported tabular outputs (`.csv`)
- `studies/`: multi-file study outputs (e.g., ablation/state formulation tables + figures)
- `images/`: additional image assets generated during experiments
- `xai/`: model outputs and explainability run artifacts

## Conventions

- Keep deterministic scripts under `scripts/`; keep generated files under `results/`.
- Prefer subfolders per experiment (`results/studies/<study_name>/`) for reproducibility.
- Avoid writing new artifacts to repository root.
