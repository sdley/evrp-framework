# Scripts

This folder contains non-package executable scripts used for experiments and paper figures.

- `experiments/`: experiment runners, variant comparisons, diagnostics
- `figures/`: scripts that generate publication-ready figures from checkpoints/results

These scripts are intentionally separate from `rl4evrp/` to keep package code clean and reusable.

Output convention:

- Write generated artifacts under `results/`
- Use `results/studies/` for multi-file experiment runs
- Use `results/figures/` for standalone figure exports
