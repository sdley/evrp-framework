# Refactoring Notes

## Repository cleanup (2026-06)

- Moved one-off experiment drivers from root into `scripts/experiments/`
- Moved figure generation scripts into `scripts/figures/`
- Moved validation scripts into `tests/`
- Consolidated documentation to `README.md` and `QUICKSTART.md`
- Simplified onboarding script (`INSTALL.sh`) and packaging metadata (`setup.py`)

## Non-goals of this cleanup

- No changes to published conference result artifacts
- No destructive pruning of notebooks or generated figures/CSVs
- No behavior changes to the core RL package API
