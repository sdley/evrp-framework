# RL4EVRP

A research-oriented framework for Electric Vehicle Routing Problems (EVRP) with deep reinforcement learning and explainability tooling.

## What is in this repository

- Reusable package code in `rl4evrp/`
- Config-driven experiments via YAML in `rl4evrp/config/`
- Reproducible notebooks for analysis in the repository root
- Offline experiment and figure scripts in `scripts/`

## Repository layout

```text
.
├── rl4evrp/                  # Package source
│   ├── agents/
│   ├── config/
│   ├── environment/
│   ├── models/
│   ├── utils/
│   └── xai.py
├── scripts/
│   ├── experiments/          # Experiment runners and comparisons
│   └── figures/              # Figure generation scripts
├── tests/                    # Validation/test scripts
├── run.ipynb                 # Main runnable notebook example
├── QUICKSTART.md             # Practical 5-minute setup
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick usage

```python
import rl4evrp as rl
from rl4evrp.utils import train_agent, evaluate_agent

framework = rl.RL4EVRP()
model = framework.build().complete_model()

train_instances = [framework.generate_instance(seed=i) for i in range(200)]
eval_instances = [framework.generate_instance(seed=1000 + i) for i in range(50)]

results = train_agent(model, train_instances, n_episodes=800, eval_instances=eval_instances)
stats = evaluate_agent(model, eval_instances)
print(stats["mean_distance"])
```

## Running scripts

- Experiments: `python scripts/experiments/<script>.py ...`
- Figures: `python scripts/figures/<script>.py ...`
- Validation: `python tests/<script>.py ...`

## Reproducibility

- Hyperparameters and environment settings are in YAML files under `rl4evrp/config/`
- Set seeds through config and/or your script entry points
- Keep generated artifacts (PDF/PNG/CSV/checkpoints) out of package code

## Notes

This repository still includes conference result artifacts (plots/csv/notebooks). The codebase cleanup organizes execution code and docs, while preserving existing outputs.
