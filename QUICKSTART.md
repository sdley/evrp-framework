# Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Sanity check import

```bash
python - <<'PY'
import rl4evrp as rl
framework = rl.RL4EVRP()
model = framework.build().complete_model()
print('Framework ready on', framework.device)
print('Parameter count:', sum(p.numel() for p in model.parameters()))
PY
```

## 3. Generate sample data and train briefly

```bash
python - <<'PY'
import rl4evrp as rl
from rl4evrp.utils import train_agent

framework = rl.RL4EVRP()
model = framework.build().complete_model()
instances = [framework.generate_instance(seed=i) for i in range(16)]
train_agent(model, instances, n_episodes=10, device=str(framework.device))
print('Short training run completed')
PY
```

## 4. Open notebook workflow

- Main demo: `run.ipynb`
- Additional study notebooks: `decision.ipynb`, `variant.ipynb`, `full_state_transfer_eval.ipynb`

## 5. Run organized scripts

- Experiments: `scripts/experiments/`
- Figures: `scripts/figures/`
- Validation/test scripts: `tests/`
