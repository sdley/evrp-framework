# RL4EVRP Quick Start Guide

Get started with RL4EVRP in 5 minutes!

## Installation

```bash
# Install the package
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1️⃣ Initialize Framework

```python
import rl4evrp as rl

# Create framework instance
framework = rl.RL4EVRP()

# Framework loads all configs automatically
print(framework)  # Shows loaded configuration sections
```

### 2️⃣ Build Model

```python
# Create model builder
builder = framework.build()

# Build A2C agent (contains encoder + decoder)
model = builder.complete_model()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {model.device}")
```

### 3️⃣ Generate Data

```python
# Generate training instances
train_instances = [framework.generate_instance(seed=i) for i in range(200)]

# Generate evaluation instances (different seeds)
eval_instances = [framework.generate_instance(seed=1000+i) for i in range(50)]

print(f"Training instances: {len(train_instances)}")
print(f"Evaluation instances: {len(eval_instances)}")
```

### 4️⃣ Train Agent

```python
from rl4evrp.utils import train_agent

# Train for 800 episodes (or fewer to test)
results = train_agent(
    model,
    train_instances,
    n_episodes=800,
    device=str(framework.device),
    eval_instances=eval_instances,
    save_interval=50
)

# Results contain:
# - train_rewards: episode rewards
# - losses: training loss per episode
# - entropies: policy entropy per episode
# - eval_rewards: evaluation performance
```

### 5️⃣ Evaluate

```python
from rl4evrp.utils import evaluate_agent

# Evaluate on test set
stats = evaluate_agent(
    model,
    eval_instances,
    greedy=True,
    n_eval=50
)

print(f"Mean distance: {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
```

## Configuration

Edit YAML files to customize:

### `rl4evrp/config/problem.yaml`

```yaml
problem:
  n_customers: 15 # Problem size
  charger_prob: 0.15 # Charger density
  cargo_capacity: 30.0
  battery_capacity: 100.0
```

### `rl4evrp/config/model.yaml`

```yaml
encoder:
  embed_dim: 128 # Embedding dimension
  n_heads: 8 # Attention heads
  n_layers: 3 # Transformer layers

training:
  lr: 3.0e-4 # Learning rate
  gamma: 0.99 # Discount factor
  entropy_coefficient: 0.01
```

### `rl4evrp/config/env.yaml`

```yaml
device: cuda # cuda or cpu
output_directory: results_xai
```

## Advanced Features

### Running Single Episode

```python
from rl4evrp.utils import run_episode

# Run episode and collect route
reward, route, distance, info, transitions, traces, env = run_episode(
    model,
    instance,
    greedy=True,           # Use greedy policy
    collect_traces=True    # Collect XAI traces
)

print(f"Route: {route}")
print(f"Distance: {distance:.2f}")
print(f"Reward: {reward:.2f}")
```

### Counterfactual Analysis

```python
from rl4evrp.xai import CounterfactualAnalyzer

# Analyze how battery affects decisions
cf_analysis = CounterfactualAnalyzer.analyze_sensitivity(
    model,
    observation,
    perturbation_factors=[0.5, 0.75, 1.0, 1.25, 1.5]
)

print(f"Original action: {cf_analysis['original_action']}")
print(f"With half battery: {cf_analysis['battery_perturbations'][0.5]}")
```

### Feature Importance

```python
from rl4evrp.xai import FeatureImportance

# Ablation-based importance for feature
importance = FeatureImportance.logit_ablation(
    model,
    observation,
    node_idx=5,
    feature_idx=0  # x coordinate
)

print(f"Feature importance: {importance:.4f}")
```

## Multi-Seed Training

For robust results, train with multiple seeds:

```python
import torch
import numpy as np
import random

seeds = framework.get_seeds()  # [42, 123, 777]

all_results = {}
for seed in seeds:
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Rebuild model with seed
    model = builder.complete_model()

    # Train
    results = train_agent(model, train_instances, n_episodes=800)
    all_results[seed] = results

# Average results across seeds
print(f"Avg final reward: {np.mean([r['train_rewards'][-1] for r in all_results.values()]):.2f}")
```

## Common Tasks

### Save & Load Model

```python
import torch

# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model.load_state_dict(torch.load('model.pt'))
```

### Visualize Training

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Training rewards
axes[0, 0].plot(results['train_rewards'])
axes[0, 0].set_title('Training Reward')

# Training loss
axes[0, 1].plot(results['losses'])
axes[0, 1].set_title('Training Loss')

# Entropy
axes[1, 0].plot(results['entropies'])
axes[1, 0].set_title('Policy Entropy')

# Eval rewards
if results['eval_rewards']:
    axes[1, 1].plot(results['eval_rewards'])
    axes[1, 1].set_title('Evaluation Reward')

plt.tight_layout()
plt.show()
```

### Visualize Route

```python
import matplotlib.pyplot as plt
from rl4evrp.utils import run_episode

# Run episode
reward, route, distance, info, _, _, env = run_episode(model, instance, greedy=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
coords = instance['coords']
types = instance['node_types']

# Depot
ax.plot(coords[0, 0], coords[0, 1], 'r*', markersize=20, label='Depot')

# Customers
mask = types == 1
ax.scatter(coords[mask, 0], coords[mask, 1], c='blue', label='Customers')

# Chargers
mask = types == 2
if mask.any():
    ax.scatter(coords[mask, 0], coords[mask, 1], c='green', marker='s', label='Chargers')

# Route
route_coords = coords[route]
ax.plot(route_coords[:, 0], route_coords[:, 1], 'k-', alpha=0.3)

ax.set_title(f'Route (Distance={distance:.2f})')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'rl4evrp'`

- **Solution**: Run `pip install -e .` in the framework directory

**Issue**: CUDA out of memory

- **Solution**: Reduce `embed_dim` or `n_heads` in `model.yaml`, or set `device: cpu` in `env.yaml`

**Issue**: Poor training convergence

- **Solution**: Try adjusting `lr`, `entropy_coefficient`, or `gamma` in `model.yaml`

## Next Steps

1. **Explore examples**: Check `run.ipynb` for a complete example
2. **Read the docs**: See `README.md` for detailed API documentation
3. **Customize configs**: Modify YAML files to match your problem
4. **Extend framework**: Implement custom encoders, decoders, or reward functions
5. **Use XAI tools**: Leverage attention tracking and counterfactual analysis

## Documentation

- 📖 [Full README](README.md)
- 📓 [Example Notebook](run.ipynb)
- 🔧 [API Reference](#)
- 🎯 [Architecture Guide](#)

Happy training! 🚗⚡
