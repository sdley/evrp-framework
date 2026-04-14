# RL4EVRP: Deep Reinforcement Learning Framework for Electric Vehicle Routing Problems

**RL4EVRP** is a modular, production-grade framework for solving Electric Vehicle Routing Problems (EVRP) using Deep Reinforcement Learning with built-in explainability features.

## Features

✨ **Core Capabilities**

- GAT (Graph Attention Transformer) encoder for node embedding
- Attention-based action decoder for interpretability
- A2C (Advantage Actor-Critic) training algorithm
- Multi-seed reproducibility and comprehensive diagnostics

🛠 **Architecture**

- **Configuration-driven**: All hyperparameters in YAML files
- **Modular design**: Easy to extend and customize
- **Best practices**: Gradient clipping, learning rate scheduling, entropy regularization
- **Energy-aware**: Proper battery management and feasibility constraints

🔍 **Explainability (XAI-ready)**

- Attention weight tracking for interpretability
- Per-step decision traces
- Counterfactual state analysis
- Feature importance through ablation (extensible)

## Quick Start

### 1. Installation

```bash
# Clone and navigate to framework
cd evrp-framework

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Configuration

The framework uses YAML-based configuration:

```
rl4evrp/config/
├── problem.yaml     # Problem size, chargers, capacities
├── model.yaml       # Architecture, training hyperparameters
└── env.yaml         # Device, LLM, output settings
```

Edit `rl4evrp/config/` files to customize:

```yaml
# problem.yaml
problem:
  n_customers: 15
  charger_prob: 0.15
  cargo_capacity: 30.0
  battery_capacity: 100.0

# model.yaml
training:
  lr: 3.0e-4
  gamma: 0.99
  entropy_coefficient: 0.01
```

### 3. Run the Framework

```python
import rl4evrp as rl

# Initialize framework
framework = rl.RL4EVRP()

# Read configs
model_config = framework.read_yaml('model')
problem_config = framework.read_yaml('problem')

# Build model
model_builder = framework.build()
model = model_builder.complete_model()

# Generate data
train_instances = [framework.generate_instance(seed=i) for i in range(200)]
eval_instances = [framework.generate_instance(seed=1000+i) for i in range(50)]

# Train
from rl4evrp.utils import train_agent
results = train_agent(
    model,
    train_instances,
    n_episodes=800,
    device=str(framework.device)
)

# Evaluate
from rl4evrp.utils import evaluate_agent
stats = evaluate_agent(model, eval_instances)
print(f"Mean distance: {stats['mean_distance']:.2f}")
```

See [`run.ipynb`](run.ipynb) for a complete example notebook.

## Project Structure

```
rl4evrp/
├── __init__.py              # Main RL4EVRP class
├── config/
│   ├── __init__.py         # Config management
│   ├── problem.yaml        # Problem configuration
│   ├── model.yaml          # Model & training config
│   └── env.yaml            # Environment configuration
├── environment/
│   └── __init__.py         # EVRPEnv, instance generation
├── models/
│   └── __init__.py         # Encoder, Decoder, Attention
├── agents/
│   └── __init__.py         # A2CAgent
├── utils/
│   └── __init__.py         # Training, evaluation, runners
├── config/
│   └── ...                 # YAML configuration files
├── run.ipynb               # Demo notebook
├── requirements.txt        # Python dependencies
└── setup.py               # Package setup
```

## API Overview

### Framework Initialization

```python
import rl4evrp as rl

# Initialize with default configs
framework = rl.RL4EVRP()

# Or with custom config directory
framework = rl.RL4EVRP(config_dir='/path/to/configs')
```

### Configuration Management

```python
# Read entire sections
problem_cfg = framework.read_yaml('problem')
model_cfg = framework.read_yaml('model')

# Get specific values with dot notation
n_customers = framework.config.get('problem.n_customers')
lr = framework.config.get('training.lr')

# Print all configs
framework.print_config()
```

### Model Building

```python
# Build models using builder pattern
builder = framework.build()

# Individual components
encoder = builder.encoder()
decoder = builder.decoder()

# Complete A2C agent
agent = builder.complete_model()
```

### Instance Generation

```python
# Single instance
inst = framework.generate_instance(seed=42)

# Multiple instances
instances = [framework.generate_instance(seed=i) for i in range(100)]
```

### Environment

```python
from rl4evrp.environment import EVRPEnv, generate_instance

# Create environment
inst = generate_instance(n_customers=15, seed=42)
env = EVRPEnv(inst, reward_mode='distance')

# Interact
obs = env.reset()
action = 1
next_obs, reward, done, info = env.step(action)
```

### Training & Evaluation

```python
from rl4evrp.utils import run_episode, train_agent, evaluate_agent

# Run single episode
reward, route, distance, info, transitions, traces, env = run_episode(
    agent,
    instance,
    collect_traces=True  # For XAI
)

# Train for multiple episodes
results = train_agent(
    agent,
    train_instances,
    n_episodes=800,
    eval_instances=eval_instances
)

# Evaluate on set of instances
stats = evaluate_agent(agent, test_instances, greedy=True)
```

## Configuration Files

### problem.yaml

Defines EVRP problem characteristics:

```yaml
problem:
  n_customers: 15 # Number of customers
  charger_prob: 0.15 # Fraction of chargers
  cargo_capacity: 30.0 # Vehicle capacity
  battery_capacity: 100.0 # Battery capacity

episode:
  n_episodes: 800 # Episodes per training run
  max_steps: null # null = 4 * #nodes
```

### model.yaml

Specifies architecture and training:

```yaml
encoder:
  type: gat # 'gat' or 'mlp'
  embed_dim: 128
  n_heads: 8
  n_layers: 3

training:
  lr: 3.0e-4
  gamma: 0.99
  entropy_coefficient: 0.01
  value_coefficient: 0.5
  seeds: [42, 123, 777]
```

### env.yaml

Configures runtime environment:

```yaml
device: cuda # 'cuda' or 'cpu'
output_directory: results_xai
llm:
  enabled: false
  provider: groq
  model: llama3-8b-8192
```

## Environment Details

### Node Types

- **Depot (0)**: Recharges both battery and cargo
- **Customer (1)**: Must be visited once, has demand
- **Charger (2)**: Recharges battery only

### Constraints

- **Battery**: Must have enough charge to reach destination and return to depot
- **Cargo**: Must have enough capacity to serve customer demand
- **Feasibility**: Invalid actions are masked during decoding

### Rewards

- **Service**: +0.2 per customer served
- **Completion**: +2.0 for completing route at depot
- **Distance mode**: -dist / d_max (minimize distance)
- **Inverse-distance mode**: 0.4 / (dist + 0.05) (reward short hops)
- **Violations**: -1.0 per battery violation

## Multi-Seed Training

Train with multiple random seeds for robustness:

```python
seeds = framework.get_seeds()  # [42, 123, 777]

all_results = {}
for seed in seeds:
    agent = builder.complete_model()  # Reinitialize
    results = train_agent(agent, train_instances)
    all_results[seed] = results
```

## Extensibility

### Custom Encoder

```python
from torch import nn
from rl4evrp.models import EVRPEncoder

# Subclass and override
class CustomEncoder(EVRPEncoder):
    def forward(self, x):
        # Your custom logic
        pass
```

### Custom Reward Function

```python
def custom_reward(env, action, info):
    # Your reward logic
    return reward_value
```

## Best Practices

✅ **Configuration**

- Keep problem/model/env configs separate
- Use YAML for easy hyperparameter tuning
- Version control your configs

✅ **Training**

- Use multiple seeds for statistical significance
- Save checkpoints at regular intervals
- Monitor training with tensorboard or plotly

✅ **Evaluation**

- Hold-out test set for unbiased evaluation
- Report mean ± std over multiple runs
- Compare with baselines

✅ **Code Quality**

- Type hints throughout
- Docstrings for all public APIs
- Modular components for reusability

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy, Pandas, Matplotlib
- PyYAML, python-dotenv

Optional:

- Groq API (for LLM explanations)
- Jupyter (for notebooks)

## License

[MIT License](LICENSE)

## Citation

If you use RL4EVRP in your research, please cite:

```bibtex
@software{rl4evrp2024,
  title={RL4EVRP: Deep Reinforcement Learning Framework for Electric Vehicle Routing},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl4evrp}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check existing documentation
- Review example notebooks

---

**Happy Reinforcement Learning! 🚗⚡**
