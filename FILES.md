# RL4EVRP Framework - Complete File Structure

## Root Directory

```
evrp-framework/
├── .env                      # Environment variables (local, not in git)
├── .env.example              # Template for .env file
├── .gitignore                # Git ignore patterns
├── README.md                 # Full documentation & API reference
├── QUICKSTART.md             # 5-minute quick start guide
├── REFACTORING.md            # Refactoring summary & migration guide
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
├── run.ipynb                 # Example notebook (like your screenshot)
└── rl4evrp/                  # Main framework package
```

## Framework Package Structure

```
rl4evrp/
├── __init__.py                              # Main RL4EVRP class & public API
│   - RL4EVRP()                              # Framework initializer
│   - ModelBuilder                           # Model construction pattern
│   - read_yaml()                            # Utility for reading configs
│
├── config/
│   ├── __init__.py                          # Config loader class
│   │   - Config                             # Configuration manager
│   │   - get_config()                       # Global config instance
│   ├── problem.yaml                         # Problem configuration
│   │   - Problem size, capacities, episodes
│   ├── model.yaml                           # Model & training config
│   │   - Architecture, hyperparameters
│   └── env.yaml                             # Environment configuration
│       - Device, output, LLM settings
│
├── environment/
│   └── __init__.py                          # Environment module
│       - EVRPEnv                            # Main environment class
│       - generate_instance()                # Single instance generator
│       - make_dataset()                     # Batch instance generator
│       - build_node_features()              # Feature engineering
│
├── models/
│   └── __init__.py                          # Model components
│       - MultiHeadAttention                 # Attention mechanism
│       - GATEncoderLayer                    # Transformer encoder layer
│       - EVRPEncoder                        # Full encoder (GAT or MLP)
│       - EVRPDecoder                        # Action decoder
│
├── agents/
│   └── __init__.py                          # RL agents
│       - A2CAgent                           # A2C agent (complete model)
│
├── utils/
│   └── __init__.py                          # Utilities & runners
│       - run_episode()                      # Single episode runner
│       - train_agent()                      # Training loop
│       - evaluate_agent()                   # Evaluation harness
│
└── xai.py                                   # Explainable AI utilities
    - AttentionTracer                        # Trace attention patterns
    - CounterfactualAnalyzer                 # State perturbation analysis
    - FeatureImportance                      # Feature importance methods
```

## Configuration Files (YAML)

### `rl4evrp/config/problem.yaml`

Defines the EVRP problem characteristics:

- Number of customers (excluding depot)
- Charger probability
- Vehicle capacities (cargo & battery)
- Episode settings and max steps
- Reward modes (distance vs inverse-distance)
- Node feature definitions

### `rl4evrp/config/model.yaml`

Specifies neural network and training:

- **Encoder**: Type (GAT/MLP), dimensions, heads, layers
- **Decoder**: Architecture parameters
- **Value head**: Hidden dimensions
- **Training**: Optimizer, learning rate, scheduling
- **A2C-specific**: Gamma, entropy/value coefficients
- **Multiple seeds**: For reproducibility

### `rl4evrp/config/env.yaml`

Runtime environment configuration:

- Device selection (CUDA/CPU)
- Output directory for results
- LLM configuration (Groq API)
- Reward tuning parameters
- Diagnostics settings
- Reproducibility options

## Key Classes & APIs

### RL4EVRP (Main Framework)

```python
class RL4EVRP:
    def __init__(config_dir=None)         # Initialize with configs
    def read_yaml(section)                # Get config section
    def build()                           # Get ModelBuilder
    def generate_instance(seed)           # Generate EVRP instance
    def create_environment(inst, mode)    # Create EVRPEnv
    def get_seeds()                       # Get training seeds
    def print_config()                    # Print all configs
```

### Environment (EVRPEnv)

```python
class EVRPEnv:
    def __init__(inst, reward_mode)
    def reset()                           # Reset to initial state
    def step(action)                      # Take one step
    # Properties: battery, cargo, route, total_d, visited, etc.
```

### Agent (A2CAgent)

```python
class A2CAgent(nn.Module):
    def select_action(obs, greedy=False)  # Choose action
    def update(transitions)               # Train on episode
    def _forward(obs, return_attn=False)  # Forward pass
    def get_action_for_inference(obs)     # Inference only
```

### Training Utils

```python
def run_episode(agent, inst, device, greedy, collect_traces)
    # Returns: reward, route, distance, info, transitions, traces, env

def train_agent(agent, instances, n_episodes, device, eval_instances)
    # Returns: results dict with metrics

def evaluate_agent(agent, instances, device, greedy, n_eval)
    # Returns: stats dict with mean/std metrics
```

## File Organization by Responsibility

### Configuration & Setup

- `.env` - Secrets (GROQ_API_KEY, etc.)
- `.gitignore` - Version control
- `requirements.txt` - Dependencies
- `setup.py` - Package installation
- `YAML files` - All hyperparameters

### Documentation

- `README.md` - Full API documentation
- `QUICKSTART.md` - 5-minute start guide
- `REFACTORING.md` - What changed & why
- `FILES.md` - This file

### Code Modules

- `rl4evrp/__init__.py` - Framework orchestration
- `rl4evrp/config/` - Configuration management
- `rl4evrp/environment/` - Problem & environment
- `rl4evrp/models/` - Neural architecture
- `rl4evrp/agents/` - RL agent
- `rl4evrp/utils/` - Training & utilities
- `rl4evrp/xai.py` - Explainable AI

### Examples

- `run.ipynb` - End-to-end example

## Usage Pattern

### 1. Configure

```bash
# Edit configuration files
vim rl4evrp/config/problem.yaml    # Problem size
vim rl4evrp/config/model.yaml      # Architecture
vim .env                          # Secrets
```

### 2. Import Framework

```python
import rl4evrp as rl
framework = rl.RL4EVRP()
```

### 3. Build & Train

```python
model = framework.build().complete_model()
instances = [framework.generate_instance(i) for i in range(200)]
results = rl4evrp.utils.train_agent(model, instances)
```

### 4. Evaluate & Deploy

```python
stats = rl4evrp.utils.evaluate_agent(model, test_instances)
torch.save(model.state_dict(), 'model.pt')
```

## What's New vs Original Notebook

| Aspect           | Before              | After                     |
| ---------------- | ------------------- | ------------------------- |
| Organization     | Monolithic notebook | Modular package           |
| Lines of code    | 2300+ in one file   | ~600 per module           |
| Configuration    | Hardcoded values    | YAML files                |
| Reusability      | Copy-paste code     | Import & use              |
| Testing          | Not possible        | Easy unit tests           |
| Documentation    | Markdown cells      | Docstrings + README       |
| Environment vars | Scattered           | Centralized .env          |
| Reproducibility  | Implicit            | Explicit (seeds, configs) |
| Extensibility    | Fork & modify       | Subclass & extend         |

## Quick Reference

### Initialize

```python
import rl4evrp as rl
framework = rl.RL4EVRP()
```

### Configure

Edit `rl4evrp/config/problem.yaml`, `model.yaml`, `env.yaml`

### Generate Data

```python
instances = [framework.generate_instance(seed=i) for i in range(200)]
```

### Build Model

```python
model = framework.build().complete_model()
```

### Train

```python
from rl4evrp.utils import train_agent
results = train_agent(model, instances, n_episodes=800)
```

### Evaluate

```python
from rl4evrp.utils import evaluate_agent
stats = evaluate_agent(model, eval_instances)
```

### Run Episode

```python
from rl4evrp.utils import run_episode
reward, route, dist, info, _, _, env = run_episode(model, inst, greedy=True)
```

### XAI Analysis

```python
from rl4evrp.xai import CounterfactualAnalyzer, FeatureImportance
cf = CounterfactualAnalyzer.analyze_sensitivity(model, obs)
imp = FeatureImportance.logit_ablation(model, obs, node_idx, feat_idx)
```

## Next Steps

1. **Install**: `pip install -e .`
2. **Read**: Start with `QUICKSTART.md`
3. **Run**: Execute `run.ipynb`
4. **Customize**: Edit YAML configs
5. **Extend**: Add custom modules as needed

---

**Framework is ready for Development, Research, and Deployment! 🚀**
