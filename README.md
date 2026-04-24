# rl4evrp

**Deep Reinforcement Learning for the Electric Vehicle Routing Problem with Explainable AI**

[![CI](https://github.com/sdley/evrp-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/sdley/evrp-framework/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`rl4evrp` is a modular Python framework for solving the **Electric Vehicle Routing Problem (EVRP)** using deep reinforcement learning. An A2C agent with a Graph Attention Network (GAT) encoder learns to route a vehicle through customers while managing battery charge and cargo capacity. Built-in XAI tools expose attention weights, per-step decision traces, and counterfactual sensitivity analysis.

---

## Features

**Reinforcement learning core**

- A2C (Advantage Actor-Critic) agent with entropy regularisation and gradient clipping
- Graph Attention Network (GAT) encoder — or swap to MLP with one flag
- Cross-attention decoder with feasibility masking (invalid actions are never selected)
- Cosine annealing learning-rate schedule and configurable discount factor

**Energy-aware environment**

- Battery and cargo constraints enforced at every step
- Three node types: depot (refills both), customer (serves demand), charger (refills battery)
- Configurable step limit, reward bonuses/penalties, and two reward modes (`distance` / `inverse_distance`)

**Scalable training**

- `OnTheFlyInstancePool` — generates instances lazily; no RAM overhead for large training sets
- Multi-seed reproducibility out of the box
- Checkpoint saving at configurable intervals; resume-friendly

**Explainability (XAI)**

- `AttentionTracer` — capture encoder and decoder attention weights at every step
- `CounterfactualAnalyzer` — measure how greedy actions shift under battery/cargo perturbations
- `FeatureImportance` — ablation-based logit sensitivity per node feature
- `GroqExplainer` — natural-language explanations of episodes, individual steps, and training diagnostic reports via Groq LLM (`pip install "rl4evrp[llm]"`)

**Developer experience**

- `src`-layout, `pyproject.toml`, `uv`-managed dependencies
- 130+ pytest tests with mocked external calls (no API key needed in CI)
- GitHub Actions CI on Python 3.10 / 3.11 / 3.12 with ruff linting
- YAML-driven configuration with dot-notation access and env-var interpolation
- Ready-to-use `scripts/train.py` and `scripts/evaluate.py` CLIs

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Features](#features)
- [The problem](#the-problem)
- [Package overview](#package-overview)
- [Environment API](#environment-api)
- [Model API](#model-api)
- [Agent API](#agent-api)
- [Training & evaluation](#training--evaluation)
- [Configuration](#configuration)
- [XAI tools](#xai-tools)
- [Scripts](#scripts)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Installation

> **Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0

### Recommended — [uv](https://docs.astral.sh/uv/)

`uv` manages the virtual environment for you automatically.

```bash
# 1. install uv (once)
curl -Ls https://astral.sh/uv/install.sh | sh

# 2. clone and enter the project
git clone https://github.com/sdley/evrp-framework
cd evrp-framework

# 3. create the virtualenv and install all dependencies
uv sync                  # creates .venv/ and installs core deps
uv sync --extra dev      # also installs pytest, jupyter, ruff

# 4. run anything inside the env
uv run python scripts/train.py --help
uv run jupyter notebook notebooks/quickstart.ipynb
```

### pip + venv (manual)

```bash
git clone https://github.com/sdley/evrp-framework
cd evrp-framework

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# install the package in editable mode
pip install -e .
# with dev extras (pytest, jupyter, ruff):
pip install -e ".[dev]"
```

### pip (once published to PyPI)

```bash
python -m venv .venv
source .venv/bin/activate
pip install rl4evrp
```

---

## Quick start

### High-level API (paper style)

If you want to stay close to how the framework was used in the paper, use `RL4EVRP` and `ModelBuilder`. They read all settings from the YAML config files automatically.

> **Note:** see [Configuration](#configuration) for how to customise or override the YAML files.

```python
import rl4evrp as rl
from rl4evrp.utils import train_agent, evaluate_agent

# --- initialise (loads config, sets device + seed) ---
framework = rl.RL4EVRP()                      # reads src/rl4evrp/config/*.yaml
# or point to your own config directory:
# framework = rl.RL4EVRP(config_dir="my_configs/")

# --- inspect config ---
model_config   = framework.read_yaml("model")
problem_config = framework.read_yaml("problem")
framework.print_config()

# --- build agent from config ---
model = framework.build().complete_model()    # A2CAgent with YAML hyperparams

# --- generate data ---
train_instances = [framework.generate_instance(seed=i)          for i in range(200)]
eval_instances  = [framework.generate_instance(seed=1_000 + i)  for i in range(50)]

# --- train ---
results = train_agent(model, train_instances, n_episodes=500,
                      eval_instances=eval_instances,
                      device=str(framework.device))

# --- evaluate ---
stats = evaluate_agent(model, eval_instances, greedy=True)
print(f"mean distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
print(f"mean reward:   {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
```

### Low-level API

If you prefer explicit control over every hyperparameter without relying on YAML files:

```python
from rl4evrp.environment import generate_instance
from rl4evrp.agents import A2CAgent
from rl4evrp.utils import OnTheFlyInstancePool, train_agent, evaluate_agent

# --- data ---
train_pool     = OnTheFlyInstancePool(lambda seed: generate_instance(n_customers=15, seed=seed), size=1000)
eval_instances = [generate_instance(n_customers=15, seed=10_000 + i) for i in range(50)]

# --- model ---
agent = A2CAgent(embed_dim=128, n_heads=8, n_layers=3,
                 lr=3e-4, gamma=0.99, n_episodes=500, device="cpu")

# --- train ---
results = train_agent(agent, train_pool, n_episodes=500,
                      eval_instances=eval_instances, save_interval=100)

# --- evaluate ---
stats = evaluate_agent(agent, eval_instances, greedy=True)
print(f"mean distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
print(f"mean reward:   {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
```

Or use the command-line scripts:

```bash
uv run python scripts/train.py    --n-episodes 500 --out-dir results/run1
uv run python scripts/evaluate.py results/run1/agent_final.pt --n-eval 100
```

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for an end-to-end walkthrough.

---

## The problem

The **Electric Vehicle Routing Problem** asks: find the shortest route for a vehicle starting and ending at a depot, visiting every customer exactly once, subject to:

- **Battery constraint** — the vehicle cannot travel further than its current charge allows. It must be able to reach the next node *and* return to the depot (or a charger) without running out.
- **Cargo constraint** — the vehicle cannot serve a customer if it carries less than that customer's demand.
- **Recharging** — the vehicle can visit charger nodes (partial graph) to refill its battery, or return to the depot to refill both battery and cargo.

### Node types

| Type | Index | On visit |
|---|---|---|
| **Depot** | 0 | Refills battery **and** cargo to full |
| **Customer** | 1 | Consumes cargo equal to its demand; marks node as visited |
| **Charger** | 2 | Refills battery to full; no demand |

Nodes are placed uniformly in the unit square. Each instance is fully parameterised by a seed, so training data is generated on the fly without storing anything.

### Reward signal

| Event | Value |
|---|---|
| Travel step | `−dist / d_max` |
| Serve a customer | `+0.2` |
| Complete route (all served + depot) | `+2.0` |
| Return to depot early | `−0.3` |
| Use a charger | `−0.05` |
| Battery violation | `−1.0` |

---

## Package overview

```
src/rl4evrp/
├── __init__.py            # public API re-exports
├── config/
│   ├── config.py          # Config loader (YAML + .env, dot-notation access)
│   ├── problem.yaml       # Problem size, rewards, node features
│   ├── model.yaml         # Architecture, optimiser, training hyperparams
│   └── env.yaml           # Device, checkpointing, LLM integration
├── environment/
│   ├── instances.py       # generate_instance · build_node_features · make_dataset
│   └── env.py             # EVRPEnv
├── models/
│   ├── attention.py       # MultiHeadAttention
│   ├── encoder.py         # GATEncoderLayer · EVRPEncoder
│   └── decoder.py         # EVRPDecoder
├── agents/
│   └── a2c.py             # A2CAgent
├── utils/
│   ├── pool.py            # OnTheFlyInstancePool
│   └── training.py        # run_episode · train_agent · evaluate_agent
└── xai/
    ├── attention.py       # AttentionTracer
    ├── counterfactual.py  # CounterfactualAnalyzer
    └── importance.py      # FeatureImportance · collect_traces_during_episode
```

All public names are re-exported from each subpackage's `__init__.py`, so both of these work:

```python
from rl4evrp.models import EVRPEncoder          # via __init__
from rl4evrp.models.encoder import EVRPEncoder  # direct
```

---

## Environment API

### `generate_instance`

```python
from rl4evrp.environment import generate_instance

inst = generate_instance(
    n_customers=15,     # customers (depot is always node 0)
    seed=42,            # for reproducibility
    charger_prob=0.15,  # fraction of customer nodes converted to chargers
    cargo_cap=30.0,
    battery_cap=100.0,
)
# inst is a plain dict:
# {
#   "coords":       ndarray (n_nodes, 2) — positions in [0, 1]²
#   "demands":      ndarray (n_nodes,)   — 0 for depot/chargers
#   "node_types":   ndarray (n_nodes,)   — 0=depot, 1=customer, 2=charger
#   "cargo_cap":    float
#   "battery_cap":  float
#   "n_nodes":      int
# }
```

### `build_node_features`

Converts an instance dict into the 7-dimensional tensor the encoder expects:

```python
from rl4evrp.environment import build_node_features

feats = build_node_features(inst)  # Tensor (n_nodes, 7)
# columns: x, y, demand_norm, is_charger, is_depot, cargo_cap_norm, battery_cap_norm
```

### `make_dataset`

Pre-generates a fixed list of instances (small datasets, testing):

```python
from rl4evrp.environment import make_dataset

instances = make_dataset(n=200, n_customers=15, seed0=0)
# returns list of 200 instance dicts
```

### `EVRPEnv`

Gym-style environment:

```python
from rl4evrp.environment import EVRPEnv

env = EVRPEnv(inst, reward_mode="distance")  # or "inverse_distance"

obs = env.reset()
# obs keys: node_features, current_node, battery_norm, cargo_norm,
#           visited_mask, valid_mask

while not env.done:
    action = ...                          # index in [0, n_nodes)
    obs, reward, done, info = env.step(action)

# info keys: dist, total_dist, all_served, n_customers_served,
#            battery, cargo, charger_visits, batt_violations, cargo_violations
```

`obs["valid_mask"]` is a boolean array of shape `(n_nodes,)`. `True` means the node is a legal next action. The mask already enforces battery feasibility and cargo capacity — just sample from valid indices.

---

## Model API

The three model components can be used independently:

### `MultiHeadAttention`

```python
from rl4evrp.models import MultiHeadAttention
import torch

mha = MultiHeadAttention(n_heads=8, input_dim=128, embed_dim=128)

# self-attention
out = mha(x)                                # (B, T, 128)

# cross-attention
out = mha(q, k, v)                          # (B, Tq, 128)

# with attention weights
out, attn = mha(x, return_attn=True)        # attn: (B, heads, T, T)

# with mask (True = mask out)
out = mha(x, mask=invalid_mask)
```

### `EVRPEncoder`

```python
from rl4evrp.models import EVRPEncoder

encoder = EVRPEncoder(
    input_dim=7,        # node feature dimension
    embed_dim=128,
    n_heads=8,
    n_layers=3,
    enc_type="gat",     # "gat" (default) or "mlp"
)

node_emb, graph_emb = encoder(feats)
# node_emb:  (B, n_nodes, 128) — per-node embeddings
# graph_emb: (B, 128)          — mean pooling of node_emb

# After a forward pass, encoder.last_attn holds the final-layer attention
# weights (B, n_heads, n_nodes, n_nodes) for XAI analysis.
```

### `EVRPDecoder`

```python
from rl4evrp.models import EVRPDecoder

decoder = EVRPDecoder(embed_dim=128, n_heads=8)

logits = decoder(
    node_emb,       # (B, n_nodes, 128)
    graph_emb,      # (B, 128)
    battery_norm,   # (B, 1)
    cargo_norm,     # (B, 1)
    cur_node_idx,   # (B,) long
    invalid_mask,   # (B, n_nodes) bool — True = illegal action
)
# logits: (B, n_nodes) — feed to Categorical for action sampling
```

---

## Agent API

`A2CAgent` wraps encoder + decoder + value head into a single trainable module:

```python
from rl4evrp.agents import A2CAgent

agent = A2CAgent(
    embed_dim=128,
    n_heads=8,
    n_layers=3,
    lr=3e-4,
    gamma=0.99,
    entropy_coef=0.01,   # entropy regularisation
    value_coef=0.5,      # weight of the value loss
    enc_type="gat",      # "gat" | "mlp"
    n_episodes=500,      # used to set the cosine LR schedule period
    device="cpu",        # "cpu" | "cuda"
)
```

### Selecting actions

```python
# during training (stochastic, gradients tracked)
action, log_prob, entropy, value = agent.select_action(obs, greedy=False)

# during evaluation (greedy, no grad)
action = agent.get_action_for_inference(obs, greedy=True)
```

### Updating

```python
# collect transitions over one episode
transitions = []
obs = env.reset()
while not done:
    action, log_prob, entropy, value = agent.select_action(obs)
    obs, reward, done, info = env.step(action)
    transitions.append({"r": reward, "lp": log_prob, "ent": entropy, "val": value})

loss, mean_entropy = agent.update(transitions)
```

### Saving and loading

```python
import torch

# save
torch.save(agent.state_dict(), "agent.pt")

# load
agent2 = A2CAgent(embed_dim=128, n_heads=8, n_layers=3, n_episodes=1)
agent2.load_state_dict(torch.load("agent.pt", map_location="cpu", weights_only=True))
agent2.eval()
```

---

## Training & evaluation

### `OnTheFlyInstancePool`

Generates instances lazily from a seed schedule. Behaves like a list but never materialises more than one instance in memory at a time — safe for very large training budgets:

```python
from rl4evrp.utils import OnTheFlyInstancePool

pool = OnTheFlyInstancePool(
    generate_fn=lambda seed: generate_instance(n_customers=15, seed=seed),
    size=100_000,
    seed_offset=0,
)

inst = pool[42]          # generates on access, wraps at size boundary
len(pool)                # 100000
```

### `train_agent`

```python
from rl4evrp.utils import train_agent

results = train_agent(
    agent,
    train_instances=pool,        # OnTheFlyInstancePool or list of dicts
    n_episodes=500,
    device="cpu",
    save_dir="results/checkpoints",   # None = no checkpoints
    eval_instances=eval_instances,    # None = no periodic eval
    save_interval=100,                # checkpoint + eval every N episodes
)

# results is a dict:
# {
#   "train_rewards": list[float],
#   "eval_rewards":  list[float] | None,
#   "losses":        list[float],
#   "entropies":     list[float],
# }
```

### `run_episode`

Run a single episode with full control:

```python
from rl4evrp.utils import run_episode

total_reward, route, total_dist, info, transitions, traces, env = run_episode(
    agent, inst,
    device="cpu",
    greedy=True,
    collect_traces=True,    # populate `traces` for XAI; slower
    bat_perturb=0.5,        # optional: multiply battery_norm by factor
    cargo_perturb=None,
)

# route: list of visited node indices
# traces: list of per-step dicts (only if collect_traces=True)
```

### `evaluate_agent`

```python
from rl4evrp.utils import evaluate_agent

stats = evaluate_agent(
    agent,
    instances=eval_instances,
    device="cpu",
    greedy=True,
    n_eval=50,              # cap the number evaluated (None = all)
)

# stats keys: mean_reward, std_reward, mean_distance, std_distance,
#             rewards (list), distances (list), routes (list)
```

---

## Configuration

The framework reads three YAML files from `src/rl4evrp/config/`. You can point to a custom directory:

```python
from rl4evrp.config import get_config

cfg = get_config()                              # default: package config dir
cfg = get_config(config_dir="/my/configs")     # custom directory

# dot-notation access with optional default
n  = cfg.get("problem.problem.n_customers", default=15)
lr = cfg.get("model.training.lr", default=3e-4)

# entire section as dict
model_cfg = cfg.get_section("model")
```

Values support environment variable interpolation:

```yaml
llm:
  api_key: ${GROQ_API_KEY}   # resolved at runtime from os.environ
```

### `problem.yaml` — instance and reward parameters

```yaml
problem:
  n_customers: 25
  charger_prob: 0.15
  cargo_capacity: 30.0
  battery_capacity: 100.0

episode:
  n_episodes: 500000
  max_steps_factor: 4    # episode limit = factor × n_nodes

reward:
  service_bonus: 0.2
  completion_bonus: 2.0
  charger_penalty: 0.05
  early_return_penalty: 0.3
  battery_violation_penalty: 1.0
```

### `model.yaml` — architecture and optimiser

```yaml
encoder:
  type: gat          # "gat" | "mlp"
  embed_dim: 128
  n_heads: 8
  n_layers: 3
  ff_dim: 256

training:
  lr: 3.0e-4
  gamma: 0.99
  entropy_coefficient: 0.01
  value_coefficient: 0.5
  grad_clip_norm: 1.0
  seeds: [42, 123, 777]
```

### `env.yaml` — runtime settings

```yaml
device: cuda              # "cuda" | "cpu"
output_directory: results_xai
checkpoint_interval: 5000

llm:
  enabled: false          # set true to enable Groq explanations
  api_key: ${GROQ_API_KEY}

reproducibility:
  seed: 42
  deterministic: true
```

### Custom YAML configuration

You are not limited to the built-in files. `Config` loads **all** `*.yaml` files it finds in the config directory, so you can drop any file in and read it with the same dot-notation API.

**1. Create your own config directory**

```
my_project/
└── configs/
    ├── problem.yaml    # override built-in defaults
    ├── model.yaml
    ├── env.yaml
    └── experiment.yaml # your own section — any name works
```

**2. Override built-in values**

Copy the file you want to change and edit only the keys you care about — the rest use their defaults when you call `cfg.get(..., default=...)`:

```yaml
# configs/problem.yaml  — only override what you need
problem:
  n_customers: 50
  battery_capacity: 150.0

reward:
  completion_bonus: 3.0
```

**3. Add your own sections**

```yaml
# configs/experiment.yaml
experiment:
  name: large-scale-run
  notes: "50 customers, bigger battery"
  tags: [v2, ablation]
```

**4. Load and read**

```python
from rl4evrp.config import Config

cfg = Config(config_dir="my_project/configs")

# override values
n = cfg.get("problem.problem.n_customers")   # -> 50

# your own section
name  = cfg.get("experiment.experiment.name")          # -> "large-scale-run"
tags  = cfg.get("experiment.experiment.tags")          # -> ["v2", "ablation"]

# inspect everything
cfg.print_config()

# get a whole section as a plain dict
exp = cfg.get_section("experiment")
```

**5. Environment variable interpolation**

Any string value of the form `${VAR_NAME}` is resolved from `os.environ` at read time:

```yaml
# configs/experiment.yaml
experiment:
  groq_key: ${GROQ_API_KEY}
  output_dir: ${OUTPUT_ROOT}/run1
```

```python
import os
os.environ["OUTPUT_ROOT"] = "/data/runs"
cfg.get("experiment.experiment.output_dir")   # -> "/data/runs/run1"
```

> The built-in config files live at `src/rl4evrp/config/` and are used as defaults when you call `get_config()` without arguments. Pointing to a custom directory **replaces** the defaults entirely, so copy over any files you still want.

---

## XAI tools

### Attention traces

Collect per-step attention data during an episode:

```python
from rl4evrp.xai import collect_traces_during_episode, analyze_decision_path

traces = collect_traces_during_episode(agent, inst, device="cpu")

# Each trace dict contains:
# step, from_node, to_node, node_type,
# battery_norm, cargo_norm, battery_abs, cargo_abs,
# action_prob, action_logit, raw_logits,
# top3_nodes, top3_probs,
# dec_attn,   # decoder cross-attention over nodes, shape (n_nodes,)
# enc_attn,   # encoder self-attention row for current node, shape (n_nodes,)
# dist_to_action, dist_to_depot,
# is_charger, is_depot, is_forced

# Summarise visits to a specific node
path = analyze_decision_path(traces, node_idx=3)
print(path["visits"])    # [step_idx, ...]
print(path["avg_prob"])  # average action probability when visiting node 3
print(path["max_prob"])
```

Attach a tracer to an existing episode loop:

```python
from rl4evrp.xai import AttentionTracer

tracer = AttentionTracer()
# pass collect_traces=True to run_episode and store the returned traces:
*_, traces, _ = run_episode(agent, inst, collect_traces=True)
for t in traces:
    tracer.add_trace(t)

all_traces = tracer.get_traces()
```

### Counterfactual analysis

Measure how the greedy action changes when you perturb battery or cargo state:

```python
from rl4evrp.xai import CounterfactualAnalyzer

obs = env.reset()

result = CounterfactualAnalyzer.analyze_sensitivity(
    agent, obs,
    perturbation_factors=[0.25, 0.5, 1.0, 1.5, 2.0],
)

print(result["original_action"])
print(result["battery_perturbations"])  # {factor: action, ...}
print(result["cargo_perturbations"])    # {factor: action, ...}

# or build a single counterfactual manually:
obs_low_battery = CounterfactualAnalyzer.perturb_battery(obs, factor=0.25)
obs_full_cargo   = CounterfactualAnalyzer.perturb_cargo(obs,  factor=2.0)
```

### Feature importance

Ablation-based importance: how much does zeroing one feature change a node's logit?

```python
from rl4evrp.xai import FeatureImportance

# feature indices: 0=x, 1=y, 2=demand_norm, 3=is_charger,
#                  4=is_depot, 5=cargo_cap_norm, 6=battery_cap_norm

score = FeatureImportance.logit_ablation(
    agent, obs,
    node_idx=3,     # which node to analyse
    feature_idx=2,  # which feature to zero
)
print(f"importance: {score:.4f}")   # absolute logit delta
```

### LLM explanations (Groq)

`GroqExplainer` uses the [Groq](https://groq.com/) API to generate natural-language explanations of agent decisions. It requires the `llm` extra and a `GROQ_API_KEY`.

**Install**

```bash
# uv
uv sync --extra llm

# pip
pip install "rl4evrp[llm]"
```

**Configure**

```bash
export GROQ_API_KEY="gsk_..."
# or add it to a .env file at the project root
```

**Explain a full episode**

```python
from rl4evrp.xai import GroqExplainer, collect_traces_during_episode

explainer = GroqExplainer()                     # reads GROQ_API_KEY from env
# optionally: GroqExplainer(api_key="gsk_...", model="llama3-8b-8192")

traces = collect_traces_during_episode(agent, inst, device="cpu")
explanation = explainer.explain_episode(traces, inst)
print(explanation)
# > "The agent efficiently served all customers while managing battery life by
#    visiting charger node 4 proactively at 38 % charge before the long detour
#    to the cluster around nodes 11–14 …"
```

Pass a custom question to focus the explanation:

```python
explanation = explainer.explain_episode(
    traces, inst,
    question="Why did the agent visit the charger before node 11?",
)
```

**Explain a single step**

```python
step_trace = traces[5]   # one dict from collect_traces_during_episode()
explanation = explainer.explain_step(step_trace, inst)
print(explanation)
# > "At 38 % battery the agent chose to detour to charger node 4 rather than
#    heading directly to node 11, because the direct route would leave it with
#    insufficient charge to return to the depot."
```

**Explain a training diagnostic**

`explain_diagnostic` reads the same numeric data that `scripts/diagnostic.py` uses to draw its figures and produces a plain-English interpretation — no image upload required.

```python
from rl4evrp.xai import GroqExplainer

# diag, drift, scores come from diagnostic.py's analyze_checkpoint /
# compute_drift / compute_health_scores, or pass drift=None if you have
# no reference checkpoint.
explanation = explainer.explain_diagnostic(
    diag,
    scores,
    drift=drift,                        # optional
    output_path="diagnostic_report.txt" # optional — saves alongside the PDF
)
print(explanation)
# > "The model scores 34/100 overall and is still in early development.
#    Layer 1 shows the highest encoder drift (3.0) suggesting it is where
#    most task-relevant structure is forming, while Layer 0 has barely moved
#    from initialisation …"
```

Or add `--explain` directly to the diagnostic script:

```bash
python scripts/diagnostic.py checkpoint.pt --output diagnostic.pdf --explain
# LLM explanation is printed and saved as diagnostic_explanation.txt

# with a reference checkpoint for drift analysis:
python scripts/diagnostic.py checkpoint.pt --reference init.pt --explain \
    --explain-output report.txt
```

**GroqExplainer API**

| Method | Returns | Description |
|---|---|---|
| `explain_episode(traces, inst, question=None)` | `str` | 3–5 sentence strategy summary for a full episode |
| `explain_step(trace, inst)` | `str` | 1–2 sentence explanation of a single decision |
| `explain_diagnostic(diag, scores, drift=None, output_path=None)` | `str` | Plain-English interpretation of a training diagnostic report |

The `traces` argument is the list returned by `collect_traces_during_episode()`. The `inst` argument is the instance dict returned by `generate_instance()`. The `diag` and `scores` arguments are the dicts returned by `analyze_checkpoint()` and `compute_health_scores()` in `scripts/diagnostic.py`.

> **Note:** `groq` is an optional dependency. Importing `rl4evrp` or `rl4evrp.xai` without the `llm` extra installed works fine — the `ImportError` is raised only when you instantiate `GroqExplainer`.

---

## Scripts

### `scripts/train.py`

```
usage: train.py [-h] [--n-customers N] [--charger-prob P] [--cargo-cap C]
                [--battery-cap B] [--n-episodes N] [--pool-size N]
                [--seed S] [--embed-dim D] [--n-heads H] [--n-layers L]
                [--lr LR] [--gamma G] [--enc-type {gat,mlp}]
                [--out-dir PATH] [--save-interval N] [--n-eval N]
                [--device {cpu,cuda,auto}]
```

Example:

```bash
uv run python scripts/train.py \
    --n-customers 25     \
    --n-episodes 2000    \
    --embed-dim 128      \
    --out-dir results/run1
```

Outputs: `results/run1/checkpoints/agent_episode_*.pt`, `results/run1/agent_final.pt`, `results/run1/results.json`.

### `scripts/evaluate.py`

```bash
uv run python scripts/evaluate.py results/run1/agent_final.pt \
    --n-eval 200 \
    --out results/run1/eval.json
```

Flags must match training settings (`--embed-dim`, `--n-layers`, etc.).

### `scripts/diagnostic.py`

Inspects a checkpoint's internal weight statistics to answer *"has my model actually learned something?"* — independent of reward curves.

```
usage: diagnostic.py [-h] [--reference REF] [--output PATH] [--name NAME]
                     [--explain] [--explain-output PATH]
                     checkpoint
```

```bash
# basic health report → model_diagnostic.pdf
python scripts/diagnostic.py results/run1/agent_final.pt

# compare against an earlier checkpoint to see where learning happened
python scripts/diagnostic.py results/run1/agent_final.pt \
    --reference results/run1/checkpoints/agent_episode_0.pt \
    --output diagnostic.pdf

# add an LLM explanation (requires GROQ_API_KEY and rl4evrp[llm])
python scripts/diagnostic.py results/run1/agent_final.pt --explain
```

The PDF contains ten panels covering health scores (0–100), per-layer attention weight statistics, feature embedding spread, encoder vs decoder weight distributions, per-component drift, LayerNorm adaptation, and a plain-English verdict. The `--explain` flag appends a Groq-generated narrative saved alongside the PDF.

---

## Testing

```bash
uv run pytest                         # all 124 tests
uv run pytest -v                      # verbose output
uv run pytest tests/test_models.py    # one module
uv run pytest -k "test_encoder"       # keyword filter
```

Test coverage by module:

| File | What is tested |
|---|---|
| `test_instances.py` | `generate_instance`, `build_node_features`, `make_dataset` |
| `test_env.py` | `EVRPEnv` reset, valid mask, step mechanics, distance matrix |
| `test_models.py` | `MultiHeadAttention`, `GATEncoderLayer`, `EVRPEncoder` (gat+mlp), `EVRPDecoder` |
| `test_agents.py` | `A2CAgent` forward, select_action, update, inference |
| `test_utils.py` | `OnTheFlyInstancePool`, `run_episode`, `evaluate_agent` |
| `test_xai.py` | `AttentionTracer`, `CounterfactualAnalyzer`, `FeatureImportance`, trace helpers, `GroqExplainer` (episode, step, diagnostic — mocked API) |
| `test_config.py` | `Config` dot-notation, sections, singleton, YAML loading |

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Run `uv run pytest` — all tests must pass
4. Run `uv run ruff check src/ tests/` — no linting errors
5. Open a pull request against `develop`

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{rl4evrp2026,
  title  = {rl4evrp: Deep Reinforcement Learning for the Electric Vehicle
            Routing Problem with Explainable AI},
  author = {Noucier, Dimeth and Diallo, Souleymane and Habibi, Imen and
            Mabiala, Jeremie and Diouf, Mame Diarra and Mulamba, Elie},
  year   = {2026},
  url    = {https://github.com/sdley/evrp-framework}
}
```

## Acknowledgements
This project was developed as part of the 2024 IDEATHON organized by the Deep Learning Indaba 2024. 
We thank all the the Deep Learning Indaba organizers and mentors for their support and guidance throughout the development of this framework.
