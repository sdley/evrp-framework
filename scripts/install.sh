#!/usr/bin/env bash
# Install and smoke-test the rl4evrp package with uv.
set -euo pipefail

echo "RL4EVRP — installation"
echo "======================"

if ! command -v uv &>/dev/null; then
    echo "uv not found. Install it with:"
    echo "  curl -Ls https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv $(uv --version)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Installing from $PROJECT_ROOT ..."
uv sync

echo ""
echo "Running smoke test ..."
uv run python - <<'PYEOF'
from rl4evrp.environment import generate_instance, EVRPEnv
from rl4evrp.agents import A2CAgent

inst = generate_instance(n_customers=10, seed=0)
agent = A2CAgent(embed_dim=64, n_heads=4, n_layers=2, n_episodes=5, device="cpu")
obs = EVRPEnv(inst).reset()
action, *_ = agent.select_action(obs)
print(f"OK — first action: {action}")
PYEOF

echo ""
echo "Done. Next steps:"
echo "  uv run python scripts/train.py --help"
echo "  uv run python scripts/evaluate.py --help"
echo "  uv run jupyter notebook notebooks/quickstart.ipynb"
