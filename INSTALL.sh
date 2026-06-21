#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Installing dependencies"
python3 -m pip install -r requirements.txt

echo "[2/3] Installing package in editable mode"
python3 -m pip install -e .

echo "[3/3] Running import smoke test"
python - <<'PY'
import rl4evrp as rl
framework = rl.RL4EVRP()
model = framework.build().complete_model()
print('OK: framework import and model build succeeded')
print('Device:', framework.device)
print('Parameters:', sum(p.numel() for p in model.parameters()))
PY

echo "Setup completed successfully"
