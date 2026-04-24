"""
High-level entry point that mirrors the original API:

    import rl4evrp as rl

    framework = rl.RL4EVRP()
    model     = framework.build().complete_model()
    instances = [framework.generate_instance(seed=i) for i in range(200)]
    results   = train_agent(model, instances, n_episodes=500)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import get_config
from .environment import EVRPEnv, generate_instance
from .models import EVRPEncoder, EVRPDecoder
from .agents import A2CAgent


class RL4EVRP:
    """High-level entry point: loads config, sets up device/seed/output, exposes builder."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: Path to a directory containing problem.yaml / model.yaml / env.yaml.
                        Defaults to the built-in config directory.
        """
        self.config = get_config(config_dir)
        self._setup_device()
        self._setup_seed()
        self._setup_output()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_device(self):
        device_cfg = self.config.get("env.device", "auto")
        if device_cfg in ("cuda", "auto"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"device: {self.device}")

    def _setup_seed(self):
        seed = self.config.get("env.reproducibility.seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.get("env.reproducibility.deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_output(self):
        output_dir = self.config.get("env.output_directory", "results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> "ModelBuilder":
        """Return a ModelBuilder pre-loaded with this framework's config."""
        return ModelBuilder(self)

    def generate_instance(self, seed: int = None) -> dict:
        """Generate a single EVRP instance using the problem config defaults."""
        return generate_instance(
            n_customers=self.config.get("problem.problem.n_customers", 15),
            seed=seed,
            charger_prob=self.config.get("problem.problem.charger_prob", 0.15),
            cargo_cap=self.config.get("problem.problem.cargo_capacity", 30.0),
            battery_cap=self.config.get("problem.problem.battery_capacity", 100.0),
        )

    def create_environment(self, inst: dict, reward_mode: str = "distance") -> EVRPEnv:
        """Wrap an instance dict in an EVRPEnv."""
        return EVRPEnv(inst, reward_mode=reward_mode)

    def get_seeds(self) -> List[int]:
        """Return the list of training seeds from model.yaml."""
        return self.config.get("model.training.seeds", [42, 123, 777])

    def read_yaml(self, section: str) -> Dict:
        """Return an entire YAML section as a plain dict (e.g. 'model', 'problem')."""
        return self.config.get_section(section)

    def print_config(self):
        """Pretty-print all loaded configuration values."""
        self.config.print_config()


class ModelBuilder:
    """Construct encoder, decoder, or a full A2C agent from the loaded config."""

    def __init__(self, framework: RL4EVRP):
        self.framework = framework
        self.model_cfg = framework.config.get_section("model")

    def encoder(self) -> EVRPEncoder:
        enc = self.model_cfg.get("encoder", {})
        return EVRPEncoder(
            input_dim=self.framework.config.get("problem.node_features.feature_dim", 7),
            embed_dim=enc.get("embed_dim", 128),
            n_heads=enc.get("n_heads", 8),
            n_layers=enc.get("n_layers", 3),
            enc_type=enc.get("type", "gat"),
        ).to(self.framework.device)

    def decoder(self) -> EVRPDecoder:
        dec = self.model_cfg.get("decoder", {})
        return EVRPDecoder(
            embed_dim=dec.get("embed_dim", 128),
            n_heads=dec.get("n_heads", 8),
        ).to(self.framework.device)

    def agent(self) -> A2CAgent:
        train = self.model_cfg.get("training", {})
        enc   = self.model_cfg.get("encoder", {})
        a = A2CAgent(
            embed_dim=enc.get("embed_dim", 128),
            n_heads=enc.get("n_heads", 8),
            n_layers=enc.get("n_layers", 3),
            lr=train.get("lr", 3e-4),
            gamma=train.get("gamma", 0.99),
            entropy_coef=train.get("entropy_coefficient", 0.01),
            value_coef=train.get("value_coefficient", 0.5),
            enc_type=enc.get("type", "gat"),
            n_episodes=self.framework.config.get("problem.episode.n_episodes", 500),
            device=str(self.framework.device),
        )
        print(f"agent params: {sum(p.numel() for p in a.parameters()):,}")
        return a

    def complete_model(self) -> A2CAgent:
        """Build and return a fully configured A2CAgent (same as agent())."""
        return self.agent()
