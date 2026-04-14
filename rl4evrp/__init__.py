"""
RL4EVRP: Main framework class for Electric Vehicle Routing Problem with RL.
Deep Reinforcement Learning Framework for Electric Vehicle Routing with Explainable AI.

Quick Start:
    >>> import rl4evrp as rl
    >>> framework = rl.RL4EVRP()
    >>> model = framework.build().complete_model()
    >>> instances = [framework.generate_instance(seed=i) for i in range(200)]
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Core imports
from .config import Config, get_config
from .environment import EVRPEnv, generate_instance, make_dataset, build_node_features
from .models import EVRPEncoder, EVRPDecoder, MultiHeadAttention
from .agents import A2CAgent
from .utils import run_episode, train_agent, evaluate_agent

# XAI utilities
from . import xai

# Version
__version__ = "0.1.0"
__author__ = "RL4EVRP Contributors"

# Public API
__all__ = [
    'RL4EVRP',
    'ModelBuilder',
    'EVRPEnv',
    'A2CAgent',
    'EVRPEncoder',
    'EVRPDecoder',
    'generate_instance',
    'make_dataset',
    'build_node_features',
    'run_episode',
    'train_agent',
    'evaluate_agent',
    'Config',
    'get_config',
    'xai',
]


class RL4EVRP:
    """Main framework for EVRP with Deep RL and XAI."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the RL4EVRP framework.
        
        Args:
            config_dir: Path to directory with YAML configs
        """
        self.config = get_config(config_dir)
        self._setup_device()
        self._setup_seed()
        self._setup_output()
    
    def _setup_device(self):
        """Setup compute device (CUDA or CPU)."""
        device_cfg = self.config.get('env.device', 'cuda')
        
        if device_cfg == 'cuda' or device_cfg == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"✓ Device: {self.device}")
    
    def _setup_seed(self):
        """Setup reproducibility seeds."""
        seed = self.config.get('env.reproducibility.seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Set deterministic mode
        if self.config.get('env.reproducibility.deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_output(self):
        """Setup output directory."""
        output_dir = self.config.get('env.output_directory', 'results_xai')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"✓ Output directory: {self.output_dir}")
    
    def read_yaml(self, section: str) -> Dict:
        """
        Read a configuration section.
        
        Args:
            section: Section name (e.g., 'problem', 'model', 'training')
        
        Returns:
            Configuration dictionary
        """
        return self.config.get_section(section)
    
    def build(self) -> 'ModelBuilder':
        """
        Builder pattern for model construction.
        
        Returns:
            ModelBuilder instance
        """
        return ModelBuilder(self)
    
    def generate_instance(self, seed: int = None) -> dict:
        """Generate a random EVRP instance."""
        cfg = self.config.get_section('problem')
        return generate_instance(
            n_customers=cfg.get('n_customers', 15),
            seed=seed,
            charger_prob=cfg.get('charger_prob', 0.15),
            cargo_cap=cfg.get('cargo_capacity', 30.0),
            battery_cap=cfg.get('battery_capacity', 100.0)
        )
    
    def create_environment(self, inst: dict, reward_mode: str = 'distance') -> EVRPEnv:
        """Create EVRP environment."""
        inst['reward_mode'] = reward_mode
        return EVRPEnv(inst, reward_mode=reward_mode)
    
    def get_seeds(self) -> List[int]:
        """Get training seeds from config."""
        return self.config.get('model.training.seeds', [42, 123, 777])
    
    def print_config(self):
        """Print all loaded configurations."""
        self.config.print_config()


class ModelBuilder:
    """Builder for constructing models and agents."""
    
    def __init__(self, framework: 'RL4EVRP'):
        """
        Initialize builder.
        
        Args:
            framework: RL4EVRP framework instance
        """
        self.framework = framework
        self.model_cfg = framework.config.get_section('model')
        self.problem_cfg = framework.config.get_section('problem')
    
    def encoder(self) -> EVRPEncoder:
        """Build encoder."""
        enc_cfg = self.model_cfg.get('encoder', {})
        return EVRPEncoder(
            input_dim=self.framework.config.get('problem.node_features.feature_dim', 7),
            embed_dim=enc_cfg.get('embed_dim', 128),
            n_heads=enc_cfg.get('n_heads', 8),
            n_layers=enc_cfg.get('n_layers', 3),
            enc_type=enc_cfg.get('type', 'gat')
        ).to(self.framework.device)
    
    def decoder(self) -> EVRPDecoder:
        """Build decoder."""
        dec_cfg = self.model_cfg.get('decoder', {})
        return EVRPDecoder(
            embed_dim=dec_cfg.get('embed_dim', 128),
            n_heads=dec_cfg.get('n_heads', 8)
        ).to(self.framework.device)
    
    def agent(self) -> A2CAgent:
        """Build A2C agent."""
        model_cfg = self.model_cfg
        train_cfg = model_cfg.get('training', {})
        encoder_cfg = model_cfg.get('encoder', {})
        
        agent = A2CAgent(
            embed_dim=encoder_cfg.get('embed_dim', 128),
            n_heads=encoder_cfg.get('n_heads', 8),
            n_layers=encoder_cfg.get('n_layers', 3),
            lr=train_cfg.get('lr', 3e-4),
            gamma=train_cfg.get('gamma', 0.99),
            entropy_coef=train_cfg.get('entropy_coefficient', 0.01),
            value_coef=train_cfg.get('value_coefficient', 0.5),
            enc_type=encoder_cfg.get('type', 'gat'),
            n_episodes=self.framework.config.get('problem.episode.n_episodes', 800),
            device=str(self.framework.device)
        )
        
        print(f"✓ Agent initialized with {sum(p.numel() for p in agent.parameters())} parameters")
        return agent
    
    def complete_model(self):
        """Build complete model (agent only, as it contains encoder+decoder)."""
        return self.agent()


# Convenience functions
def read_yaml(yaml_path: str) -> Dict:
    """Read YAML config file."""
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Make main classes easily importable
__all__.extend(['read_yaml'])

