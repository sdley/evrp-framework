"""
Configuration management for RL4EVRP framework.
Handles loading YAML configs, environment variables, and provides unified access.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Unified configuration management for RL4EVRP."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration from YAML files and environment variables.
        
        Args:
            config_dir: Directory containing YAML config files.
                       Defaults to rl4evrp/config/ relative to this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        self._configs = {}
        
        # Load .env file
        env_file = self.config_dir.parent.parent / ".env"
        if env_file.exists():
            load_dotenv(str(env_file))
        
        # Load all YAML files
        self._load_all_configs()
        
    def _load_all_configs(self):
        """Load all YAML config files from config directory."""
        for yaml_file in sorted(self.config_dir.glob("*.yaml")):
            config_name = yaml_file.stem
            with open(yaml_file, 'r', encoding='utf-8') as f:
                self._configs[config_name] = yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Examples:
            config.get('problem.n_customers')  # Returns 15
            config.get('training.lr')          # Returns 3.0e-4
            config.get('nonexistent', 42)      # Returns 42
        """
        keys = key.split('.')
        value = self._configs
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        # Handle environment variable substitution
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get an entire config section."""
        return self._configs.get(section, {})
    
    def get_all(self) -> Dict:
        """Get all configurations."""
        return self._configs
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(sections={list(self._configs.keys())})"
    
    def print_config(self):
        """Pretty print all configurations."""
        for section, config in self._configs.items():
            print(f"\n{'='*60}")
            print(f"[{section.upper()}]")
            print(f"{'='*60}")
            self._print_dict(config)
    
    @staticmethod
    def _print_dict(d: Dict, indent: int = 0):
        """Recursively print dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                Config._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        Config._print_dict(item, indent + 1)
                    else:
                        print("  " * (indent + 1) + f"- {item}")
            else:
                print("  " * indent + f"{key}: {value}")


# Singleton instance for convenience
_global_config = None


def get_config(config_dir: Optional[Path] = None) -> Config:
    """Get or create global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_dir)
    return _global_config
