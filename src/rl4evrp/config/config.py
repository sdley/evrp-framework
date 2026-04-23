import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Unified configuration management for RL4EVRP."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: Directory containing YAML config files.
                        Defaults to rl4evrp/config/ relative to this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}

        env_file = self.config_dir.parent.parent / ".env"
        if env_file.exists():
            load_dotenv(str(env_file))

        self._load_all_configs()

    def _load_all_configs(self):
        for yaml_file in sorted(self.config_dir.glob("*.yaml")):
            config_name = yaml_file.stem
            with open(yaml_file, 'r') as f:
                self._configs[config_name] = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.

        Examples:
            config.get('problem.n_customers')
            config.get('training.lr', 3e-4)
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

        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)

        return value

    def get_section(self, section: str) -> Dict:
        return self._configs.get(section, {})

    def get_all(self) -> Dict:
        return self._configs

    def print_config(self):
        for section, config in self._configs.items():
            print(f"\n{'='*60}\n[{section.upper()}]\n{'='*60}")
            self._print_dict(config)

    @staticmethod
    def _print_dict(d: Dict, indent: int = 0):
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

    def __repr__(self) -> str:
        return f"Config(sections={list(self._configs.keys())})"


_global_config: Optional[Config] = None


def get_config(config_dir: Optional[Path] = None) -> Config:
    """Get or create the global Config singleton."""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_dir)
    return _global_config
