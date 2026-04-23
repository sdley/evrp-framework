import pytest
from pathlib import Path

from rl4evrp.config.config import Config, get_config


CONFIG_DIR = Path(__file__).parent.parent / "src" / "rl4evrp" / "config"


@pytest.fixture(autouse=True)
def reset_global_config():
    """Each test gets a fresh Config singleton."""
    import rl4evrp.config.config as cfg_mod
    cfg_mod._global_config = None
    yield
    cfg_mod._global_config = None


class TestConfig:
    def test_loads_yaml_files(self):
        cfg = Config(CONFIG_DIR)
        assert "problem" in cfg.get_all()
        assert "model" in cfg.get_all()
        assert "env" in cfg.get_all()

    def test_dot_notation_get(self):
        cfg = Config(CONFIG_DIR)
        n = cfg.get("problem.problem.n_customers")
        assert isinstance(n, int)
        assert n > 0

    def test_default_on_missing_key(self):
        cfg = Config(CONFIG_DIR)
        val = cfg.get("nonexistent.key", default=42)
        assert val == 42

    def test_get_section_returns_dict(self):
        cfg = Config(CONFIG_DIR)
        section = cfg.get_section("model")
        assert isinstance(section, dict)

    def test_get_section_missing_returns_empty(self):
        cfg = Config(CONFIG_DIR)
        assert cfg.get_section("does_not_exist") == {}

    def test_get_all_returns_dict(self):
        cfg = Config(CONFIG_DIR)
        assert isinstance(cfg.get_all(), dict)

    def test_repr_contains_sections(self):
        cfg = Config(CONFIG_DIR)
        r = repr(cfg)
        assert "Config" in r
        assert "problem" in r

    def test_print_config_runs(self, capsys):
        cfg = Config(CONFIG_DIR)
        cfg.print_config()
        out = capsys.readouterr().out
        assert "PROBLEM" in out or "MODEL" in out


class TestGetConfig:
    def test_returns_config_instance(self):
        cfg = get_config(CONFIG_DIR)
        assert isinstance(cfg, Config)

    def test_singleton_same_object(self):
        a = get_config(CONFIG_DIR)
        b = get_config(CONFIG_DIR)
        assert a is b

    def test_singleton_reset_creates_new(self):
        import rl4evrp.config.config as cfg_mod
        a = get_config(CONFIG_DIR)
        cfg_mod._global_config = None
        b = get_config(CONFIG_DIR)
        assert a is not b
