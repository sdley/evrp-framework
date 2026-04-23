import pytest
from rl4evrp.environment import generate_instance, EVRPEnv
from rl4evrp.agents import A2CAgent


@pytest.fixture
def small_inst():
    """Tiny deterministic instance (8 customers) for fast tests."""
    return generate_instance(n_customers=8, seed=42)


@pytest.fixture
def env(small_inst):
    return EVRPEnv(small_inst)


@pytest.fixture
def obs(env):
    return env.reset()


@pytest.fixture
def agent():
    return A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")
