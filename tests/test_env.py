import numpy as np
import pytest

from rl4evrp.environment import EVRPEnv, generate_instance


@pytest.fixture
def inst():
    return generate_instance(n_customers=8, seed=0)


@pytest.fixture
def env(inst):
    return EVRPEnv(inst)


class TestReset:
    def test_starts_at_depot(self, env):
        obs = env.reset()
        assert obs["current_node"] == 0

    def test_full_battery(self, env):
        obs = env.reset()
        assert obs["battery_norm"] == pytest.approx(1.0)

    def test_full_cargo(self, env):
        obs = env.reset()
        assert obs["cargo_norm"] == pytest.approx(1.0)

    def test_obs_keys(self, env):
        obs = env.reset()
        assert {"node_features", "current_node", "battery_norm",
                "cargo_norm", "visited_mask", "valid_mask"}.issubset(obs)

    def test_not_done(self, env):
        env.reset()
        assert not env.done

    def test_idempotent(self, env):
        env.reset()
        env.step(1)
        obs = env.reset()
        assert obs["current_node"] == 0
        assert not env.done


class TestValidMask:
    def test_depot_valid_when_not_at_depot(self, env, inst):
        # Depot is excluded only when the agent is currently there.
        # Move to a customer first, then verify depot is reachable.
        env.reset()
        customers = [j for j in range(1, inst["n_nodes"]) if inst["node_types"][j] == 1]
        for cust in customers:
            if env._valid_mask()[cust]:
                obs, _, _, _ = env.step(cust)
                assert obs["valid_mask"][0]
                return
        pytest.skip("no reachable customer to move away from depot")

    def test_no_self_loop(self, env):
        obs = env.reset()
        assert not obs["valid_mask"][0] or env.cur != 0
        # current node is never in its own valid mask
        assert not obs["valid_mask"][obs["current_node"]]

    def test_at_least_one_valid(self, env):
        obs = env.reset()
        assert obs["valid_mask"].any()

    def test_mask_length(self, env, inst):
        obs = env.reset()
        assert len(obs["valid_mask"]) == inst["n_nodes"]


class TestStep:
    def test_returns_four_tuple(self, env):
        env.reset()
        valid = np.where(env._valid_mask())[0]
        result = env.step(int(valid[0]))
        assert len(result) == 4

    def test_reward_is_numeric(self, env):
        env.reset()
        valid = np.where(env._valid_mask())[0]
        _, reward, _, _ = env.step(int(valid[0]))
        assert isinstance(reward, (float, np.floating))

    def test_route_grows(self, env):
        env.reset()
        assert len(env.route) == 1
        valid = np.where(env._valid_mask())[0]
        env.step(int(valid[0]))
        assert len(env.route) == 2

    def test_customer_marked_visited(self, env, inst):
        env.reset()
        customers = np.where(inst["node_types"] == 1)[0]
        # Force a visit to first customer if it is valid
        for cust in customers:
            mask = env._valid_mask()
            if mask[cust]:
                env.step(int(cust))
                assert env.visited[cust]
                break

    def test_charger_restores_battery(self, env, inst):
        env.reset()
        chargers = np.where(inst["node_types"] == 2)[0]
        if len(chargers) == 0:
            pytest.skip("no chargers in this instance")
        env.battery = 10.0
        mask = env._valid_mask()
        for ch in chargers:
            if mask[ch]:
                env.step(int(ch))
                assert env.battery == pytest.approx(inst["battery_cap"])
                break

    def test_depot_restores_cargo(self, env):
        obs = env.reset()
        env.cargo = 0.0
        env.step(0)
        assert env.cargo == pytest.approx(env.cargo_cap)

    def test_episode_terminates(self, env, inst):
        env.reset()
        for _ in range(4 * inst["n_nodes"] + 10):
            if env.done:
                break
            mask = env._valid_mask()
            action = int(np.where(mask)[0][0])
            env.step(action)
        assert env.done


class TestDistanceMatrix:
    def test_symmetric(self, env):
        env.reset()
        D = env.D
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_diagonal_zero(self, env):
        env.reset()
        np.testing.assert_allclose(np.diag(env.D), 0.0, atol=1e-6)
