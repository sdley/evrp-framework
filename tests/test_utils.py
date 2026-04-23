import pytest
import numpy as np

from rl4evrp.environment import generate_instance
from rl4evrp.agents import A2CAgent
from rl4evrp.utils.pool import OnTheFlyInstancePool, _resolve_training_instance
from rl4evrp.utils.training import run_episode, train_agent, evaluate_agent


@pytest.fixture
def agent():
    return A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")


@pytest.fixture
def inst():
    return generate_instance(n_customers=8, seed=0)


class TestOnTheFlyInstancePool:
    def _make_pool(self, size=5):
        def gen(seed):
            return generate_instance(n_customers=8, seed=seed)
        return OnTheFlyInstancePool(gen, size=size, seed_offset=100)

    def test_len(self):
        pool = self._make_pool(5)
        assert len(pool) == 5

    def test_getitem_returns_dict(self):
        pool = self._make_pool()
        assert isinstance(pool[0], dict)

    def test_index_wraps_around(self):
        pool = self._make_pool(3)
        a = pool[0]
        b = pool[3]
        np.testing.assert_array_equal(a["coords"], b["coords"])

    def test_different_indices_differ(self):
        pool = self._make_pool(5)
        assert not np.array_equal(pool[0]["coords"], pool[1]["coords"])

    def test_zero_size_raises(self):
        with pytest.raises(ValueError):
            OnTheFlyInstancePool(lambda seed: {}, size=0)

    def test_is_independent_copy(self):
        pool = self._make_pool()
        a = pool[0]
        a["coords"][0, 0] = 999.0
        b = pool[0]
        assert b["coords"][0, 0] != 999.0


class TestResolveTrainingInstance:
    def test_from_sequence(self):
        instances = [generate_instance(seed=i) for i in range(3)]
        inst = _resolve_training_instance(instances, episode=1)
        np.testing.assert_array_equal(inst["coords"], instances[1]["coords"])

    def test_wraps_around_sequence(self):
        instances = [generate_instance(seed=i) for i in range(3)]
        inst = _resolve_training_instance(instances, episode=4)  # 4 % 3 == 1
        np.testing.assert_array_equal(inst["coords"], instances[1]["coords"])

    def test_from_callable(self):
        def provider(ep):
            return generate_instance(seed=ep)
        inst = _resolve_training_instance(provider, episode=7)
        expected = generate_instance(seed=7)
        np.testing.assert_array_equal(inst["coords"], expected["coords"])

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            _resolve_training_instance([], episode=0)

    def test_returns_new_dict_object(self):
        # _resolve_training_instance returns a shallow dict copy — the dict
        # is a new object but numpy array values are still shared references.
        instances = [generate_instance(seed=0)]
        inst = _resolve_training_instance(instances, episode=0)
        assert inst is not instances[0]


class TestRunEpisode:
    def test_returns_seven_elements(self, agent, inst):
        result = run_episode(agent, inst, greedy=True)
        assert len(result) == 7

    def test_reward_is_numeric(self, agent, inst):
        total_reward, *_ = run_episode(agent, inst, greedy=True)
        assert isinstance(total_reward, (float, np.floating))

    def test_route_starts_at_depot(self, agent, inst):
        _, route, *_ = run_episode(agent, inst, greedy=True)
        assert route[0] == 0

    def test_distance_positive(self, agent, inst):
        _, _, dist, *_ = run_episode(agent, inst, greedy=True)
        assert dist > 0.0

    def test_traces_none_by_default(self, agent, inst):
        *_, traces, _ = run_episode(agent, inst, greedy=True)
        assert traces is None

    def test_traces_collected_when_requested(self, agent, inst):
        *_, traces, _ = run_episode(agent, inst, greedy=True, collect_traces=True)
        assert traces is not None
        assert len(traces) > 0

    def test_trace_has_expected_keys(self, agent, inst):
        *_, traces, _ = run_episode(agent, inst, greedy=True, collect_traces=True)
        keys = {"step", "from_node", "to_node", "battery_norm", "cargo_norm",
                "dec_attn", "enc_attn", "action_prob"}
        assert keys.issubset(traces[0])

    def test_battery_perturbation(self, agent, inst):
        result = run_episode(agent, inst, greedy=True, bat_perturb=0.5)
        assert len(result) == 7


class TestTrainAgent:
    """End-to-end integration: train_agent runs without error and returns sane values."""

    @pytest.fixture
    def small_agent(self):
        return A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")

    @pytest.fixture
    def small_pool(self):
        return OnTheFlyInstancePool(
            lambda seed: generate_instance(n_customers=6, seed=seed),
            size=4,
        )

    def test_returns_dict_with_expected_keys(self, small_agent, small_pool):
        result = train_agent(small_agent, small_pool, n_episodes=5, device="cpu")
        assert {"train_rewards", "losses", "eval_rewards"}.issubset(result)

    def test_train_rewards_length(self, small_agent, small_pool):
        result = train_agent(small_agent, small_pool, n_episodes=5, device="cpu")
        assert len(result["train_rewards"]) == 5

    def test_losses_finite(self, small_agent, small_pool):
        result = train_agent(small_agent, small_pool, n_episodes=5, device="cpu")
        assert all(np.isfinite(l) for l in result["losses"])

    def test_weights_change_after_training(self, small_pool):
        agent = A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")
        params_before = [p.clone().detach() for p in agent.parameters()]
        train_agent(agent, small_pool, n_episodes=5, device="cpu")
        params_after = list(agent.parameters())
        assert any(
            not p_before.equal(p_after)
            for p_before, p_after in zip(params_before, params_after)
        )

    def test_eval_rewards_populated_when_instances_given(self, small_agent, small_pool):
        eval_instances = [generate_instance(n_customers=6, seed=9000 + i) for i in range(3)]
        result = train_agent(
            small_agent, small_pool,
            n_episodes=5, device="cpu",
            eval_instances=eval_instances,
        )
        assert len(result["eval_rewards"]) > 0

    def test_max_steps_limit_respected(self):
        """EVRPEnv with max_steps=5 should terminate episodes early."""
        from rl4evrp.environment import EVRPEnv
        inst = generate_instance(n_customers=15, seed=0)
        env = EVRPEnv(inst, max_steps=5)
        env.reset()
        steps = 0
        done = False
        while not done:
            valid = env._valid_mask()
            action = int(np.where(valid)[0][0])
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps <= 5


class TestEvaluateAgent:
    def test_returns_dict(self, agent):
        instances = [generate_instance(seed=i) for i in range(3)]
        stats = evaluate_agent(agent, instances, greedy=True)
        assert isinstance(stats, dict)

    def test_has_expected_keys(self, agent):
        instances = [generate_instance(seed=i) for i in range(3)]
        stats = evaluate_agent(agent, instances, greedy=True)
        assert {"mean_reward", "std_reward", "mean_distance",
                "std_distance", "rewards", "distances", "routes"}.issubset(stats)

    def test_n_eval_limits_instances(self, agent):
        instances = [generate_instance(seed=i) for i in range(5)]
        stats = evaluate_agent(agent, instances, n_eval=2)
        assert len(stats["rewards"]) == 2

    def test_rewards_are_finite(self, agent):
        instances = [generate_instance(seed=i) for i in range(3)]
        stats = evaluate_agent(agent, instances)
        assert all(np.isfinite(r) for r in stats["rewards"])
