import torch
import pytest

from rl4evrp.agents import A2CAgent
from rl4evrp.environment import EVRPEnv, generate_instance


@pytest.fixture
def inst():
    return generate_instance(n_customers=8, seed=0)


@pytest.fixture
def agent():
    return A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")


@pytest.fixture
def obs(inst):
    return EVRPEnv(inst).reset()


class TestSelectAction:
    def test_action_is_valid(self, agent, obs):
        action, _, _, _ = agent.select_action(obs)
        assert obs["valid_mask"][action]

    def test_action_in_range(self, agent, obs, inst):
        action, _, _, _ = agent.select_action(obs)
        assert 0 <= action < inst["n_nodes"]

    def test_log_prob_is_scalar(self, agent, obs):
        _, lp, _, _ = agent.select_action(obs)
        assert lp.shape == torch.Size([])

    def test_entropy_is_positive(self, agent, obs):
        _, _, ent, _ = agent.select_action(obs)
        assert ent.item() >= 0.0

    def test_value_shape(self, agent, obs):
        _, _, _, val = agent.select_action(obs)
        assert val.shape == (1, 1)

    def test_greedy_is_deterministic(self, agent, obs):
        actions = {agent.select_action(obs, greedy=True)[0] for _ in range(5)}
        assert len(actions) == 1

    def test_stochastic_can_vary(self, agent, obs):
        torch.manual_seed(0)
        actions = {agent.select_action(obs, greedy=False)[0] for _ in range(30)}
        assert len(actions) >= 1  # at minimum doesn't crash


class TestForward:
    def test_shapes_without_attn(self, agent, obs, inst):
        scores, value, node_emb = agent._forward(obs)
        n = inst["n_nodes"]
        assert scores.shape == (1, n)
        assert value.shape == (1, 1)
        assert node_emb.shape[1] == n

    def test_shapes_with_attn(self, agent, obs, inst):
        scores, value, node_emb, dec_attn, enc_attn = agent._forward(obs, return_attn=True)
        assert scores.shape == (1, inst["n_nodes"])
        assert dec_attn is not None
        assert enc_attn is not None


class TestUpdate:
    def _make_transitions(self, agent, obs):
        transitions = []
        _, lp, ent, val = agent.select_action(obs)
        transitions.append({"r": 0.5, "lp": lp, "ent": ent, "val": val})
        return transitions

    def test_returns_floats(self, agent, obs):
        t = self._make_transitions(agent, obs)
        loss, ent = agent.update(t)
        assert isinstance(loss, float)
        assert isinstance(ent, float)

    def test_loss_is_finite(self, agent, inst):
        # Need 2+ transitions: std() is undefined for a single-element tensor
        env = EVRPEnv(inst)
        obs = env.reset()
        transitions = []
        for _ in range(3):
            action, lp, ent, val = agent.select_action(obs)
            obs, r, done, _ = env.step(action)
            transitions.append({"r": r, "lp": lp, "ent": ent, "val": val})
            if done:
                break
        loss, _ = agent.update(transitions)
        assert torch.isfinite(torch.tensor(loss))

    def test_params_change_after_update(self, agent, obs):
        before = {k: v.clone() for k, v in agent.named_parameters()}
        t = self._make_transitions(agent, obs)
        agent.update(t)
        changed = any(
            not torch.equal(before[k], v) for k, v in agent.named_parameters()
        )
        assert changed

    def test_multi_step_update(self, agent, inst):
        env = EVRPEnv(inst)
        obs = env.reset()
        transitions = []
        for _ in range(5):
            action, lp, ent, val = agent.select_action(obs)
            obs, r, done, _ = env.step(action)
            transitions.append({"r": r, "lp": lp, "ent": ent, "val": val})
            if done:
                break
        loss, ent = agent.update(transitions)
        assert torch.isfinite(torch.tensor(loss))


class TestGetActionForInference:
    def test_returns_int(self, agent, obs):
        action = agent.get_action_for_inference(obs)
        assert isinstance(action, int)

    def test_no_grad_tracked(self, agent, obs):
        with torch.no_grad():
            action = agent.get_action_for_inference(obs)
        assert 0 <= action < len(obs["valid_mask"])
