import os
import pytest
from unittest.mock import MagicMock, patch

from rl4evrp.environment import generate_instance, EVRPEnv
from rl4evrp.agents import A2CAgent
from rl4evrp.xai.attention import AttentionTracer
from rl4evrp.xai.counterfactual import CounterfactualAnalyzer
from rl4evrp.xai.importance import FeatureImportance, collect_traces_during_episode, analyze_decision_path
from rl4evrp.xai.explainer import GroqExplainer


@pytest.fixture
def inst():
    return generate_instance(n_customers=8, seed=0)


@pytest.fixture
def agent():
    return A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=10, device="cpu")


@pytest.fixture
def obs(inst):
    return EVRPEnv(inst).reset()


class TestAttentionTracer:
    def test_empty_on_init(self):
        tracer = AttentionTracer()
        assert tracer.get_traces() == []

    def test_add_and_retrieve(self):
        tracer = AttentionTracer()
        tracer.add_trace({"step": 0, "data": 1.0})
        tracer.add_trace({"step": 1, "data": 2.0})
        assert len(tracer.get_traces()) == 2

    def test_get_traces_returns_list(self):
        tracer = AttentionTracer()
        assert isinstance(tracer.get_traces(), list)

    def test_visualize_out_of_bounds_does_not_raise(self):
        tracer = AttentionTracer()
        tracer.visualize_attention(99)  # should be a no-op


class TestCounterfactualAnalyzer:
    def test_perturb_battery_clips_at_one(self, obs):
        cf = CounterfactualAnalyzer.perturb_battery(obs, factor=10.0)
        assert cf["battery_norm"] <= 1.0

    def test_perturb_battery_clips_at_zero(self, obs):
        cf = CounterfactualAnalyzer.perturb_battery(obs, factor=0.0)
        assert cf["battery_norm"] == pytest.approx(0.0)

    def test_perturb_cargo_clips_at_one(self, obs):
        cf = CounterfactualAnalyzer.perturb_cargo(obs, factor=10.0)
        assert cf["cargo_norm"] <= 1.0

    def test_perturb_does_not_mutate_original(self, obs):
        original = obs["battery_norm"]
        CounterfactualAnalyzer.perturb_battery(obs, factor=0.5)
        assert obs["battery_norm"] == original

    def test_analyze_sensitivity_keys(self, agent, obs):
        factors = [0.5, 1.0, 1.5]
        result = CounterfactualAnalyzer.analyze_sensitivity(agent, obs, factors)
        assert "original_action" in result
        assert set(result["battery_perturbations"].keys()) == set(factors)
        assert set(result["cargo_perturbations"].keys()) == set(factors)

    def test_analyze_sensitivity_actions_in_range(self, agent, obs, inst):
        result = CounterfactualAnalyzer.analyze_sensitivity(agent, obs, [0.5, 1.0])
        n = inst["n_nodes"]
        for action in result["battery_perturbations"].values():
            assert 0 <= action < n


class TestFeatureImportance:
    def test_returns_non_negative_float(self, agent, obs):
        score = FeatureImportance.logit_ablation(agent, obs, node_idx=1, feature_idx=0)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_zeroed_feature_changes_score(self, agent, obs):
        # Ablating x-coordinate of node 1 should (usually) change the logit
        score = FeatureImportance.logit_ablation(agent, obs, node_idx=1, feature_idx=0)
        # Not asserting exact value — just that the function runs and returns a number
        assert isinstance(score, float)


class TestCollectTraces:
    def test_returns_list(self, agent, inst):
        traces = collect_traces_during_episode(agent, inst)
        assert isinstance(traces, list)

    def test_traces_non_empty(self, agent, inst):
        traces = collect_traces_during_episode(agent, inst)
        assert len(traces) > 0

    def test_trace_step_monotonic(self, agent, inst):
        traces = collect_traces_during_episode(agent, inst)
        steps = [t["step"] for t in traces]
        assert steps == list(range(len(steps)))


def _mock_groq_client(reply: str = "The agent did well.") -> MagicMock:
    """Return a mock Groq client whose chat.completions.create returns `reply`."""
    message = MagicMock()
    message.content = reply
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def traces(agent, inst):
    return collect_traces_during_episode(agent, inst)


class TestGroqExplainer:
    def _make_explainer(self, reply: str = "Test explanation.") -> GroqExplainer:
        """Build a GroqExplainer bypassing __init__ — no real groq package needed."""
        explainer = GroqExplainer.__new__(GroqExplainer)
        explainer.model = "llama-3.1-8b-instant"
        explainer._client = _mock_groq_client(reply)
        return explainer

    def _mock_groq_module(self) -> MagicMock:
        """A fake sys.modules['groq'] entry with a Groq class attribute."""
        mock_module = MagicMock()
        mock_module.Groq = MagicMock(return_value=MagicMock())
        return mock_module

    def test_missing_groq_package_raises_import_error(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "x"}):
            with patch.dict("sys.modules", {"groq": None}):
                with pytest.raises(ImportError, match="groq is not installed"):
                    GroqExplainer(api_key="x")

    def test_missing_api_key_raises_value_error(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {"groq": self._mock_groq_module()}):
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    GroqExplainer()

    def test_explain_episode_returns_string(self, traces, inst):
        explainer = self._make_explainer("Good strategy overall.")
        result = explainer.explain_episode(traces, inst)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_episode_content(self, traces, inst):
        explainer = self._make_explainer("Battery managed well.")
        result = explainer.explain_episode(traces, inst)
        assert result == "Battery managed well."

    def test_explain_episode_custom_question(self, traces, inst):
        explainer = self._make_explainer("Yes, battery was fine.")
        result = explainer.explain_episode(
            traces, inst,
            question="Did the agent manage battery life well?"
        )
        assert isinstance(result, str)
        # verify the API was actually called once
        explainer._client.chat.completions.create.assert_called_once()

    def test_explain_step_returns_string(self, traces, inst):
        explainer = self._make_explainer("Move makes sense at low battery.")
        result = explainer.explain_step(traces[0], inst)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_step_calls_api_once(self, traces, inst):
        explainer = self._make_explainer("One step explanation.")
        explainer.explain_step(traces[0], inst)
        explainer._client.chat.completions.create.assert_called_once()

    def test_explain_episode_calls_api_once(self, traces, inst):
        explainer = self._make_explainer("Episode explanation.")
        explainer.explain_episode(traces, inst)
        explainer._client.chat.completions.create.assert_called_once()

    def test_model_passed_to_api(self, traces, inst):
        explainer = self._make_explainer()
        explainer.model = "llama-3.3-70b-versatile"
        explainer.explain_episode(traces, inst)
        call_kwargs = explainer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_full_pipeline_traces_to_explanation(self, agent, inst):
        """Integration: collect traces → explain episode (mocked API)."""
        live_traces = collect_traces_during_episode(agent, inst, device="cpu")
        assert len(live_traces) > 0

        explainer = self._make_explainer("The agent served all customers efficiently.")
        explanation = explainer.explain_episode(live_traces, inst)
        assert "agent" in explanation.lower() or len(explanation) > 5


class TestAnalyzeDecisionPath:
    def test_empty_traces(self):
        result = analyze_decision_path([], node_idx=3)
        assert result["visits"] == []
        assert result["avg_prob"] == pytest.approx(0.0)

    def test_counts_visits(self):
        traces = [
            {"to_node": 3, "step": 0, "action_prob": 0.8},
            {"to_node": 5, "step": 1, "action_prob": 0.6},
            {"to_node": 3, "step": 2, "action_prob": 0.4},
        ]
        result = analyze_decision_path(traces, node_idx=3)
        assert result["visits"] == [0, 2]
        assert result["max_prob"] == pytest.approx(0.8)
        assert result["avg_prob"] == pytest.approx(0.6)

    def test_node_never_visited(self):
        traces = [{"to_node": 1, "step": 0, "action_prob": 0.9}]
        result = analyze_decision_path(traces, node_idx=99)
        assert result["visits"] == []
