import torch
from typing import Dict, List


class FeatureImportance:
    """Compute per-feature importance via logit ablation."""

    @staticmethod
    def logit_ablation(agent, obs: dict, node_idx: int, feature_idx: int) -> float:
        """
        Measure how much the logit for node_idx changes when feature_idx is zeroed.

        Args:
            agent: A2CAgent instance
            obs: Observation dict
            node_idx: Node whose logit to track
            feature_idx: Feature dimension to ablate

        Returns:
            Absolute logit delta (importance score)
        """
        with torch.no_grad():
            baseline_logits, _, _ = agent._forward(obs)
        baseline = baseline_logits[0, node_idx].item()

        obs_ablated = dict(obs)
        feats = obs_ablated['node_features'].clone()
        feats[node_idx, feature_idx] = 0.0
        obs_ablated['node_features'] = feats

        with torch.no_grad():
            ablated_logits, _, _ = agent._forward(obs_ablated)
        ablated = ablated_logits[0, node_idx].item()

        return abs(baseline - ablated)


def collect_traces_during_episode(agent, inst: dict, device: str = 'cpu') -> List[Dict]:
    """Run one greedy episode and return per-step attention traces."""
    from rl4evrp.utils import run_episode

    _, _, _, _, _, traces, _ = run_episode(
        agent, inst, device=device, greedy=True, collect_traces=True
    )
    return traces


def analyze_decision_path(traces: List[Dict], node_idx: int) -> Dict:
    """Summarize how often and when node_idx was chosen across an episode."""
    analysis: Dict = {
        'node_idx': node_idx,
        'visits': [],
        'avg_prob': 0.0,
        'max_prob': 0.0,
    }

    for trace in traces:
        if trace['to_node'] == node_idx:
            analysis['visits'].append(trace['step'])
            analysis['avg_prob'] += trace['action_prob']
            analysis['max_prob'] = max(analysis['max_prob'], trace['action_prob'])

    if analysis['visits']:
        analysis['avg_prob'] /= len(analysis['visits'])

    return analysis
