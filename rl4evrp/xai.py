"""
Explainable AI (XAI) utilities for RL4EVRP.
Includes attention visualization, feature importance, and counterfactual analysis.

This module is extensible and designed to work seamlessly with the framework.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class AttentionTracer:
    """Trace and visualize attention patterns during episode execution."""
    
    def __init__(self):
        """Initialize attention tracer."""
        self.traces = []
    
    def add_trace(self, trace: Dict):
        """Add a step trace with attention information."""
        self.traces.append(trace)
    
    def get_traces(self) -> List[Dict]:
        """Get all collected traces."""
        return self.traces
    
    def visualize_attention(self, step: int):
        """Visualize attention at specific step (extensible)."""
        if step < len(self.traces):
            trace = self.traces[step]
            # Implement attention visualization
            pass


class CounterfactualAnalyzer:
    """Analyze agent behavior under state perturbations."""
    
    @staticmethod
    def perturb_battery(obs: dict, factor: float) -> dict:
        """
        Create counterfactual observation with perturbed battery state.
        
        Args:
            obs: Original observation
            factor: Multiplication factor (0.5 = half battery, 1.5 = extra battery)
        
        Returns:
            Perturbed observation
        """
        obs_cf = dict(obs)
        obs_cf['battery_norm'] = float(np.clip(obs['battery_norm'] * factor, 0, 1))
        return obs_cf
    
    @staticmethod
    def perturb_cargo(obs: dict, factor: float) -> dict:
        """
        Create counterfactual observation with perturbed cargo state.
        
        Args:
            obs: Original observation
            factor: Multiplication factor
        
        Returns:
            Perturbed observation
        """
        obs_cf = dict(obs)
        obs_cf['cargo_norm'] = float(np.clip(obs['cargo_norm'] * factor, 0, 1))
        return obs_cf
    
    @staticmethod
    def analyze_sensitivity(agent, obs: dict, perturbation_factors: List[float]) -> Dict:
        """
        Analyze how agent decisions change with state perturbations.
        
        Args:
            agent: A2CAgent instance
            obs: Original observation
            perturbation_factors: Factors to test [0.5, 0.75, 1.0, 1.25, 1.5]
        
        Returns:
            Dictionary with sensitivity analysis results
        """
        results = {
            'battery_perturbations': {},
            'cargo_perturbations': {}
        }
        
        # Get original action
        with torch.no_grad():
            original_action, _, _, _ = agent.select_action(obs, greedy=True)
        results['original_action'] = original_action
        
        # Perturb battery
        for factor in perturbation_factors:
            obs_cf = CounterfactualAnalyzer.perturb_battery(obs, factor)
            with torch.no_grad():
                action, _, _, _ = agent.select_action(obs_cf, greedy=True)
            results['battery_perturbations'][factor] = int(action)
        
        # Perturb cargo
        for factor in perturbation_factors:
            obs_cf = CounterfactualAnalyzer.perturb_cargo(obs, factor)
            with torch.no_grad():
                action, _, _, _ = agent.select_action(obs_cf, greedy=True)
            results['cargo_perturbations'][factor] = int(action)
        
        return results


class FeatureImportance:
    """Compute feature importance through various methods."""
    
    @staticmethod
    def logit_ablation(agent, obs: dict, node_idx: int, feature_idx: int) -> float:
        """
        Ablation-based importance: measure logit change when feature is zeroed.
        
        Args:
            agent: A2CAgent instance
            obs: Observation dict
            node_idx: Node to analyze
            feature_idx: Feature dimension to ablate
        
        Returns:
            Importance score (logit delta)
        """
        # Get baseline logits
        with torch.no_grad():
            baseline_logits, _, _ = agent._forward(obs)
        baseline_logit = baseline_logits[0, node_idx].item()
        
        # Create ablated observation
        obs_ablated = dict(obs)
        feats = obs_ablated['node_features'].clone()
        feats[node_idx, feature_idx] = 0.0
        obs_ablated['node_features'] = feats
        
        # Get ablated logits
        with torch.no_grad():
            ablated_logits, _, _ = agent._forward(obs_ablated)
        ablated_logit = ablated_logits[0, node_idx].item()
        
        # Return importance as logit change
        return abs(baseline_logit - ablated_logit)


# Integration helpers
def collect_traces_during_episode(agent, inst: dict, device: str = 'cpu') -> List[Dict]:
    """Collect attention traces during an episode (for XAI analysis)."""
    from rl4evrp.utils import run_episode
    
    _, _, _, _, _, traces, _ = run_episode(
        agent,
        inst,
        device=device,
        greedy=True,
        collect_traces=True
    )
    
    return traces


def analyze_decision_path(traces: List[Dict], node_idx: int) -> Dict:
    """Analyze how often and when a specific node is visited."""
    analysis = {
        'node_idx': node_idx,
        'visits': [],
        'avg_prob': 0.0,
        'max_prob': 0.0
    }
    
    for trace in traces:
        if trace['to_node'] == node_idx:
            analysis['visits'].append(trace['step'])
            analysis['avg_prob'] += trace['action_prob']
            analysis['max_prob'] = max(analysis['max_prob'], trace['action_prob'])
    
    if analysis['visits']:
        analysis['avg_prob'] /= len(analysis['visits'])
    
    return analysis
