import numpy as np
import torch
from typing import Dict, List


class CounterfactualAnalyzer:
    """Analyze agent decisions under state perturbations."""

    @staticmethod
    def perturb_battery(obs: dict, factor: float) -> dict:
        """Return a copy of obs with battery_norm scaled by factor."""
        obs_cf = dict(obs)
        obs_cf['battery_norm'] = float(np.clip(obs['battery_norm'] * factor, 0, 1))
        return obs_cf

    @staticmethod
    def perturb_cargo(obs: dict, factor: float) -> dict:
        """Return a copy of obs with cargo_norm scaled by factor."""
        obs_cf = dict(obs)
        obs_cf['cargo_norm'] = float(np.clip(obs['cargo_norm'] * factor, 0, 1))
        return obs_cf

    @staticmethod
    def analyze_sensitivity(agent, obs: dict, perturbation_factors: List[float]) -> Dict:
        """
        Measure how the greedy action changes across a range of state perturbations.

        Args:
            agent: A2CAgent instance
            obs: Original observation
            perturbation_factors: Multipliers to test, e.g. [0.5, 0.75, 1.0, 1.25, 1.5]

        Returns:
            Dict with original_action, battery_perturbations, cargo_perturbations
        """
        results: Dict = {'battery_perturbations': {}, 'cargo_perturbations': {}}

        with torch.no_grad():
            original_action, _, _, _ = agent.select_action(obs, greedy=True)
        results['original_action'] = original_action

        for factor in perturbation_factors:
            with torch.no_grad():
                action, _, _, _ = agent.select_action(
                    CounterfactualAnalyzer.perturb_battery(obs, factor), greedy=True
                )
            results['battery_perturbations'][factor] = int(action)

        for factor in perturbation_factors:
            with torch.no_grad():
                action, _, _, _ = agent.select_action(
                    CounterfactualAnalyzer.perturb_cargo(obs, factor), greedy=True
                )
            results['cargo_perturbations'][factor] = int(action)

        return results
