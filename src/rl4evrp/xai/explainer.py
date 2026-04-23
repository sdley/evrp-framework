"""
LLM-powered natural-language explanation of agent decisions via Groq.

Requires the optional `llm` extra:
    pip install "rl4evrp[llm]"
    # or: uv sync --extra llm

and a GROQ_API_KEY environment variable (or a .env file at the project root).
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # Load GROQ_API_KEY from .env if present


_NODE_TYPE = {0: "depot", 1: "customer", 2: "charger"}


def _format_step(trace: Dict) -> str:
    node_type = _NODE_TYPE.get(trace["node_type"], "unknown")
    forced = " [FORCED — no other valid action]" if trace.get("is_forced") else ""
    top3 = ", ".join(
        f"node {n} ({p:.0%})"
        for n, p in zip(trace["top3_nodes"], trace["top3_probs"])
    )
    return (
        f"Step {trace['step']}: {trace['from_node']} → {trace['to_node']} "
        f"({node_type}){forced}\n"
        f"  battery={trace['battery_norm']:.0%}  cargo={trace['cargo_norm']:.0%}  "
        f"chosen prob={trace['action_prob']:.0%}\n"
        f"  top alternatives: {top3}"
    )


def _build_prompt(traces: List[Dict], inst: Dict, question: Optional[str]) -> str:
    n_customers = int((inst["node_types"] == 1).sum())
    n_chargers  = int((inst["node_types"] == 2).sum())
    steps_text  = "\n".join(_format_step(t) for t in traces)

    default_q = (
        "Explain the agent's overall strategy and highlight the two or three "
        "most interesting decision points."
    )

    return textwrap.dedent(f"""
        You are an expert in reinforcement learning and vehicle routing.
        An A2C agent just completed an Electric Vehicle Routing Problem episode.

        INSTANCE SUMMARY
        ----------------
        Nodes: {inst['n_nodes']}  (1 depot, {n_customers} customers, {n_chargers} chargers)
        Battery capacity: {inst['battery_cap']}
        Cargo capacity:   {inst['cargo_cap']}

        DECISION TRACE (one line per step)
        ------------------------------------
        {steps_text}

        QUESTION
        --------
        {question or default_q}

        Answer in plain English, 3-5 sentences. Be specific about node indices,
        battery/cargo levels, and trade-offs when they matter.
    """).strip()


class GroqExplainer:
    """
    Wraps the Groq API to produce natural-language explanations of EVRP episodes.

    Usage
    -----
    >>> from rl4evrp.xai import GroqExplainer, collect_traces_during_episode
    >>> explainer = GroqExplainer()                   # reads GROQ_API_KEY from env
    >>> traces = collect_traces_during_episode(agent, inst)
    >>> print(explainer.explain_episode(traces, inst))
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq is not installed. Run: pip install 'rl4evrp[llm]'"
            ) from None
        
        if api_key is None and "GROQ_API_KEY" not in os.environ:
            raise ValueError(
                "GROQ_API_KEY not found in environment. Set it or create a .env file."
            )
        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def explain_episode(
        self,
        traces: List[Dict],
        inst: Dict,
        question: Optional[str] = None,
    ) -> str:
        """
        Generate a natural-language explanation of a full episode.

        Args:
            traces:   Output of collect_traces_during_episode()
            inst:     The instance dict the episode ran on
            question: Optional specific question to answer. Defaults to a
                      general strategy + highlight summary.

        Returns:
            Plain-text explanation from the LLM.
        """
        prompt = _build_prompt(traces, inst, question)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def explain_step(self, trace: Dict, inst: Dict) -> str:
        """
        Explain a single decision step.

        Args:
            trace: One element from a traces list
            inst:  The instance dict

        Returns:
            One or two sentence explanation of the step.
        """
        node_type = _NODE_TYPE.get(trace["node_type"], "unknown")
        prompt = textwrap.dedent(f"""
            An RL agent solving an Electric Vehicle Routing Problem took this action:
            - Moved from node {trace['from_node']} to node {trace['to_node']} ({node_type})
            - Battery before move: {trace['battery_norm']:.0%} of {inst['battery_cap']} units
            - Cargo before move:   {trace['cargo_norm']:.0%} of {inst['cargo_cap']} units
            - Chosen with probability {trace['action_prob']:.0%}
            - Distance to action: {trace['dist_to_action']:.3f}
            - Distance from action back to depot: {trace['dist_to_depot']:.3f}
            - {'This was the only valid action.' if trace.get('is_forced') else ''}

            In one or two sentences, explain why this move makes sense given
            the battery and cargo state.
        """).strip()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()
