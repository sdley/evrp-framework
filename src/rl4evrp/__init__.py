"""
rl4evrp — Deep Reinforcement Learning for the Electric Vehicle Routing Problem.

Quickstart
----------
>>> from rl4evrp import generate_instance, A2CAgent
>>> from rl4evrp.utils import train_agent, evaluate_agent, OnTheFlyInstancePool

Public subpackages
------------------
rl4evrp.environment   generate_instance | build_node_features | make_dataset | EVRPEnv
rl4evrp.models        EVRPEncoder | EVRPDecoder | MultiHeadAttention
rl4evrp.agents        A2CAgent
rl4evrp.utils         run_episode | train_agent | evaluate_agent | OnTheFlyInstancePool
rl4evrp.config        Config | get_config
rl4evrp.xai           AttentionTracer | CounterfactualAnalyzer | FeatureImportance
                      collect_traces_during_episode | analyze_decision_path
                      GroqExplainer  (requires pip install "rl4evrp[llm]")


Authors
-------
- Dimeth Noucier < 
- Souleyman Diallo <
- Imen Habibi <
- Jérémie Mabiala <
- Mame Diarra <
- Elie Mulamba <

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
----------------
This project was developed as part of the 2024 IDEATHON organized by Deep Learning Indaba 2024. 
We thank all the Deep Learning Indaba organizers and mentors for their support and guidance throughout the development of this framework.
"""

from .config import Config, get_config
from .environment import EVRPEnv, generate_instance, build_node_features, make_dataset
from .models import EVRPEncoder, EVRPDecoder, MultiHeadAttention
from .agents import A2CAgent
from .utils import run_episode, train_agent, evaluate_agent, OnTheFlyInstancePool
from . import xai

__version__ = "0.1.0"

__all__ = [
    # config
    "Config",
    "get_config",
    # environment
    "EVRPEnv",
    "generate_instance",
    "build_node_features",
    "make_dataset",
    # models
    "EVRPEncoder",
    "EVRPDecoder",
    "MultiHeadAttention",
    # agents
    "A2CAgent",
    # utils
    "run_episode",
    "train_agent",
    "evaluate_agent",
    "OnTheFlyInstancePool",
    # xai
    "xai",
]
