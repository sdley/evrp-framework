"""
A2C (Advantage Actor-Critic) Agent for EVRP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional

from rl4evrp.models import EVRPEncoder, EVRPDecoder


class A2CAgent(nn.Module):
    """Advantage Actor-Critic agent for EVRP."""

    def __init__(
        self,
        input_dim: int = 10,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        enc_type: str = "gat",
        n_episodes: int = 800,
        device: str = "cpu",
        use_time_context: bool = True,
    ):
        """
        Initialize A2C agent.

        Args:
            input_dim: Node feature dimension
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            lr: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss weight
            enc_type: Encoder type ('gat' or 'mlp')
            n_episodes: Total number of episodes (for scheduler)
            device: Device to use ('cpu' or 'cuda')
            use_time_context: Whether decoder context includes time_norm
        """
        super().__init__()

        self.gamma = gamma
        self.ent_coef = entropy_coef
        self.val_coef = value_coef
        self.device = device
        self.use_time_context = use_time_context

        # Build network
        self.encoder = EVRPEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            enc_type=enc_type,
        )

        self.decoder = EVRPDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            use_time_context=use_time_context,
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_episodes, eta_min=lr / 10
        )

        self.to(device)

    def _forward(self, obs: dict, return_attn: bool = False):
        """
        Forward pass through the network.

        Args:
            obs: Observation dict from environment
            return_attn: Whether to return attention weights

        Returns:
            action_logits: (1, n_nodes)
            value: (1, 1)
            node_emb: (1, n_nodes, embed_dim)
            decoder_attn: (optional) decoder attention
            encoder_attn: (optional) encoder attention
        """
        feats = obs["node_features"].unsqueeze(0).to(self.device)
        bat_norm = torch.tensor([[obs["battery_norm"]]], dtype=torch.float32, device=self.device)
        cargo_norm = torch.tensor([[obs["cargo_norm"]]], dtype=torch.float32, device=self.device)
        time_norm = torch.tensor([[obs.get("time_norm", 0.0)]], dtype=torch.float32, device=self.device)
        cur_idx = torch.tensor([obs["current_node"]], dtype=torch.long, device=self.device)
        invalid_mask = torch.tensor(~obs["valid_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

        # Encode nodes
        node_emb, graph_emb = self.encoder(feats)

        # Decode action
        if return_attn:
            scores, dec_attn = self.decoder(
                node_emb=node_emb,
                graph_emb=graph_emb,
                battery_norm=bat_norm,
                cargo_norm=cargo_norm,
                time_norm=time_norm,
                cur_node_idx=cur_idx,
                invalid_mask=invalid_mask,
                return_attn=True,
            )
            value = self.value_head(graph_emb)
            return scores, value, node_emb, dec_attn, self.encoder.last_attn
        else:
            scores = self.decoder(
                node_emb=node_emb,
                graph_emb=graph_emb,
                battery_norm=bat_norm,
                cargo_norm=cargo_norm,
                time_norm=time_norm,
                cur_node_idx=cur_idx,
                invalid_mask=invalid_mask,
                return_attn=False,
            )
            value = self.value_head(graph_emb)
            return scores, value, node_emb

    def select_action(self, obs: dict, greedy: bool = False):
        """
        Select an action from the current observation.

        Args:
            obs: Observation dict from environment
            greedy: If True, select argmax action; otherwise sample

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: Estimated state value
        """
        scores, value, node_emb = self._forward(obs)

        dist = Categorical(logits=scores.squeeze(0))

        if greedy:
            action = dist.probs.argmax().item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def update(self, transitions: List[Dict]) -> Tuple[float, float]:
        """
        Update agent using collected transitions.

        Args:
            transitions: List of transition dicts with keys:
                        'r' (reward), 'lp' (log_prob), 'ent' (entropy), 'val' (value)

        Returns:
            loss: Total loss value
            mean_entropy: Mean entropy of action distributions
        """
        rewards = [t["r"] for t in transitions]
        log_probs = torch.stack([t["lp"] for t in transitions])
        entropies = torch.stack([t["ent"] for t in transitions])
        values = torch.stack([t["val"].squeeze() for t in transitions])

        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + self.val_coef * value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return float(total_loss.item()), float(entropies.mean().item())

    def get_action_for_inference(self, obs: dict, greedy: bool = True):
        """
        Get action for inference/evaluation (no gradient tracking).
        """
        with torch.no_grad():
            action, _, _, _ = self.select_action(obs, greedy=greedy)
        return action