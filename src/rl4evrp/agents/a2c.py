import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional

from rl4evrp.models import EVRPEncoder, EVRPDecoder


class A2CAgent(nn.Module):
    """Advantage Actor-Critic agent for EVRP."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 3,
                 lr: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 enc_type: str = 'gat', n_episodes: int = 800,
                 device: str = 'cpu'):
        super().__init__()
        self.gamma = gamma
        self.ent_coef = entropy_coef
        self.val_coef = value_coef
        self.device = device

        self.encoder = EVRPEncoder(7, embed_dim, n_heads, n_layers, enc_type)
        self.decoder = EVRPDecoder(embed_dim, n_heads)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_episodes, eta_min=lr / 10
        )
        self.to(device)

    def _forward(self, obs: dict, return_attn: bool = False):
        """
        Args:
            obs: Observation dict from EVRPEnv
            return_attn: Whether to also return encoder/decoder attention
        Returns:
            action_logits, value, node_emb[, dec_attn, enc_attn]
        """
        feats = obs['node_features'].unsqueeze(0).to(self.device)
        bat_norm = torch.FloatTensor([[obs['battery_norm']]]).to(self.device)
        cargo_norm = torch.FloatTensor([[obs['cargo_norm']]]).to(self.device)
        cur_idx = torch.LongTensor([obs['current_node']]).to(self.device)
        invalid_mask = torch.BoolTensor(~obs['valid_mask']).unsqueeze(0).to(self.device)

        node_emb, graph_emb = self.encoder(feats)

        if return_attn:
            scores, dec_attn = self.decoder(
                node_emb, graph_emb, bat_norm, cargo_norm,
                cur_idx, invalid_mask, return_attn=True,
            )
            value = self.value_head(graph_emb)
            return scores, value, node_emb, dec_attn, self.encoder.last_attn
        else:
            scores = self.decoder(node_emb, graph_emb, bat_norm, cargo_norm, cur_idx, invalid_mask)
            value = self.value_head(graph_emb)
            return scores, value, node_emb

    def select_action(self, obs: dict, greedy: bool = False):
        """
        Args:
            obs: Observation dict
            greedy: If True, argmax; otherwise sample
        Returns:
            (action, log_prob, entropy, value)
        """
        scores, value, _ = self._forward(obs)
        dist = Categorical(logits=scores.squeeze(0))
        action = dist.probs.argmax().item() if greedy else dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        return action, log_prob, dist.entropy(), value

    def update(self, transitions: List[Dict]) -> Tuple[float, float]:
        """
        A2C update from a collected episode.

        Args:
            transitions: List of dicts with keys 'r', 'lp', 'ent', 'val'
        Returns:
            (total_loss, mean_entropy)
        """
        rewards = [t['r'] for t in transitions]
        log_probs = torch.stack([t['lp'] for t in transitions])
        entropies = torch.stack([t['ent'] for t in transitions])
        values = torch.stack([t['val'].squeeze() for t in transitions])

        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        std = returns.std() if returns.numel() > 1 else torch.tensor(1.0, device=self.device)
        returns = (returns - returns.mean()) / (std + 1e-8)

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

    def get_action_for_inference(self, obs: dict, greedy: bool = True) -> int:
        with torch.no_grad():
            action, _, _, _ = self.select_action(obs, greedy=greedy)
        return action
