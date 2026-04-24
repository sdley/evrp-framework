import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class EVRPDecoder(nn.Module):
    """Action decoder with cross-attention to node embeddings."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.proj_ctx = nn.Linear(embed_dim + 2, embed_dim)
        self.cross_attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)

    def forward(self, node_emb, graph_emb, battery_norm, cargo_norm,
                cur_node_idx, invalid_mask=None, return_attn=False):
        """
        Args:
            node_emb: (B, n_nodes, embed_dim)
            graph_emb: (B, embed_dim)
            battery_norm: (B, 1)
            cargo_norm: (B, 1)
            cur_node_idx: (B,)
            invalid_mask: Boolean mask where True means invalid action
            return_attn: Whether to return attention weights
        Returns:
            logits: (B, n_nodes)
            attn: (optional) attention weights
        """
        B, n, D = node_emb.shape

        idx = cur_node_idx.long().view(B, 1, 1).expand(B, 1, D)
        cur_emb = node_emb.gather(1, idx).squeeze(1)

        ctx = self.proj_ctx(torch.cat([cur_emb, battery_norm, cargo_norm], -1)).unsqueeze(1)

        if return_attn:
            attended, attn = self.cross_attn(
                ctx, node_emb, node_emb, mask=invalid_mask, return_attn=True
            )
        else:
            attended = self.cross_attn(ctx, node_emb, node_emb, mask=invalid_mask)
            attn = None

        scores = (attended * node_emb).sum(-1)

        if invalid_mask is not None:
            scores = scores.masked_fill(invalid_mask, float('-inf'))

        return (scores, attn) if return_attn else scores
