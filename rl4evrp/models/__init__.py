"""
Neural network models: Encoder (GAT), Decoder, and attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self/cross-attention mechanism with numerical stability."""

    def __init__(self, n_heads: int, input_dim: int, embed_dim: int):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None, return_attn=False):
        if k is None:
            k = q
        if v is None:
            v = k

        B, Tq, _ = q.shape
        _, Tk, _ = k.shape
        H, D = self.n_heads, self.head_dim

        Q = self.W_q(q).view(B, Tq, H, D).transpose(1, 2)
        K = self.W_k(k).view(B, Tk, H, D).transpose(1, 2)
        V = self.W_v(v).view(B, Tk, H, D).transpose(1, 2)

        scores = self.norm_factor * torch.matmul(Q, K.transpose(-2, -1))

        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
            scores = scores.masked_fill(m, float("-inf"))

        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(attn, V)
        out = self.W_o(out.transpose(1, 2).contiguous().view(B, Tq, H * D))

        return (out, attn) if return_attn else out


class GATEncoderLayer(nn.Module):
    """Graph Attention Transformer layer with residual connections."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int = 256, dropout: float = 0.0):
        super().__init__()

        self.attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class EVRPEncoder(nn.Module):
    """Node encoding network (GAT or MLP-based)."""

    def __init__(self, input_dim: int = 7, embed_dim: int = 128,
                 n_heads: int = 8, n_layers: int = 3, enc_type: str = "gat"):
        super().__init__()

        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, embed_dim)
        self.enc_type = enc_type

        if enc_type == "gat":
            self.layers = nn.ModuleList([
                GATEncoderLayer(embed_dim, n_heads)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(embed_dim)
                )
                for _ in range(n_layers)
            ])

        self.last_attn = None

    def forward(self, x):
        """
        x: (B, n_nodes, input_dim)
        """
        h = self.embed(x)

        for i, layer in enumerate(self.layers):
            if self.enc_type == "gat" and isinstance(layer, GATEncoderLayer):
                if i == len(self.layers) - 1:
                    out, attn = layer.attn(h, return_attn=True)
                    self.last_attn = attn.detach()
                    h = layer.norm1(h + out)
                    h = layer.norm2(h + layer.ff(h))
                else:
                    h = layer(h)
            else:
                h = layer(h)

        return h, h.mean(dim=1)


class EVRPDecoder(nn.Module):
    """Action decoder with optional time context."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, use_time_context: bool = True):
        super().__init__()

        self.use_time_context = use_time_context
        ctx_extra = 3 if use_time_context else 2

        self.proj_ctx = nn.Linear(embed_dim + ctx_extra, embed_dim)
        self.cross_attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)

    def forward(self, node_emb, graph_emb, battery_norm, cargo_norm, time_norm,
                cur_node_idx, invalid_mask=None, return_attn=False):
        """
        node_emb: (B, n_nodes, embed_dim)
        graph_emb: (B, embed_dim)
        battery_norm: (B, 1)
        cargo_norm: (B, 1)
        time_norm: (B, 1)
        cur_node_idx: (B,)
        """
        B, n, D = node_emb.shape

        idx = cur_node_idx.long().view(B, 1, 1).expand(B, 1, D)
        cur_emb = node_emb.gather(1, idx).squeeze(1)

        if self.use_time_context:
            ctx_in = torch.cat([cur_emb, battery_norm, cargo_norm, time_norm], dim=-1)
        else:
            ctx_in = torch.cat([cur_emb, battery_norm, cargo_norm], dim=-1)

        ctx = self.proj_ctx(ctx_in).unsqueeze(1)

        if return_attn:
            attended, attn = self.cross_attn(ctx, node_emb, node_emb,
                                             mask=invalid_mask, return_attn=True)
        else:
            attended = self.cross_attn(ctx, node_emb, node_emb, mask=invalid_mask)
            attn = None

        scores = (attended * node_emb).sum(-1)

        if invalid_mask is not None:
            scores = scores.masked_fill(invalid_mask, float("-inf"))

        return (scores, attn) if return_attn else scores