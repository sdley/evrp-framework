import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self/cross-attention mechanism with numerical stability."""

    def __init__(self, n_heads: int, input_dim: int, embed_dim: int):
        super().__init__()
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None, return_attn=False):
        """
        Args:
            q: (B, Tq, input_dim)
            k: (B, Tk, input_dim), defaults to q
            v: (B, Tk, input_dim), defaults to k
            mask: Boolean mask (B, Tk) where True means mask out
            return_attn: Whether to return attention weights
        Returns:
            output: (B, Tq, embed_dim)
            attn: (optional) (B, n_heads, Tq, Tk)
        """
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
            scores = scores.masked_fill(m, float('-inf'))

        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(attn, V)
        out = self.W_o(out.transpose(1, 2).contiguous().view(B, Tq, H * D))

        return (out, attn) if return_attn else out
