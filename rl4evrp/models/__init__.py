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
        """
        Initialize multi-head attention.
        
        Args:
            n_heads: Number of attention heads
            input_dim: Input feature dimension
            embed_dim: Output embedding dimension (must be divisible by n_heads)
        """
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
        """
        Compute multi-head attention.
        
        Args:
            q: Query tensor (B, Tq, input_dim)
            k: Key tensor (B, Tk, input_dim), defaults to q
            v: Value tensor (B, Tk, input_dim), defaults to k
            mask: Boolean mask (B, Tk) where True means mask out
            return_attn: Whether to return attention weights
        
        Returns:
            output: Attention output (B, Tq, embed_dim)
            attn: (optional) Attention weights (B, n_heads, Tq, Tk)
        """
        if k is None:
            k = q
        if v is None:
            v = k
        
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape
        H, D = self.n_heads, self.head_dim
        
        # Project and reshape: (B, T, D) -> (B, n_heads, T, head_dim)
        Q = self.W_q(q).view(B, Tq, H, D).transpose(1, 2)
        K = self.W_k(k).view(B, Tk, H, D).transpose(1, 2)
        V = self.W_v(v).view(B, Tk, H, D).transpose(1, 2)
        
        # Compute attention scores: (B, H, Tq, Tk)
        scores = self.norm_factor * torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply mask
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
            scores = scores.masked_fill(m, float('-inf'))
        
        # Softmax with numerical stability
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        
        # Apply attention to values: (B, H, Tq, head_dim)
        out = torch.matmul(attn, V)
        
        # Concatenate heads and project: (B, Tq, embed_dim)
        out = self.W_o(out.transpose(1, 2).contiguous().view(B, Tq, H * D))
        
        return (out, attn) if return_attn else out


class GATEncoderLayer(nn.Module):
    """Graph Attention Transformer layer with residual connections."""
    
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int = 256, dropout: float = 0.0):
        """
        Initialize GAT encoder layer.
        
        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate (not used, kept for compatibility)
        """
        super().__init__()
        
        # Attention sublayer
        self.attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (B, n, embed_dim)
        
        Returns:
            h: Output tensor (B, n, embed_dim)
        """
        # Self-attention + residual
        x = self.norm1(x + self.attn(x))
        
        # Feed-forward + residual
        x = self.norm2(x + self.ff(x))
        
        return x


class EVRPEncoder(nn.Module):
    """Node encoding network (GAT or MLP-based)."""
    
    def __init__(self, input_dim: int = 7, embed_dim: int = 128,
                 n_heads: int = 8, n_layers: int = 3, enc_type: str = 'gat'):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input feature dimension (node features)
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            enc_type: 'gat' for graph attention or 'mlp' for simple MLP
        """
        super().__init__()
        
        self.embed = nn.Linear(input_dim, embed_dim)
        self.enc_type = enc_type
        
        if enc_type == 'gat':
            self.layers = nn.ModuleList([
                GATEncoderLayer(embed_dim, n_heads)
                for _ in range(n_layers)
            ])
        else:  # mlp
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(embed_dim)
                )
                for _ in range(n_layers)
            ])
        
        self.last_attn = None  # Stored for XAI analysis
    
    def forward(self, x):
        """
        Encode node features.
        
        Args:
            x: Node features (B, n_nodes, 7)
        
        Returns:
            node_embeddings: (B, n_nodes, embed_dim)
            graph_embedding: (B, embed_dim) [mean pooling]
        """
        h = self.embed(x)
        
        for i, layer in enumerate(self.layers):
            if self.enc_type == 'gat' and isinstance(layer, GATEncoderLayer):
                # Store attention from last GAT layer for XAI
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
    """Action decoder with cross-attention to node embeddings."""
    
    def __init__(self, embed_dim: int = 128, n_heads: int = 8):
        """
        Initialize decoder.
        
        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        
        # Project current node + state to context
        self.proj_ctx = nn.Linear(embed_dim + 2, embed_dim)
        
        # Cross-attention: context attends to node embeddings
        self.cross_attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)
    
    def forward(self, node_emb, graph_emb, battery_norm, cargo_norm,
                cur_node_idx, invalid_mask=None, return_attn=False):
        """
        Decode action logits from current state and node embeddings.
        
        Args:
            node_emb: Node embeddings (B, n_nodes, embed_dim)
            graph_emb: Graph-level embedding (B, embed_dim)
            battery_norm: Battery state (B, 1)
            cargo_norm: Cargo state (B, 1)
            cur_node_idx: Current node index (B,)
            invalid_mask: Boolean mask where True means invalid action
            return_attn: Whether to return attention weights
        
        Returns:
            logits: Action logits (B, n_nodes)
            attn: (optional) Attention weights
        """
        B, n, D = node_emb.shape
        
        # Get current node embedding
        idx = cur_node_idx.long().view(B, 1, 1).expand(B, 1, D)
        cur_emb = node_emb.gather(1, idx).squeeze(1)
        
        # Project context (current node + battery + cargo)
        ctx = self.proj_ctx(torch.cat([cur_emb, battery_norm, cargo_norm], -1)).unsqueeze(1)
        
        # Cross-attention
        if return_attn:
            attended, attn = self.cross_attn(ctx, node_emb, node_emb,
                                             mask=invalid_mask, return_attn=True)
        else:
            attended = self.cross_attn(ctx, node_emb, node_emb, mask=invalid_mask)
            attn = None
        
        # Compute action logits as dot product with node embeddings
        scores = (attended * node_emb).sum(-1)  # (B, n)
        
        # Mask invalid actions
        if invalid_mask is not None:
            scores = scores.masked_fill(invalid_mask, float('-inf'))
        
        return (scores, attn) if return_attn else scores
