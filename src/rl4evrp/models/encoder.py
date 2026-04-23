import torch.nn as nn

from .attention import MultiHeadAttention


class GATEncoderLayer(nn.Module):
    """Graph Attention Transformer layer with residual connections."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class EVRPEncoder(nn.Module):
    """Node encoding network (GAT or MLP-based)."""

    def __init__(self, input_dim: int = 7, embed_dim: int = 128,
                 n_heads: int = 8, n_layers: int = 3, enc_type: str = 'gat'):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.enc_type = enc_type

        if enc_type == 'gat':
            self.layers = nn.ModuleList([
                GATEncoderLayer(embed_dim, n_heads) for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(n_layers)
            ])

        self.last_attn = None  # stored for XAI

    def forward(self, x):
        """
        Args:
            x: (B, n_nodes, input_dim)
        Returns:
            node_embeddings: (B, n_nodes, embed_dim)
            graph_embedding: (B, embed_dim)
        """
        h = self.embed(x)

        for i, layer in enumerate(self.layers):
            if self.enc_type == 'gat' and isinstance(layer, GATEncoderLayer):
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
