import torch
import pytest

from rl4evrp.models.attention import MultiHeadAttention
from rl4evrp.models.encoder import GATEncoderLayer, EVRPEncoder
from rl4evrp.models.decoder import EVRPDecoder


B, N, D = 2, 10, 64  # batch, nodes, embed_dim


class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(n_heads=4, input_dim=D, embed_dim=D)

    def test_self_attention_shape(self, mha):
        x = torch.randn(B, N, D)
        out = mha(x)
        assert out.shape == (B, N, D)

    def test_cross_attention_shape(self, mha):
        q = torch.randn(B, 1, D)
        kv = torch.randn(B, N, D)
        out = mha(q, kv, kv)
        assert out.shape == (B, 1, D)

    def test_returns_attn_when_requested(self, mha):
        x = torch.randn(B, N, D)
        out, attn = mha(x, return_attn=True)
        assert out.shape == (B, N, D)
        assert attn.shape == (B, 4, N, N)  # (B, heads, Tq, Tk)

    def test_mask_zeroes_out_positions(self, mha):
        x = torch.randn(B, N, D)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, -1] = True  # mask last node
        out, attn = mha(x, mask=mask, return_attn=True)
        assert (attn[..., -1] == 0).all()

    def test_embed_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(n_heads=3, input_dim=D, embed_dim=D)  # 64 % 3 != 0

    def test_output_is_finite(self, mha):
        x = torch.randn(B, N, D)
        out = mha(x)
        assert torch.isfinite(out).all()

    def test_all_masked_gives_finite_output(self, mha):
        x = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.bool)  # all masked
        out = mha(x, mask=mask)
        assert torch.isfinite(out).all()


class TestGATEncoderLayer:
    @pytest.fixture
    def layer(self):
        return GATEncoderLayer(embed_dim=D, n_heads=4, ff_dim=128)

    def test_output_shape(self, layer):
        x = torch.randn(B, N, D)
        out = layer(x)
        assert out.shape == (B, N, D)

    def test_residual_changes_input(self, layer):
        x = torch.randn(B, N, D)
        out = layer(x)
        assert not torch.allclose(x, out)

    def test_output_is_finite(self, layer):
        x = torch.randn(B, N, D)
        assert torch.isfinite(layer(x)).all()


class TestEVRPEncoder:
    @pytest.fixture(params=["gat", "mlp"])
    def encoder(self, request):
        return EVRPEncoder(input_dim=7, embed_dim=D, n_heads=4,
                           n_layers=2, enc_type=request.param)

    def test_output_shapes(self, encoder):
        x = torch.randn(B, N, 7)
        node_emb, graph_emb = encoder(x)
        assert node_emb.shape == (B, N, D)
        assert graph_emb.shape == (B, D)

    def test_graph_emb_is_mean(self, encoder):
        x = torch.randn(B, N, 7)
        node_emb, graph_emb = encoder(x)
        torch.testing.assert_close(graph_emb, node_emb.mean(dim=1))

    def test_gat_stores_attention(self):
        enc = EVRPEncoder(input_dim=7, embed_dim=D, n_heads=4,
                          n_layers=2, enc_type="gat")
        x = torch.randn(B, N, 7)
        enc(x)
        assert enc.last_attn is not None
        assert enc.last_attn.shape == (B, 4, N, N)

    def test_mlp_no_attention(self):
        enc = EVRPEncoder(input_dim=7, embed_dim=D, n_heads=4,
                          n_layers=2, enc_type="mlp")
        x = torch.randn(B, N, 7)
        enc(x)
        assert enc.last_attn is None

    def test_output_is_finite(self, encoder):
        x = torch.randn(B, N, 7)
        node_emb, graph_emb = encoder(x)
        assert torch.isfinite(node_emb).all()
        assert torch.isfinite(graph_emb).all()


class TestEVRPDecoder:
    @pytest.fixture
    def decoder(self):
        return EVRPDecoder(embed_dim=D, n_heads=4)

    @pytest.fixture
    def decoder_inputs(self):
        node_emb = torch.randn(B, N, D)
        graph_emb = torch.randn(B, D)
        battery = torch.rand(B, 1)
        cargo = torch.rand(B, 1)
        cur_idx = torch.zeros(B, dtype=torch.long)
        return node_emb, graph_emb, battery, cargo, cur_idx

    def test_logits_shape(self, decoder, decoder_inputs):
        logits = decoder(*decoder_inputs)
        assert logits.shape == (B, N)

    def test_returns_attn_when_requested(self, decoder, decoder_inputs):
        logits, attn = decoder(*decoder_inputs, return_attn=True)
        assert logits.shape == (B, N)
        assert attn is not None

    def test_invalid_mask_sets_inf(self, decoder, decoder_inputs):
        node_emb, graph_emb, bat, cargo, cur_idx = decoder_inputs
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, 0] = True
        logits = decoder(node_emb, graph_emb, bat, cargo, cur_idx, invalid_mask=mask)
        assert (logits[:, 0] == float("-inf")).all()

    def test_output_is_finite_without_mask(self, decoder, decoder_inputs):
        logits = decoder(*decoder_inputs)
        assert torch.isfinite(logits).all()
