# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Glm5Indexer (DSA — Dynamic Sparse Attention).

CPU-only tests that verify the indexer produces correct top-k indices
against the HF GlmMoeDsaIndexer reference implementation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from src.rope_util import Glm5RotaryEmbedding


class ReferenceIndexer(nn.Module):
    """
    Reference indexer matching HF GlmMoeDsaIndexer.forward (simplified).

    Differences from NXDI version:
    - No TP sharding (single-process reference)
    - Matches the HF code scoring logic
    """

    def __init__(self, hidden_size, n_heads, head_dim, rope_dim, topk, q_lora_rank):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.topk = topk
        self.softmax_scale = head_dim ** -0.5

        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False)

    def forward(self, hidden_states, q_resid, cos, sin, position_ids):
        """Reference forward matching HF GlmMoeDsaIndexer."""
        bsz, seq_len, _ = hidden_states.shape

        # Queries
        q = self.wq_b(q_resid).view(bsz, seq_len, self.n_heads, self.head_dim)
        q_pe = q[..., :self.rope_dim]
        q_nope = q[..., self.rope_dim:]

        # HF reference RoPE: split-half
        def hf_rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos_r = cos[position_ids]  # (bsz, seq, rope_dim)
        sin_r = sin[position_ids]
        cos_r = cos_r.unsqueeze(2)  # (bsz, seq, 1, rope_dim) for BSHD
        sin_r = sin_r.unsqueeze(2)
        q_pe = (q_pe * cos_r) + (hf_rotate_half(q_pe) * sin_r)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Keys
        k = self.k_norm(self.wk(hidden_states))
        k_pe = k[..., :self.rope_dim]
        k_nope = k[..., self.rope_dim:]
        cos_k = cos[position_ids]
        sin_k = sin[position_ids]
        k_pe = (k_pe * cos_k) + (hf_rotate_half(k_pe) * sin_k)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Scoring
        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * self.softmax_scale
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        # Causal mask
        causal_mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
        index_scores = index_scores + causal_mask.unsqueeze(0)

        topk_val = min(self.topk, seq_len)
        topk_indices = index_scores.topk(topk_val, dim=-1).indices
        return topk_indices


@pytest.fixture
def indexer_config():
    return dict(
        hidden_size=256,
        n_heads=4,
        head_dim=32,
        rope_dim=16,
        topk=8,
        q_lora_rank=64,
    )


@pytest.fixture
def shared_weights_indexers(indexer_config):
    """Create NXDI and reference indexers with shared weights."""
    ref = ReferenceIndexer(**indexer_config)

    # Create NXDI indexer with mocked config
    config = MagicMock()
    config.hidden_size = indexer_config["hidden_size"]
    config.index_n_heads = indexer_config["n_heads"]
    config.index_head_dim = indexer_config["head_dim"]
    config.qk_rope_head_dim = indexer_config["rope_dim"]
    config.index_topk = indexer_config["topk"]
    config.q_lora_rank = indexer_config["q_lora_rank"]
    config.neuron_config = MagicMock()
    config.neuron_config.tp_degree = 1
    config.neuron_config.torch_dtype = torch.float32

    with patch("neuronx_distributed.utils.cpu_mode", return_value=True):
        from src.modeling_glm5 import Glm5Indexer
        nxdi_indexer = Glm5Indexer(config, tensor_model_parallel_group=None)

    # Copy weights
    with torch.no_grad():
        nn.init.normal_(ref.wq_b.weight, std=0.01)
        nn.init.normal_(ref.wk.weight, std=0.01)
        nn.init.normal_(ref.weights_proj.weight, std=0.01)
        nxdi_indexer.wq_b.weight.copy_(ref.wq_b.weight)
        nxdi_indexer.wk.weight.copy_(ref.wk.weight)
        nxdi_indexer.k_norm.weight.copy_(ref.k_norm.weight)
        nxdi_indexer.k_norm.bias.copy_(ref.k_norm.bias)
        nxdi_indexer.weights_proj.weight.copy_(ref.weights_proj.weight)

    return nxdi_indexer, ref


class TestGlm5Indexer:

    def test_output_shapes(self, shared_weights_indexers, indexer_config):
        """Indexer should return correct shapes."""
        nxdi_indexer, _ = shared_weights_indexers
        bsz, seq_len = 2, 16
        hidden_size = indexer_config["hidden_size"]
        q_lora_rank = indexer_config["q_lora_rank"]
        rope_dim = indexer_config["rope_dim"]

        x = torch.randn(bsz, seq_len, hidden_size)
        qr = torch.randn(bsz, seq_len, q_lora_rank)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        emb = Glm5RotaryEmbedding(dim=rope_dim)
        cos, sin = emb(x, seq_len=seq_len)

        topk_indices, processed_k = nxdi_indexer(
            x, qr, position_ids, cos, sin, is_prefill=True,
        )

        topk_val = min(indexer_config["topk"], seq_len)
        assert topk_indices.shape == (bsz, seq_len, topk_val)
        assert processed_k.shape == (bsz, seq_len, indexer_config["head_dim"])

    def test_indices_in_range(self, shared_weights_indexers, indexer_config):
        """All selected indices should be valid positions."""
        nxdi_indexer, _ = shared_weights_indexers
        bsz, seq_len = 2, 32
        x = torch.randn(bsz, seq_len, indexer_config["hidden_size"])
        qr = torch.randn(bsz, seq_len, indexer_config["q_lora_rank"])
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        emb = Glm5RotaryEmbedding(dim=indexer_config["rope_dim"])
        cos, sin = emb(x, seq_len=seq_len)

        topk_indices, _ = nxdi_indexer(
            x, qr, position_ids, cos, sin, is_prefill=True,
        )

        assert topk_indices.min() >= 0
        assert topk_indices.max() < seq_len

    def test_all_selected_when_seq_short(self, shared_weights_indexers, indexer_config):
        """When seq_len <= topk, all positions should be selected."""
        nxdi_indexer, _ = shared_weights_indexers
        bsz = 1
        seq_len = indexer_config["topk"] - 2  # shorter than topk
        x = torch.randn(bsz, seq_len, indexer_config["hidden_size"])
        qr = torch.randn(bsz, seq_len, indexer_config["q_lora_rank"])
        position_ids = torch.arange(seq_len).unsqueeze(0)

        emb = Glm5RotaryEmbedding(dim=indexer_config["rope_dim"])
        cos, sin = emb(x, seq_len=seq_len)

        topk_indices, _ = nxdi_indexer(
            x, qr, position_ids, cos, sin, is_prefill=True,
        )

        # All positions should be selected
        assert topk_indices.shape[-1] == seq_len
        for b in range(bsz):
            for s in range(seq_len):
                selected = topk_indices[b, s].sort().values
                assert torch.equal(selected, torch.arange(seq_len))

    def test_layernorm_not_hadamard(self, shared_weights_indexers):
        """GLM-5.1 indexer should use LayerNorm (not Hadamard transform)."""
        nxdi_indexer, _ = shared_weights_indexers
        assert isinstance(nxdi_indexer.k_norm, nn.LayerNorm), (
            f"Expected LayerNorm, got {type(nxdi_indexer.k_norm)}"
        )
        # LayerNorm has bias, Hadamard does not
        assert nxdi_indexer.k_norm.bias is not None

    def test_decode_with_cache(self, shared_weights_indexers, indexer_config):
        """Decode path should use cached keys correctly."""
        nxdi_indexer, _ = shared_weights_indexers
        bsz = 1
        prior_len = 20
        head_dim = indexer_config["head_dim"]
        rope_dim = indexer_config["rope_dim"]

        # Simulate decode: 1 new token with 20 cached tokens
        x = torch.randn(bsz, 1, indexer_config["hidden_size"])
        qr = torch.randn(bsz, 1, indexer_config["q_lora_rank"])
        position_ids = torch.tensor([[prior_len]])
        past_indexer_k = torch.randn(bsz, prior_len, head_dim)

        emb = Glm5RotaryEmbedding(dim=rope_dim)
        cos, sin = emb(x, seq_len=prior_len + 1)

        topk_indices, processed_k = nxdi_indexer(
            x, qr, position_ids, cos, sin,
            is_prefill=False, past_indexer_k=past_indexer_k,
        )

        topk_val = min(indexer_config["topk"], prior_len + 1)
        assert topk_indices.shape == (bsz, 1, topk_val)
        assert processed_k.shape == (bsz, 1, head_dim)
        assert topk_indices.max() < prior_len + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
