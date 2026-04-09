# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-5.1 RoPE utilities.

Tests the split-half (Llama/NeoX) RoPE implementation against
the HuggingFace GlmMoeDsa reference implementation.
"""

import pytest
import torch

from src.rope_util import (
    Glm5RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_bshd,
)


class TestRotateHalf:
    """Test the split-half rotation function."""

    def test_basic_rotation(self):
        """rotate_half should swap and negate halves: [-x2, x1]."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 64)
        result = rotate_half(x)
        assert result.shape == x.shape

    def test_double_rotation_negates(self):
        """Applying rotate_half twice should negate the input."""
        x = torch.randn(2, 4, 8, 64)
        result = rotate_half(rotate_half(x))
        torch.testing.assert_close(result, -x)


class TestGlm5RotaryEmbedding:
    """Test the RoPE embedding generation."""

    def test_output_shapes(self):
        dim = 64
        emb = Glm5RotaryEmbedding(dim=dim, max_position_embeddings=4096, base=1000000.0)
        x = torch.randn(2, 4, 8, dim, dtype=torch.bfloat16)
        cos, sin = emb(x, seq_len=128)
        assert cos.shape == (128, dim)
        assert sin.shape == (128, dim)
        assert cos.dtype == torch.bfloat16
        assert sin.dtype == torch.bfloat16

    def test_cos_sin_range(self):
        """cos and sin values should be in [-1, 1]."""
        emb = Glm5RotaryEmbedding(dim=64, max_position_embeddings=4096)
        cos, sin = emb(torch.randn(1, 1, 1, 64), seq_len=1024)
        assert cos.float().abs().max() <= 1.0 + 1e-6
        assert sin.float().abs().max() <= 1.0 + 1e-6

    def test_cos_sin_symmetry(self):
        """cos/sin should be cat(freqs, freqs) — first half equals second half."""
        emb = Glm5RotaryEmbedding(dim=64)
        cos, sin = emb(torch.randn(1, 1, 1, 64), seq_len=100)
        cos_f = cos.float()
        sin_f = sin.float()
        torch.testing.assert_close(cos_f[:, :32], cos_f[:, 32:])
        torch.testing.assert_close(sin_f[:, :32], sin_f[:, 32:])

    def test_high_theta_base(self):
        """With theta=1M, lower-frequency components should vary slowly."""
        emb = Glm5RotaryEmbedding(dim=64, base=1000000.0)
        cos, sin = emb(torch.randn(1, 1, 1, 64), seq_len=100)
        # Last unique frequency component (lowest freq) should barely change over 100 positions
        # inv_freq[0] = 1.0 (highest freq), inv_freq[31] ≈ 0 (lowest freq)
        # cos/sin are cat(freqs, freqs), so index 31 = last unique frequency
        cos_f = cos.float()
        delta = (cos_f[-1, 31] - cos_f[0, 31]).abs()
        assert delta < 0.01, f"Lowest frequency changed by {delta} over 100 positions"


class TestApplyRotaryPosEmb:
    """Test the RoPE application functions."""

    def test_bhsd_layout(self):
        """apply_rotary_pos_emb should work with BHSD layout (bsz, heads, seq, dim)."""
        dim = 64
        emb = Glm5RotaryEmbedding(dim=dim)
        cos, sin = emb(torch.randn(1, 1, 1, dim), seq_len=16)
        position_ids = torch.arange(8).unsqueeze(0)
        q = torch.randn(1, 4, 8, dim)
        result = apply_rotary_pos_emb(q, cos, sin, position_ids)
        assert result.shape == q.shape

    def test_bshd_layout(self):
        """apply_rotary_pos_emb_bshd should work with BSHD layout (bsz, seq, heads, dim)."""
        dim = 64
        emb = Glm5RotaryEmbedding(dim=dim)
        cos, sin = emb(torch.randn(1, 1, 1, dim), seq_len=16)
        position_ids = torch.arange(8).unsqueeze(0)
        q = torch.randn(1, 8, 4, dim)  # BSHD
        result = apply_rotary_pos_emb_bshd(q, cos, sin, position_ids)
        assert result.shape == q.shape

    def test_bshd_layout_3d(self):
        """apply_rotary_pos_emb_bshd should work with 3D input (bsz, seq, dim)."""
        dim = 64
        emb = Glm5RotaryEmbedding(dim=dim)
        cos, sin = emb(torch.randn(1, 1, 1, dim), seq_len=16)
        position_ids = torch.arange(8).unsqueeze(0)
        k = torch.randn(1, 8, dim)  # BSD (no head dim)
        result = apply_rotary_pos_emb_bshd(k, cos, sin, position_ids)
        assert result.shape == k.shape

    def test_norm_preservation(self):
        """RoPE should approximately preserve vector norms."""
        dim = 64
        emb = Glm5RotaryEmbedding(dim=dim)
        cos, sin = emb(torch.randn(1, 1, 1, dim), seq_len=16)
        position_ids = torch.arange(8).unsqueeze(0)
        q = torch.randn(1, 4, 8, dim)
        result = apply_rotary_pos_emb(q, cos, sin, position_ids)

        orig_norm = q.norm(dim=-1)
        result_norm = result.norm(dim=-1)
        ratio = result_norm / (orig_norm + 1e-8)
        assert (ratio - 1.0).abs().max() < 0.05, "RoPE should preserve norms"

    def test_matches_hf_reference(self):
        """Verify against the HF GlmMoeDsa reference RoPE implementation."""
        dim = 64
        base = 1000000.0
        seq_len = 8
        bsz = 1
        n_heads = 4

        # Reference: same as HF modeling_glm_moe_dsa.py
        def hf_rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def hf_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            return (x * cos) + (hf_rotate_half(x) * sin)

        # Compute inv_freq and cos/sin the HF way
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position_ids = torch.arange(seq_len).unsqueeze(0)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb_hf = torch.cat((freqs, freqs), dim=-1)
        cos_hf = emb_hf.cos().squeeze(0)  # (seq_len, dim)
        sin_hf = emb_hf.sin().squeeze(0)

        # Apply HF reference (index by position_ids first, as HF does)
        q = torch.randn(bsz, n_heads, seq_len, dim)
        cos_hf_pos = cos_hf[position_ids]  # (bsz, seq_len, dim)
        sin_hf_pos = sin_hf[position_ids]
        q_pe_hf = hf_apply_rotary_pos_emb(q, cos_hf_pos, sin_hf_pos, unsqueeze_dim=1)

        # Apply our implementation
        our_emb = Glm5RotaryEmbedding(dim=dim, base=base)
        cos_ours, sin_ours = our_emb(q[:, 0, :, :], seq_len=seq_len)
        q_pe_ours = apply_rotary_pos_emb(q, cos_ours, sin_ours, position_ids)

        torch.testing.assert_close(q_pe_ours, q_pe_hf, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
