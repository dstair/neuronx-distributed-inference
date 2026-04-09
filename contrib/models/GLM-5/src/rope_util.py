# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RoPE utilities for GLM-5.1.

GLM-5.1 uses standard Llama/NeoX split-half RoPE with:
  - theta = 1,000,000 (high base frequency)
  - No YaRN context extension (rope_type = "default")
  - dim = qk_rope_head_dim = 64

Split-half convention: rotate_half splits x into first/second halves,
i.e. x[..., :d//2] and x[..., d//2:].  This is the standard Llama/NeoX
convention used by the HF GlmMoeDsa reference implementation.
"""

import torch
import torch.utils.checkpoint
from torch import nn


class Glm5RotaryEmbedding(nn.Module):
    """
    Standard RoPE (no YaRN) with split-half rotation.

    Produces a frequency table indexed by position, then cos/sin embeddings.
    The output cos/sin have shape (seq_len, 2 * dim//2) = (seq_len, dim),
    stored as cat(freqs, freqs) for compatibility with split-half rotate_half.
    """

    def __init__(self, dim, max_position_embeddings=202752, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

    def get_freqs_table(self, device, seq_len):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        return freqs

    def forward(self, x, seq_len=None, freqs=None):
        device = x.device
        dtype = x.dtype
        if freqs is None:
            freqs = self.get_freqs_table(device, seq_len)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor):
    """Split-half rotation (Llama/NeoX convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids):
    """
    Apply split-half RoPE to a tensor.

    Args:
        q: Input tensor. Last dim = rope_dim.
           Typical shapes: (bsz, n_heads, seq_len, rope_dim) for BHSD layout.
        cos: Cosine table, shape (max_seq_len, rope_dim), from Glm5RotaryEmbedding.
        sin: Sine table, shape (max_seq_len, rope_dim).
        position_ids: (bsz, seq_len) or (1, seq_len).

    Returns:
        Tensor with RoPE applied, same shape as q.
    """
    # cos/sin stored as cat(freqs, freqs); take the actual unique frequencies
    cos_half = cos.chunk(2, dim=-1)[0][position_ids]  # (bsz, seq_len, dim//2)
    sin_half = sin.chunk(2, dim=-1)[0][position_ids]  # (bsz, seq_len, dim//2)
    # Reconstruct full-dim cos/sin for split-half convention
    cos_full = torch.cat([cos_half, cos_half], dim=-1)[0]  # (seq_len, dim)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)[0]  # (seq_len, dim)

    # Add singleton dims for broadcasting with multi-head inputs
    # 3D input (bsz, seq, d): no extra dim needed
    # 4D input (bsz, heads, seq, d) — BHSD: unsqueeze at dim 0 for batch and dim 1 for heads
    for _ in range(q.dim() - 3):
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)

    q_embed = (q * cos_full) + (rotate_half(q) * sin_full)
    return q_embed.to(q.dtype)


def apply_rotary_pos_emb_bshd(x: torch.Tensor, cos, sin, position_ids):
    """
    Apply split-half RoPE to a tensor in BSHD layout (used by the DSA Indexer).

    Args:
        x: Input tensor. Shape (bsz, seq_len, [n_heads,] rope_dim).
           3D for keys, 4D for queries.
        cos: Cosine table from Glm5RotaryEmbedding.
        sin: Sine table from Glm5RotaryEmbedding.
        position_ids: (bsz, seq_len) or (1, seq_len).

    Returns:
        Tensor with RoPE applied, same shape as x.
    """
    cos_half = cos.chunk(2, dim=-1)[0][position_ids]  # (bsz, seq_len, dim//2)
    sin_half = sin.chunk(2, dim=-1)[0][position_ids]  # (bsz, seq_len, dim//2)
    cos_full = torch.cat([cos_half, cos_half], dim=-1)[0]  # (seq_len, dim)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)[0]  # (seq_len, dim)

    # Add singleton dims for broadcasting with multi-head inputs (BSHD layout)
    for _ in range(x.dim() - 3):
        cos_full = cos_full.unsqueeze(-2)
        sin_full = sin_full.unsqueeze(-2)

    result = (x * cos_full) + (rotate_half(x) * sin_full)
    return result.to(x.dtype)
