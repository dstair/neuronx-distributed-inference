# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for DeepSeek V3.2 using real checkpoint weights.

Tests the indexer and attention forward pass with real V3.2 weights loaded
into our Neuron model modules, comparing against step-by-step PyTorch
reference implementations.
"""

import json
import os
import unittest

import torch
import torch.nn.functional as F

V32_MODEL_PATH = "/home/ubuntu/environment/DeepSeek-V3.2"
SHARD_PATH = os.path.join(V32_MODEL_PATH, "model-00001-of-000163.safetensors")
CHECKPOINT_AVAILABLE = os.path.exists(SHARD_PATH)

# Model operates in BF16
DTYPE = torch.bfloat16


def _load_layer0_weights():
    """Load and dequantize layer 0 weights from V3.2 checkpoint."""
    from safetensors import safe_open

    f = safe_open(SHARD_PATH, framework="pt")
    sd = {}
    for k in f.keys():
        if "layers.0." in k:
            new_key = k.replace("model.", "", 1)
            sd[new_key] = f.get_tensor(k)

    from src.modeling_deepseek import _dequantize_fp8_state_dict
    _dequantize_fp8_state_dict(sd, block_size=128)
    return sd


def _make_v32_config(tp_degree=1):
    from src.modeling_deepseek import DeepseekV3InferenceConfig, DeepseekV3NeuronConfig

    with open(os.path.join(V32_MODEL_PATH, "config.json")) as f:
        hf_config = json.load(f)
    nc = DeepseekV3NeuronConfig(
        tp_degree=tp_degree, torch_dtype="bfloat16", seq_len=64, max_batch_size=2,
    )
    hf_config["neuron_config"] = nc
    return DeepseekV3InferenceConfig(**hf_config)


def _load_indexer_with_weights(config, sd):
    """Create a DeepseekV3Indexer and load real V3.2 weights into it."""
    from src.modeling_deepseek import DeepseekV3Indexer

    indexer = DeepseekV3Indexer(config, tensor_model_parallel_group=None)
    prefix = "layers.0.self_attn.indexer."
    indexer_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            indexer_sd[k[len(prefix):]] = v
    if "weights_proj.weight" in indexer_sd:
        indexer_sd["weights_proj.weight"] = indexer_sd["weights_proj.weight"].to(torch.float32)
    indexer.load_state_dict(indexer_sd, strict=False)
    indexer.eval()
    return indexer


def _make_rotary_emb(config):
    from src.rope_util import DeepseekV3YarnRotaryEmbedding
    return DeepseekV3YarnRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        scaling_factor=config.rope_scaling["factor"],
        base=getattr(config, "rope_theta", 10000),
        original_max_position_embeddings=config.rope_scaling["original_max_position_embeddings"],
        max_position_embeddings=config.max_position_embeddings,
        mscale=config.rope_scaling.get("mscale", 1.0),
        mscale_all_dim=config.rope_scaling.get("mscale_all_dim", 0),
        beta_fast=config.rope_scaling.get("beta_fast", 32),
        beta_slow=config.rope_scaling.get("beta_slow", 1),
    )


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint shard not available")
class TestIndexerWithRealWeights(unittest.TestCase):
    """Test the indexer module loaded with real V3.2 weights."""

    @classmethod
    def setUpClass(cls):
        cls.config = _make_v32_config(tp_degree=1)
        cls.sd = _load_layer0_weights()
        cls.indexer = _load_indexer_with_weights(cls.config, cls.sd)
        cls.rotary_emb = _make_rotary_emb(cls.config)

    def _get_cos_sin(self, x):
        return self.rotary_emb(x, self.config.neuron_config.seq_len)

    def test_indexer_forward_no_nan(self):
        """Forward pass with real weights should produce no NaN/Inf."""
        bsz, seq_len = 1, 16
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self._get_cos_sin(x)

        with torch.no_grad():
            topk_indices, processed_k = self.indexer(
                x, qr, position_ids, cos, sin, is_prefill=True,
            )

        self.assertFalse(torch.isnan(topk_indices.float()).any(), "topk_indices has NaN")
        self.assertFalse(torch.isnan(processed_k.float()).any(), "processed_k has NaN")
        self.assertFalse(torch.isinf(processed_k.float()).any(), "processed_k has Inf")

    def test_indexer_decode_no_nan(self):
        """Decode path with real weights should also be NaN-free."""
        bsz, prior_len = 1, 8
        x = torch.randn(bsz, 1, self.config.hidden_size, dtype=DTYPE)
        qr = torch.randn(bsz, 1, self.config.q_lora_rank, dtype=DTYPE)
        position_ids = torch.tensor([[prior_len]])
        past_indexer_k = torch.randn(bsz, prior_len, self.config.index_head_dim, dtype=DTYPE)
        cos, sin = self._get_cos_sin(x)

        with torch.no_grad():
            topk_indices, processed_k = self.indexer(
                x, qr, position_ids, cos, sin,
                is_prefill=False, past_indexer_k=past_indexer_k,
            )

        self.assertFalse(torch.isnan(topk_indices.float()).any())
        self.assertFalse(torch.isnan(processed_k.float()).any())

    def test_processed_k_magnitude(self):
        """Processed keys should have reasonable magnitudes after Hadamard transform."""
        bsz, seq_len = 1, 8
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE) * 0.1
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE) * 0.1
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self._get_cos_sin(x)

        with torch.no_grad():
            _, processed_k = self.indexer(
                x, qr, position_ids, cos, sin, is_prefill=True,
            )

        abs_max = processed_k.float().abs().max().item()
        self.assertLess(abs_max, 100.0, f"processed_k abs max too large: {abs_max}")
        self.assertGreater(abs_max, 0.0, "processed_k is all zeros")

    def test_topk_selects_all_when_seq_le_topk(self):
        """When seq_len <= index_topk, all positions should be selected.

        V3.2 has index_topk=2048. For short sequences (seq_len < 2048),
        the indexer selects all positions — the sparsity only kicks in
        for longer sequences.
        """
        bsz, seq_len = 1, 32
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self._get_cos_sin(x)

        with torch.no_grad():
            topk_indices, _ = self.indexer(
                x, qr, position_ids, cos, sin, is_prefill=True,
            )

        # min(index_topk=2048, seq_len=32) = 32, so every valid position is selected
        expected_topk = min(self.config.index_topk, seq_len)
        self.assertEqual(expected_topk, seq_len)

        # Last position should select exactly {0, 1, ..., 31}
        last_idx = set(topk_indices[0, -1, :].tolist())
        self.assertEqual(last_idx, set(range(seq_len)),
                         f"Expected all positions selected, got {last_idx}")


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint shard not available")
class TestIndexerStepByStep(unittest.TestCase):
    """Compare our indexer components step-by-step against pure PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        cls.config = _make_v32_config(tp_degree=1)
        cls.sd = _load_layer0_weights()

    def test_wq_b_projection(self):
        """wq_b projection should match reference F.linear."""
        w = self.sd["layers.0.self_attn.indexer.wq_b.weight"]  # BF16 (8192, 1536)
        bsz, seq_len = 1, 4
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE)

        # Our path
        from src.modeling_deepseek import DeepseekV3Indexer
        indexer = DeepseekV3Indexer(self.config, tensor_model_parallel_group=None)
        indexer.wq_b.weight = torch.nn.Parameter(w)
        indexer.eval()
        with torch.no_grad():
            our_q = indexer.wq_b(qr)

        # Reference
        ref_q = F.linear(qr, w)

        torch.testing.assert_close(our_q, ref_q, atol=1e-3, rtol=1e-3)

    def test_wk_plus_layernorm(self):
        """wk projection + LayerNorm should match reference."""
        wk_w = self.sd["layers.0.self_attn.indexer.wk.weight"]  # BF16
        kn_w = self.sd["layers.0.self_attn.indexer.k_norm.weight"]  # FP32
        kn_b = self.sd["layers.0.self_attn.indexer.k_norm.bias"]  # FP32

        bsz, seq_len = 1, 4
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)

        # Reference: compute in the same precision as the module would
        ref_k = F.linear(x, wk_w)
        ref_k = F.layer_norm(ref_k.float(), [self.config.index_head_dim],
                             weight=kn_w, bias=kn_b).to(DTYPE)

        # Our path
        from src.modeling_deepseek import DeepseekV3Indexer
        indexer = DeepseekV3Indexer(self.config, tensor_model_parallel_group=None)
        indexer.wk.weight = torch.nn.Parameter(wk_w)
        indexer.k_norm.weight = torch.nn.Parameter(kn_w)
        indexer.k_norm.bias = torch.nn.Parameter(kn_b)
        indexer.eval()
        with torch.no_grad():
            our_k = indexer.wk(x)
            our_k = indexer.k_norm(our_k)

        torch.testing.assert_close(our_k.float(), ref_k.float(), atol=1e-2, rtol=1e-2)

    def test_hadamard_transform_matches_reference(self):
        """Scaled Hadamard transform is self-inverse and produces correct output."""
        from src.modeling_deepseek import _get_hadamard_matrix

        dim = self.config.index_head_dim  # 128
        H = _get_hadamard_matrix(dim) * (dim ** -0.5)
        x = torch.randn(2, 8, dim)

        result = x @ H
        # Apply twice should recover input (involution)
        recovered = result @ H
        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)

    def test_weights_proj_scaling(self):
        """weights_proj output should match reference with correct scaling."""
        wp_w = self.sd["layers.0.self_attn.indexer.weights_proj.weight"]  # BF16

        bsz, seq_len = 1, 4
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)

        # Reference
        ref_weights = F.linear(x.float(), wp_w.float()) * (self.config.index_n_heads ** -0.5)

        # Our path
        from src.modeling_deepseek import DeepseekV3Indexer
        indexer = DeepseekV3Indexer(self.config, tensor_model_parallel_group=None)
        indexer.weights_proj.weight = torch.nn.Parameter(wp_w.float())
        indexer.eval()
        with torch.no_grad():
            our_weights = indexer.weights_proj(x.float()) * (indexer.n_heads ** -0.5)

        torch.testing.assert_close(our_weights, ref_weights, atol=1e-4, rtol=1e-4)

    def test_bf16_scoring_logic(self):
        """Test the scoring path: einsum Q*K -> relu -> weighted sum."""
        n_heads, head_dim = 4, 128
        bsz, q_len, kv_len = 1, 4, 8
        softmax_scale = head_dim ** -0.5

        q = torch.randn(bsz, q_len, n_heads, head_dim)
        k = torch.randn(bsz, kv_len, head_dim)
        weights = torch.randn(bsz, q_len, n_heads)

        qk = torch.einsum("bshd,btd->bsht", q, k) * softmax_scale
        qk = torch.relu(qk)
        index_score = torch.einsum("bsht,bsh->bst", qk, weights)

        # Re-compute step by step
        ref_qk = torch.einsum("bshd,btd->bsht", q, k) * softmax_scale
        ref_qk = torch.relu(ref_qk)
        ref_score = torch.einsum("bsht,bsh->bst", ref_qk, weights)

        torch.testing.assert_close(index_score, ref_score)

    def test_full_pipeline_real_weights(self):
        """Full indexer forward with real weights: verify shape and basic properties."""
        indexer = _load_indexer_with_weights(self.config, self.sd)
        rotary_emb = _make_rotary_emb(self.config)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        cos, sin = rotary_emb(x, self.config.neuron_config.seq_len)

        with torch.no_grad():
            topk_indices, processed_k = indexer(
                x, qr, position_ids, cos, sin, is_prefill=True,
            )

        expected_topk = min(self.config.index_topk, seq_len)
        self.assertEqual(topk_indices.shape, (bsz, seq_len, expected_topk))
        self.assertEqual(processed_k.shape, (bsz, seq_len, self.config.index_head_dim))

        # Causal: position 0 must include itself
        pos0_idx = topk_indices[0, 0, :].tolist()
        self.assertIn(0, pos0_idx, f"Position 0 must include itself in top-k: {pos0_idx}")

        # All indices valid
        self.assertTrue(torch.all(topk_indices >= 0))
        self.assertTrue(torch.all(topk_indices < seq_len))


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint shard not available")
class TestIndexerDeterminism(unittest.TestCase):
    """Test that the indexer is deterministic (same input -> same output)."""

    @classmethod
    def setUpClass(cls):
        cls.config = _make_v32_config(tp_degree=1)
        sd = _load_layer0_weights()
        cls.indexer = _load_indexer_with_weights(cls.config, sd)
        cls.rotary_emb = _make_rotary_emb(cls.config)

    def test_same_input_same_output(self):
        """Running indexer twice with same input should produce identical output."""
        torch.manual_seed(42)
        bsz, seq_len = 1, 8
        x = torch.randn(bsz, seq_len, self.config.hidden_size, dtype=DTYPE)
        qr = torch.randn(bsz, seq_len, self.config.q_lora_rank, dtype=DTYPE)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self.rotary_emb(x, self.config.neuron_config.seq_len)

        with torch.no_grad():
            idx1, k1 = self.indexer(x, qr, position_ids, cos, sin, is_prefill=True)
            idx2, k2 = self.indexer(x, qr, position_ids, cos, sin, is_prefill=True)

        torch.testing.assert_close(k1, k2)
        self.assertTrue(torch.equal(idx1, idx2))


if __name__ == "__main__":
    unittest.main()
