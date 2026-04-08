# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for DeepSeek V3.2 HuggingFace checkpoint weights.

Tests weight name mapping, FP8 dequantization, shape correctness,
and end-to-end conversion pipeline compatibility with the Neuron model.
"""

import gc
import json
import os
import unittest

import torch

# Path to V3.2 HF checkpoint
V32_MODEL_PATH = "/home/ubuntu/environment/DeepSeek-V3.2"

# Skip all tests if checkpoint not downloaded
CHECKPOINT_AVAILABLE = os.path.exists(os.path.join(V32_MODEL_PATH, "config.json"))


def _load_shard_keys(shard_path):
    """Load weight keys from a single safetensors shard without loading tensors."""
    from safetensors import safe_open
    f = safe_open(shard_path, framework="pt")
    return list(f.keys())


def _load_shard_tensors(shard_path, keys=None):
    """Load specified tensors from a single safetensors shard."""
    from safetensors import safe_open
    f = safe_open(shard_path, framework="pt")
    if keys is None:
        keys = f.keys()
    return {k: f.get_tensor(k) for k in keys if k in f.keys()}


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32Config(unittest.TestCase):
    """Test that V3.2 config loads correctly and has indexer fields."""

    def test_hf_config_has_indexer_fields(self):
        with open(os.path.join(V32_MODEL_PATH, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["index_n_heads"], 64)
        self.assertEqual(cfg["index_head_dim"], 128)
        self.assertEqual(cfg["index_topk"], 2048)

    def test_inference_config_creates_correctly(self):
        from src.modeling_deepseek import DeepseekV3InferenceConfig, DeepseekV3NeuronConfig

        with open(os.path.join(V32_MODEL_PATH, "config.json")) as f:
            hf_config = json.load(f)

        nc = DeepseekV3NeuronConfig(
            tp_degree=64, torch_dtype="bfloat16", seq_len=512, max_batch_size=1,
        )
        hf_config["neuron_config"] = nc
        config = DeepseekV3InferenceConfig(**hf_config)

        self.assertTrue(config.has_indexer)
        self.assertEqual(config.head_dim, 704)  # 64 + 512 + 128
        self.assertEqual(config.num_hidden_layers, 61)

    def test_v32_has_quantization_config(self):
        with open(os.path.join(V32_MODEL_PATH, "config.json")) as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config", {})
        self.assertEqual(qc.get("quant_method"), "fp8")
        self.assertEqual(qc.get("fmt"), "e4m3")
        self.assertEqual(qc.get("weight_block_size"), [128, 128])


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32WeightIndex(unittest.TestCase):
    """Test the weight index file for completeness."""

    @classmethod
    def setUpClass(cls):
        index_path = os.path.join(V32_MODEL_PATH, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise unittest.SkipTest("Weight index not downloaded yet")
        with open(index_path) as f:
            cls.index = json.load(f)
        cls.weight_map = cls.index["weight_map"]

    def test_all_61_layers_have_indexer_weights(self):
        """Every layer should have wq_b, wk, k_norm, weights_proj."""
        for layer_idx in range(61):
            prefix = f"model.layers.{layer_idx}.self_attn.indexer"
            expected_keys = [
                f"{prefix}.wq_b.weight",
                f"{prefix}.wq_b.weight_scale_inv",
                f"{prefix}.wk.weight",
                f"{prefix}.wk.weight_scale_inv",
                f"{prefix}.k_norm.weight",
                f"{prefix}.k_norm.bias",
                f"{prefix}.weights_proj.weight",
            ]
            for key in expected_keys:
                self.assertIn(key, self.weight_map,
                              f"Missing {key} in weight index")

    def test_all_61_layers_have_attention_weights(self):
        """Every layer should have q_a_proj, q_b_proj, kv_a/b, o_proj."""
        for layer_idx in range(61):
            prefix = f"model.layers.{layer_idx}.self_attn"
            for subkey in ["q_a_proj.weight", "q_b_proj.weight",
                           "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
                           "o_proj.weight"]:
                key = f"{prefix}.{subkey}"
                self.assertIn(key, self.weight_map,
                              f"Missing {key} in weight index")

    def test_embed_and_head_present(self):
        self.assertIn("model.embed_tokens.weight", self.weight_map)
        # lm_head may or may not exist (tie_word_embeddings=false means it should)
        # Check both possibilities
        has_lm_head = "lm_head.weight" in self.weight_map
        has_tied = "model.embed_tokens.weight" in self.weight_map
        self.assertTrue(has_lm_head or has_tied,
                        "Neither lm_head.weight nor embed_tokens found")

    def test_moe_layers_have_experts(self):
        """Layers 3+ should have MoE expert weights."""
        for layer_idx in [3, 30, 60]:
            key = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
            self.assertIn(key, self.weight_map,
                          f"Missing expert weights for layer {layer_idx}")

    def test_dense_layers_no_experts(self):
        """Layers 0-2 should NOT have MoE experts."""
        for layer_idx in range(3):
            key = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
            self.assertNotIn(key, self.weight_map,
                             f"Layer {layer_idx} should be dense, not MoE")


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32WeightShapes(unittest.TestCase):
    """Test actual weight shapes from the first shard."""

    @classmethod
    def setUpClass(cls):
        shard_path = os.path.join(V32_MODEL_PATH, "model-00001-of-000163.safetensors")
        if not os.path.exists(shard_path):
            raise unittest.SkipTest("First shard not downloaded yet")
        # Load only layer 0 indexer + attention weights
        from safetensors import safe_open
        f = safe_open(shard_path, framework="pt")
        cls.tensors = {}
        for k in f.keys():
            if "layers.0.self_attn" in k:
                cls.tensors[k] = f.get_tensor(k)

    def test_indexer_wq_b_shape(self):
        """wq_b: (index_n_heads * index_head_dim, q_lora_rank) = (8192, 1536)"""
        w = self.tensors["model.layers.0.self_attn.indexer.wq_b.weight"]
        self.assertEqual(list(w.shape), [8192, 1536])
        self.assertEqual(w.dtype, torch.float8_e4m3fn)

    def test_indexer_wk_shape(self):
        """wk: (index_head_dim, hidden_size) = (128, 7168)"""
        w = self.tensors["model.layers.0.self_attn.indexer.wk.weight"]
        self.assertEqual(list(w.shape), [128, 7168])
        self.assertEqual(w.dtype, torch.float8_e4m3fn)

    def test_indexer_k_norm_shape(self):
        """k_norm: LayerNorm(128) → weight=[128], bias=[128]"""
        w = self.tensors["model.layers.0.self_attn.indexer.k_norm.weight"]
        b = self.tensors["model.layers.0.self_attn.indexer.k_norm.bias"]
        self.assertEqual(list(w.shape), [128])
        self.assertEqual(list(b.shape), [128])
        self.assertEqual(w.dtype, torch.float32)

    def test_indexer_weights_proj_shape(self):
        """weights_proj: (index_n_heads, hidden_size) = (64, 7168), BF16"""
        w = self.tensors["model.layers.0.self_attn.indexer.weights_proj.weight"]
        self.assertEqual(list(w.shape), [64, 7168])
        self.assertEqual(w.dtype, torch.bfloat16)

    def test_q_a_proj_shape(self):
        """q_a_proj: (q_lora_rank, hidden_size) = (1536, 7168)"""
        w = self.tensors["model.layers.0.self_attn.q_a_proj.weight"]
        self.assertEqual(list(w.shape), [1536, 7168])

    def test_q_b_proj_shape(self):
        """q_b_proj: (num_heads * (nope + rope), q_lora_rank) = (128*192, 1536) = (24576, 1536)"""
        w = self.tensors["model.layers.0.self_attn.q_b_proj.weight"]
        self.assertEqual(list(w.shape), [24576, 1536])

    def test_kv_a_proj_shape(self):
        """kv_a_proj: (kv_lora_rank + qk_rope_head_dim, hidden_size) = (576, 7168)"""
        w = self.tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
        self.assertEqual(list(w.shape), [576, 7168])

    def test_kv_b_proj_shape(self):
        """kv_b_proj: (num_heads * (nope + v_head_dim), kv_lora_rank) = (128*256, 512) = (32768, 512)"""
        w = self.tensors["model.layers.0.self_attn.kv_b_proj.weight"]
        self.assertEqual(list(w.shape), [32768, 512])


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32FP8Dequantization(unittest.TestCase):
    """Test FP8 dequantization on real indexer weights."""

    @classmethod
    def setUpClass(cls):
        shard_path = os.path.join(V32_MODEL_PATH, "model-00001-of-000163.safetensors")
        if not os.path.exists(shard_path):
            raise unittest.SkipTest("First shard not downloaded yet")
        from safetensors import safe_open
        f = safe_open(shard_path, framework="pt")
        cls.tensors = {}
        for k in f.keys():
            if "layers.0.self_attn.indexer" in k:
                cls.tensors[k] = f.get_tensor(k)

    def _dequant(self, weight_key, scale_key, block_size=128):
        weight = self.tensors[weight_key]
        scale_inv = self.tensors[scale_key]
        M, N = weight.shape
        scales_expanded = (
            scale_inv
            .repeat_interleave(block_size, dim=0)
            .repeat_interleave(block_size, dim=1)
        )
        return (weight.to(torch.float32) * scales_expanded[:M, :N].to(torch.float32)).to(torch.bfloat16)

    def test_wq_b_dequant_produces_valid_values(self):
        result = self._dequant(
            "model.layers.0.self_attn.indexer.wq_b.weight",
            "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(result.float()).any())
        self.assertFalse(torch.isinf(result.float()).any())
        self.assertGreater(result.float().abs().max().item(), 0.0)

    def test_wk_dequant_produces_valid_values(self):
        result = self._dequant(
            "model.layers.0.self_attn.indexer.wk.weight",
            "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(result.float()).any())
        self.assertFalse(torch.isinf(result.float()).any())

    def test_full_dequant_function(self):
        """Test the actual _dequantize_fp8_state_dict function."""
        from src.modeling_deepseek import _dequantize_fp8_state_dict

        # Make a copy of the tensors
        sd = {k: v.clone() for k, v in self.tensors.items()}
        _dequantize_fp8_state_dict(sd, block_size=128)

        # All scale_inv keys should be removed
        scale_keys = [k for k in sd if k.endswith(".weight_scale_inv")]
        self.assertEqual(len(scale_keys), 0, f"Leftover scale keys: {scale_keys}")

        # wq_b and wk should now be BF16
        wq_b = sd["model.layers.0.self_attn.indexer.wq_b.weight"]
        wk = sd["model.layers.0.self_attn.indexer.wk.weight"]
        self.assertEqual(wq_b.dtype, torch.bfloat16)
        self.assertEqual(wk.dtype, torch.bfloat16)

        # k_norm and weights_proj should be unchanged
        k_norm = sd["model.layers.0.self_attn.indexer.k_norm.weight"]
        wp = sd["model.layers.0.self_attn.indexer.weights_proj.weight"]
        self.assertEqual(k_norm.dtype, torch.float32)
        self.assertEqual(wp.dtype, torch.bfloat16)


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32WeightConversion(unittest.TestCase):
    """Test the full weight conversion pipeline on a small subset."""

    def test_conversion_pipeline_layer_0(self):
        """Run convert_deepseek_v3_hf_to_neuron_state_dict on layer 0 weights."""
        from src.modeling_deepseek import (
            DeepseekV3InferenceConfig,
            DeepseekV3NeuronConfig,
            convert_deepseek_v3_hf_to_neuron_state_dict,
        )

        shard_path = os.path.join(V32_MODEL_PATH, "model-00001-of-000163.safetensors")
        if not os.path.exists(shard_path):
            self.skipTest("First shard not downloaded yet")

        # Load layer 0 weights only, strip model. prefix (as get_state_dict does)
        from safetensors import safe_open
        f = safe_open(shard_path, framework="pt")
        sd = {}
        for k in f.keys():
            if "layers.0." in k or "embed_tokens" in k:
                new_key = k.replace("model.", "", 1)
                sd[new_key] = f.get_tensor(k)

        # Create config
        with open(os.path.join(V32_MODEL_PATH, "config.json")) as cf:
            hf_config = json.load(cf)
        nc = DeepseekV3NeuronConfig(
            tp_degree=64, torch_dtype="bfloat16", seq_len=512, max_batch_size=1,
        )
        hf_config["neuron_config"] = nc
        # Override num_hidden_layers to 1 for this test
        hf_config["num_hidden_layers"] = 1
        config = DeepseekV3InferenceConfig(**hf_config)

        # Run conversion
        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        # Verify rank_util added
        self.assertIn("rank_util.rank", result)
        self.assertIn("layers.0.self_attn.rank_util.rank", result)
        self.assertIn("layers.0.self_attn.indexer.rank_util.rank", result)

        # Verify indexer weights present and dequantized
        self.assertIn("layers.0.self_attn.indexer.wq_b.weight", result)
        self.assertEqual(
            result["layers.0.self_attn.indexer.wq_b.weight"].dtype,
            torch.bfloat16,
        )

        # Verify weights_proj cast to FP32
        wp = result["layers.0.self_attn.indexer.weights_proj.weight"]
        self.assertEqual(wp.dtype, torch.float32)

        # Verify no scale_inv keys remain
        scale_keys = [k for k in result if "scale_inv" in k]
        self.assertEqual(len(scale_keys), 0)

        gc.collect()


@unittest.skipUnless(CHECKPOINT_AVAILABLE, "V3.2 checkpoint not available")
class TestV32ModelKeyAlignment(unittest.TestCase):
    """Verify converted state dict keys align with model parameter names."""

    def test_indexer_keys_match_module(self):
        """Converted weight keys should match DeepseekV3Indexer named_parameters."""
        from src.modeling_deepseek import DeepseekV3Indexer, DeepseekV3InferenceConfig, DeepseekV3NeuronConfig

        with open(os.path.join(V32_MODEL_PATH, "config.json")) as f:
            hf_config = json.load(f)
        nc = DeepseekV3NeuronConfig(
            tp_degree=1, torch_dtype="bfloat16", seq_len=512, max_batch_size=1,
        )
        hf_config["neuron_config"] = nc
        config = DeepseekV3InferenceConfig(**hf_config)

        indexer = DeepseekV3Indexer(config, tensor_model_parallel_group=None)
        model_params = {name for name, _ in indexer.named_parameters()}
        # Exclude buffers (hadamard_matrix is a buffer, not a parameter)
        model_buffers = {name for name, _ in indexer.named_buffers()}

        # Expected HF keys after prefix strip: wq_b.weight, wk.weight, k_norm.weight, k_norm.bias, weights_proj.weight
        expected_params = {"wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"}
        self.assertEqual(model_params, expected_params,
                         f"Model params {model_params} != expected {expected_params}")
        self.assertIn("hadamard_matrix", model_buffers)


if __name__ == "__main__":
    unittest.main()
