# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-5.1 state dict conversion.

CPU-only tests validating the HF -> Neuron weight transformation logic.
"""

import unittest
from unittest.mock import MagicMock

import torch

from src.modeling_glm5 import convert_glm5_hf_to_neuron_state_dict


def _make_mock_config(num_layers=4, tp_degree=2, first_k_dense=1, num_experts=8):
    """Create a mock config for testing weight conversion."""
    config = MagicMock()
    config.num_hidden_layers = num_layers
    config.num_local_experts = num_experts
    config.first_k_dense_replace = first_k_dense
    config.has_indexer = True
    config.neuron_config = MagicMock()
    config.neuron_config.tp_degree = tp_degree
    config.quantization_config = None
    config._load_config = None
    return config


class TestModelPrefixStripping(unittest.TestCase):
    """Test that 'model.' prefix is stripped from HF keys."""

    def test_model_prefix_stripped(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(100, 64),
        }
        config = _make_mock_config(num_layers=2, first_k_dense=2)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("embed_tokens.weight", result)
        self.assertIn("layers.0.input_layernorm.weight", result)
        self.assertIn("norm.weight", result)
        self.assertIn("lm_head.weight", result)
        self.assertNotIn("model.embed_tokens.weight", result)


class TestMTPLayerDropped(unittest.TestCase):
    """Test that MTP layer (layer N where N=num_hidden_layers) is dropped."""

    def test_extra_layer_dropped(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.layers.1.input_layernorm.weight": torch.randn(64),
            "model.layers.2.input_layernorm.weight": torch.randn(64),  # MTP layer
        }
        config = _make_mock_config(num_layers=2, first_k_dense=2)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("layers.0.input_layernorm.weight", result)
        self.assertIn("layers.1.input_layernorm.weight", result)
        self.assertNotIn("layers.2.input_layernorm.weight", result)


class TestTiedEmbeddings(unittest.TestCase):
    """Test handling of tied embeddings."""

    def test_lm_head_copied_from_embed(self):
        embed_weight = torch.randn(100, 64)
        state_dict = {
            "model.embed_tokens.weight": embed_weight,
            "model.norm.weight": torch.randn(64),
        }
        config = _make_mock_config(num_layers=0, first_k_dense=0)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("lm_head.weight", result)
        torch.testing.assert_close(result["lm_head.weight"], embed_weight)

    def test_explicit_lm_head_preserved(self):
        lm_head_weight = torch.randn(100, 64)
        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": lm_head_weight,
        }
        config = _make_mock_config(num_layers=0, first_k_dense=0)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        torch.testing.assert_close(result["lm_head.weight"], lm_head_weight)


class TestRankUtilities(unittest.TestCase):
    """Test rank utility tensor injection."""

    def test_global_rank_util(self):
        state_dict = {"model.norm.weight": torch.randn(64)}
        config = _make_mock_config(num_layers=2, tp_degree=4, first_k_dense=2)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("rank_util.rank", result)
        torch.testing.assert_close(
            result["rank_util.rank"],
            torch.arange(0, 4, dtype=torch.int32),
        )

    def test_per_layer_attn_rank_util(self):
        state_dict = {"model.norm.weight": torch.randn(64)}
        config = _make_mock_config(num_layers=2, tp_degree=4, first_k_dense=2)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        for i in range(2):
            key = f"layers.{i}.self_attn.rank_util.rank"
            self.assertIn(key, result)
            torch.testing.assert_close(
                result[key], torch.arange(0, 4, dtype=torch.int32),
            )

    def test_per_layer_indexer_rank_util(self):
        state_dict = {"model.norm.weight": torch.randn(64)}
        config = _make_mock_config(num_layers=2, tp_degree=4, first_k_dense=2)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        for i in range(2):
            key = f"layers.{i}.self_attn.indexer.rank_util.rank"
            self.assertIn(key, result)


class TestRouterRenaming(unittest.TestCase):
    """Test router weight renaming for MoE layers."""

    def test_gate_weight_renamed(self):
        state_dict = {
            "model.layers.1.mlp.gate.weight": torch.randn(8, 64),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(8),
        }
        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=8)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("layers.1.mlp.router.linear_router.weight", result)
        self.assertNotIn("layers.1.mlp.gate.weight", result)

    def test_bias_renamed(self):
        state_dict = {
            "model.layers.1.mlp.gate.weight": torch.randn(8, 64),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(8),
        }
        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=8)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("layers.1.mlp.router.e_score_correction_bias", result)
        self.assertNotIn("layers.1.mlp.gate.e_score_correction_bias", result)

    def test_dense_layer_skipped(self):
        """Dense layers should not have router weight renaming."""
        state_dict = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        }
        config = _make_mock_config(num_layers=2, first_k_dense=1)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        self.assertIn("layers.0.mlp.gate_proj.weight", result)


class TestExpertWeightConversion(unittest.TestCase):
    """Test expert weight renaming and transposition."""

    def test_gate_up_proj_transposed_and_renamed(self):
        """HF [E, 2*inter, hidden] -> NXDI [E, hidden, 2*inter]."""
        E, inter, hidden = 4, 16, 32
        gate_up = torch.randn(E, 2 * inter, hidden)
        state_dict = {
            "model.layers.1.mlp.experts.gate_up_proj": gate_up,
        }
        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=E)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        nxdi_key = "layers.1.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        self.assertIn(nxdi_key, result)
        self.assertEqual(result[nxdi_key].shape, (E, hidden, 2 * inter))
        torch.testing.assert_close(result[nxdi_key], gate_up.transpose(1, 2).contiguous())

    def test_down_proj_transposed_and_renamed(self):
        """HF [E, hidden, inter] -> NXDI [E, inter, hidden]."""
        E, inter, hidden = 4, 16, 32
        down = torch.randn(E, hidden, inter)
        state_dict = {
            "model.layers.1.mlp.experts.down_proj": down,
        }
        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=E)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        nxdi_key = "layers.1.mlp.expert_mlps.mlp_op.down_proj.weight"
        self.assertIn(nxdi_key, result)
        self.assertEqual(result[nxdi_key].shape, (E, inter, hidden))


class TestPerExpertWeightConversion(unittest.TestCase):
    """Test per-expert 2D weight format (FP8 checkpoint after dequantization)."""

    def test_per_expert_gate_up_fused_and_stacked(self):
        """Per-expert gate+up [inter, hidden] -> NXDI [E, hidden, 2*inter]."""
        E, inter, hidden = 4, 16, 32
        gates = [torch.randn(inter, hidden) for _ in range(E)]
        ups = [torch.randn(inter, hidden) for _ in range(E)]
        downs = [torch.randn(hidden, inter) for _ in range(E)]

        state_dict = {}
        for e in range(E):
            state_dict[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = gates[e]
            state_dict[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = ups[e]
            state_dict[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = downs[e]

        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=E)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        gate_up_key = "layers.1.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        self.assertIn(gate_up_key, result)
        self.assertEqual(result[gate_up_key].shape, (E, hidden, 2 * inter))

        # Verify content: for each expert, columns should be [gate | up] transposed
        for e in range(E):
            expected = torch.cat([gates[e], ups[e]], dim=0).T  # [hidden, 2*inter]
            torch.testing.assert_close(result[gate_up_key][e], expected)

    def test_per_expert_down_stacked(self):
        """Per-expert down [hidden, inter] -> NXDI [E, inter, hidden]."""
        E, inter, hidden = 4, 16, 32
        downs = [torch.randn(hidden, inter) for _ in range(E)]

        state_dict = {}
        for e in range(E):
            state_dict[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
            state_dict[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
            state_dict[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = downs[e]

        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=E)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        down_key = "layers.1.mlp.expert_mlps.mlp_op.down_proj.weight"
        self.assertIn(down_key, result)
        self.assertEqual(result[down_key].shape, (E, inter, hidden))

        for e in range(E):
            torch.testing.assert_close(result[down_key][e], downs[e].T)

    def test_per_expert_keys_removed(self):
        """Original per-expert keys should be removed after conversion."""
        E, inter, hidden = 4, 16, 32
        state_dict = {}
        for e in range(E):
            state_dict[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
            state_dict[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
            state_dict[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, inter)

        config = _make_mock_config(num_layers=2, first_k_dense=1, num_experts=E)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        for e in range(E):
            self.assertNotIn(f"layers.1.mlp.experts.{e}.gate_proj.weight", result)
            self.assertNotIn(f"layers.1.mlp.experts.{e}.up_proj.weight", result)
            self.assertNotIn(f"layers.1.mlp.experts.{e}.down_proj.weight", result)


class TestIndexerWeightCasting(unittest.TestCase):
    """Test that indexer weights_proj is cast to FP32."""

    def test_weights_proj_cast_to_fp32(self):
        state_dict = {
            "model.layers.0.self_attn.indexer.weights_proj.weight": torch.randn(4, 64, dtype=torch.bfloat16),
        }
        config = _make_mock_config(num_layers=1, first_k_dense=1)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        key = "layers.0.self_attn.indexer.weights_proj.weight"
        self.assertEqual(result[key].dtype, torch.float32)

    def test_weights_proj_already_fp32(self):
        state_dict = {
            "model.layers.0.self_attn.indexer.weights_proj.weight": torch.randn(4, 64, dtype=torch.float32),
        }
        config = _make_mock_config(num_layers=1, first_k_dense=1)
        result = convert_glm5_hf_to_neuron_state_dict(state_dict, config)

        key = "layers.0.self_attn.indexer.weights_proj.weight"
        self.assertEqual(result[key].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
