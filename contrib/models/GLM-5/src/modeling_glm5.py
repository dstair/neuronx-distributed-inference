# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.1 (754B MoE with MLA + DSA) implementation for NeuronX Distributed Inference.
#
# Architecture adapted from the HuggingFace transformers GlmMoeDsa implementation
# (transformers v5.4.0+) and the DeepSeek-V3 NXDI contrib implementation.
#
# Key architectural features:
#   - Multi-head Latent Attention (MLA) with LoRA-compressed Q/KV
#   - Dynamic Sparse Attention (DSA) indexer on every layer
#   - MoE: 256 routed experts + 1 shared expert, top-8 selection
#   - Sigmoid routing with e_score_correction_bias, routed_scaling_factor=2.5
#   - 3 dense layers + 75 MoE layers (78 total)
#   - Standard Llama/NeoX split-half RoPE (theta=1M, no YaRN)

import gc
import logging
from typing import List, Optional, Tuple, Type

import warnings
import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region
from neuronx_distributed.utils import cpu_mode
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from src.rope_util import (
    Glm5RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_bshd,
)
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger(__name__)


# Register glm_moe_dsa with AutoConfig so load_pretrained_config can load it.
# The installed transformers may not include this model type yet.
class _GlmMoeDsaConfig(PretrainedConfig):
    model_type = "glm_moe_dsa"

try:
    AutoConfig.register("glm_moe_dsa", _GlmMoeDsaConfig)
except ValueError:
    pass  # Already registered


# ---------------------------------------------------------------------------
# FP8 dequantization (reused from DeepSeek-V3 — same block-wise format)
# ---------------------------------------------------------------------------

def _dequantize_fp8_state_dict(state_dict: dict, block_size: int = 128) -> dict:
    """
    Dequantize FP8 block-wise weights to BF16 in-place.

    GLM-5.1-FP8 stores weights as float8_e4m3fn with per-block scale factors
    in corresponding weight_scale_inv tensors.
    """
    scale_inv_keys = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    if not scale_inv_keys:
        return state_dict

    logger.info("Dequantizing %d FP8 weights to BF16 (block_size=%d)...", len(scale_inv_keys), block_size)

    for scale_key in scale_inv_keys:
        weight_key = scale_key.replace(".weight_scale_inv", ".weight")
        if weight_key not in state_dict:
            del state_dict[scale_key]
            continue

        weight = state_dict[weight_key]
        scale_inv = state_dict[scale_key]

        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            del state_dict[scale_key]
            continue

        M, N = weight.shape
        scales_expanded = (
            scale_inv
            .repeat_interleave(block_size, dim=0)
            .repeat_interleave(block_size, dim=1)
        )
        scaled = weight.to(torch.float32) * scales_expanded[:M, :N].to(torch.float32)
        state_dict[weight_key] = scaled.to(torch.bfloat16)
        del state_dict[scale_key]

    # Remove any remaining orphan scale_inv keys
    for key in [k for k in state_dict if k.endswith(".weight_scale_inv")]:
        del state_dict[key]

    gc.collect()
    logger.info("FP8 dequantization complete.")
    return state_dict


# ---------------------------------------------------------------------------
# State dict conversion
# ---------------------------------------------------------------------------

def convert_glm5_hf_to_neuron_state_dict(state_dict: dict, config: "Glm5InferenceConfig") -> dict:
    """
    Convert HuggingFace GLM-5.1 state dict to Neuron-compatible format.

    Transformations:
    0. Dequantize FP8 weights to BF16 (if present)
    1. Strip 'model.' prefix from keys
    2. Drop MTP layer (layer 78) if present
    3. Add rank utility tensors for TP sharding
    4. Rename router weights: gate.weight -> router.linear_router.weight
    5. Rename e_score_correction_bias -> router.e_score_correction_bias
    6. Rename expert 3D tensors to NXDI naming (experts.gate_up_proj -> expert_mlps.mlp_op.gate_up_proj.weight)
    7. Stack expert down_proj for NXDI naming
    8. Cast indexer weights_proj to FP32
    """
    # Dequantize FP8 weights if present
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        load_config = getattr(config, "_load_config", None)
        if load_config:
            quant_config = getattr(load_config, "quantization_config", None)
    block_size = 128
    if quant_config and isinstance(quant_config, dict):
        wbs = quant_config.get("weight_block_size", [128, 128])
        block_size = wbs[0] if isinstance(wbs, (list, tuple)) else wbs
    _dequantize_fp8_state_dict(state_dict, block_size=block_size)

    num_hidden_layers = config.num_hidden_layers
    num_local_experts = config.num_local_experts
    tp_degree = getattr(config.neuron_config, "tp_degree", 1)
    first_k_dense = getattr(config, "first_k_dense_replace", 3)

    # Strip 'model.' prefix (HF stores as model.layers.X.foo, NXDI expects layers.X.foo)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        # Drop MTP layer (layer 78 in 78-layer model)
        if new_key.startswith(f"layers.{num_hidden_layers}."):
            continue
        new_state_dict[new_key] = value
    state_dict = new_state_dict

    # Handle tied embeddings: if lm_head.weight missing, copy from embed_tokens
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    # Add rank utilities for TP
    state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    for layer_idx in range(num_hidden_layers):
        # Add rank utility for attention
        state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        # Add rank utility for the indexer's ColumnParallelLinear layers
        state_dict[f"layers.{layer_idx}.self_attn.indexer.rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        # Cast indexer weights_proj to FP32 (must stay FP32 for precision)
        wp_key = f"layers.{layer_idx}.self_attn.indexer.weights_proj.weight"
        if wp_key in state_dict and state_dict[wp_key].dtype != torch.float32:
            state_dict[wp_key] = state_dict[wp_key].to(torch.float32)

        # Skip dense layers (no MoE conversion needed)
        if layer_idx < first_k_dense:
            continue

        # Rename router weights: gate.weight -> router.linear_router.weight
        router_key = f"layers.{layer_idx}.mlp.gate.weight"
        if router_key in state_dict:
            state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
                state_dict[router_key].detach().clone()
            )
            del state_dict[router_key]

        # Rename e_score_correction_bias for GroupLimitedRouter
        bias_key = f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        if bias_key in state_dict:
            state_dict[f"layers.{layer_idx}.mlp.router.e_score_correction_bias"] = (
                state_dict[bias_key].detach().clone()
            )
            del state_dict[bias_key]

        # Expert weight conversion — two possible source formats:
        #
        # Format A (BF16 checkpoints): fused 3D tensors
        #   experts.gate_up_proj: [E, 2*inter, hidden]
        #   experts.down_proj:    [E, hidden, inter]
        #
        # Format B (FP8 checkpoints): per-expert 2D tensors
        #   experts.{e}.gate_proj.weight: [inter, hidden]
        #   experts.{e}.up_proj.weight:   [inter, hidden]
        #   experts.{e}.down_proj.weight: [hidden, inter]
        #
        # NXDI expects:
        #   expert_mlps.mlp_op.gate_up_proj.weight: [E, hidden, 2*inter]
        #   expert_mlps.mlp_op.down_proj.weight:    [E, inter, hidden]

        nxdi_gate_up_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        nxdi_down_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"

        # Format A: fused 3D tensors
        gate_up_key = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
        if gate_up_key in state_dict:
            gate_up = state_dict.pop(gate_up_key)
            state_dict[nxdi_gate_up_key] = gate_up.transpose(1, 2).contiguous()

        down_key = f"layers.{layer_idx}.mlp.experts.down_proj"
        if down_key in state_dict:
            down = state_dict.pop(down_key)
            state_dict[nxdi_down_key] = down.transpose(1, 2).contiguous()

        # Format B: per-expert 2D tensors (FP8 checkpoint after dequantization)
        per_expert_key = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if per_expert_key in state_dict and nxdi_gate_up_key not in state_dict:
            gate_list, up_list, down_list = [], [], []
            for e in range(num_local_experts):
                gk = f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
                uk = f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
                dk = f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
                # gate: [inter, hidden], up: [inter, hidden] -> cat -> [2*inter, hidden]
                gate_list.append(state_dict.pop(gk))
                up_list.append(state_dict.pop(uk))
                down_list.append(state_dict.pop(dk))

            # Stack and transpose: [E, 2*inter, hidden] -> [E, hidden, 2*inter]
            gate_up_fused = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(gate_list, up_list)], dim=0
            )
            state_dict[nxdi_gate_up_key] = gate_up_fused.transpose(1, 2).contiguous()
            del gate_list, up_list, gate_up_fused

            # Stack and transpose: [E, hidden, inter] -> [E, inter, hidden]
            down_stacked = torch.stack(down_list, dim=0)
            state_dict[nxdi_down_key] = down_stacked.transpose(1, 2).contiguous()
            del down_list, down_stacked

        gc.collect()

    return state_dict


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------

class Glm5NeuronConfig(MoENeuronConfig):
    """Neuron hardware configuration for GLM-5.1 MoE model."""
    pass


class Glm5InferenceConfig(InferenceConfig):
    """
    Inference configuration for GLM-5.1.

    Handles MLA attention parameters, MoE routing config, DSA indexer,
    and KV cache shape overrides for MLA's compressed cache format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Standard HF config attributes expected by model_base.py
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "return_dict"):
            self.return_dict = True

        # Map HF config names to NXDI MoE names
        self.num_local_experts = getattr(self, "n_routed_experts", getattr(self, "num_experts", 256))
        self.n_shared_experts = getattr(self, "n_shared_experts", 1)
        self.num_experts_per_tok = getattr(self, "num_experts_per_tok", 8)

        # Store dense layer intermediate size before overriding with MoE size.
        # HF config uses "intermediate_size" for the dense FFN (12288).
        if not hasattr(self, "dense_intermediate_size"):
            self.dense_intermediate_size = getattr(self, "intermediate_size", 12288)

        # ExpertMLPsV2 reads config.intermediate_size for MoE expert size
        if getattr(self, "moe_intermediate_size", None) is not None:
            self.intermediate_size = self.moe_intermediate_size

        # Activation function
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"

        # Number of dense (non-MoE) layers at the start
        if not hasattr(self, "first_k_dense_replace"):
            self.first_k_dense_replace = 3

        # MoE routing config (only when MoENeuronConfig is used)
        if hasattr(self.neuron_config, "router_config"):
            self.neuron_config.router_config.dtype = torch.float32
            self.neuron_config.router_config.act_fn = "sigmoid"
            # Normalization + scaling is handled by Glm5Router, not ExpertMLPsV2
            self.neuron_config.normalize_top_k_affinities = False

        # Disable numeric CC token (workaround for all-gather/reduce-scatter)
        self.neuron_config.disable_numeric_cc_token = True

        # DSA (Dynamic Sparse Attention) parameters — always present in GLM-5.1
        self.has_indexer = True
        if not hasattr(self, "index_n_heads"):
            self.index_n_heads = 32
        if not hasattr(self, "index_head_dim"):
            self.index_head_dim = 128
        if not hasattr(self, "index_topk"):
            self.index_topk = 2048

        # Compute qk_head_dim if not set
        if not hasattr(self, "qk_head_dim"):
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # MLA KV cache: override head_dim and num_key_value_heads so the
        # KVCacheManager allocates (bsz, 1, max_len, combined_dim).
        # combined_dim = kv_lora_rank + qk_rope_head_dim + index_head_dim
        self.head_dim = self.qk_rope_head_dim + self.kv_lora_rank + self.index_head_dim
        # = 64 + 512 + 128 = 704
        self.num_key_value_heads = 1

    def add_derived_config(self):
        self.num_cores_per_group = 1

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Glm5NeuronConfig

    def get_required_attributes(self) -> List[str]:
        return [
            # MLA parameters
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            # MoE parameters
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
        ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def custom_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
    return compiler_args


# ---------------------------------------------------------------------------
# Dense MLP (for layers 0..first_k_dense_replace-1)
# ---------------------------------------------------------------------------

class Glm5DenseMLP(nn.Module):
    """
    Dense MLP for GLM-5.1 layers 0-2.

    Uses SiLU-gated architecture: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses dense_intermediate_size (12288) instead of moe_intermediate_size (2048).
    """

    def __init__(self, config: Glm5InferenceConfig):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.dense_intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        output = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return (output,)


# ---------------------------------------------------------------------------
# Router (sigmoid + bias + top-k + normalize + scale)
# ---------------------------------------------------------------------------

class Glm5Router(GroupLimitedRouter):
    """
    GroupLimitedRouter with GLM-5.1's routed_scaling_factor.

    GLM-5.1 uses n_group=1, topk_group=1, so group-limited selection degenerates
    to simple top-k. After selection, affinities are L1-normalized and scaled by
    routed_scaling_factor (2.5).

    This replaces the normalize_top_k_affinities step in ExpertMLPsV2,
    so the config must set normalize_top_k_affinities=False.
    """

    def __init__(self, routed_scaling_factor: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        topk_idx, _ = self.noaux_tc_top_k(expert_affinities)
        topk_idx = topk_idx.detach().to(dtype=torch.long)

        # Gather affinities for selected experts, normalize, and scale
        topk_weights = expert_affinities.gather(1, topk_idx)  # (T, top_k)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Scatter back to dense (T, E) layout for ExpertMLPsV2
        expert_affinities_scaled = torch.zeros_like(expert_affinities)
        expert_affinities_scaled.scatter_(1, topk_idx, topk_weights)

        return router_logits, expert_affinities_scaled, topk_idx


# ---------------------------------------------------------------------------
# DSA Indexer (Dynamic Sparse Attention)
# ---------------------------------------------------------------------------

class Glm5Indexer(nn.Module):
    """
    Dynamic Sparse Attention (DSA) Indexer for GLM-5.1.

    Computes relevance scores for each query token against all past tokens,
    then selects the top-k most relevant positions. The MLA attention layer
    uses the resulting indices as a sparse mask.

    Key differences from DeepSeek-V3.2 Indexer:
    - Uses LayerNorm (with bias) instead of Hadamard transform for key normalization
    - Uses split-half RoPE (same as main attention), not non-interleaved
    - No FP8 quantization — scoring done in BF16/FP32
    - TP sharding: wq_b and weights_proj are ColumnParallel; wk is replicated
    """

    def __init__(self, config, tensor_model_parallel_group=None):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.tp_degree = config.neuron_config.tp_degree
        self.tensor_model_parallel_group = tensor_model_parallel_group
        self.softmax_scale = self.head_dim ** -0.5

        dtype = config.neuron_config.torch_dtype

        # Determine if we can shard indexer heads across TP ranks.
        # GLM-5.1 has 32 indexer heads; at tp=64 this doesn't divide.
        # When sharding isn't possible, replicate all heads on every rank
        # and skip the all-reduce (each rank computes the full result).
        can_shard = (not cpu_mode()) and (self.n_heads % self.tp_degree == 0)
        self.indexer_sharded = can_shard

        if cpu_mode() or not can_shard:
            self.n_local_heads = self.n_heads
        else:
            self.n_local_heads = self.n_heads // self.tp_degree

        # Q projection: q_lora_rank -> index_n_heads * index_head_dim
        if tensor_model_parallel_group is not None and can_shard:
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, self.n_heads * self.head_dim,
                bias=False, gather_output=False, dtype=dtype,
                tensor_model_parallel_group=tensor_model_parallel_group,
            )
        elif tensor_model_parallel_group is not None:
            # Can't shard by heads, but shard by output dim to save memory.
            # gather_output=True reconstructs the full output.
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, self.n_heads * self.head_dim,
                bias=False, gather_output=True, dtype=dtype,
                tensor_model_parallel_group=tensor_model_parallel_group,
            )
        else:
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=dtype,
            )

        # K projection: dim -> index_head_dim (replicated, single head shared across ranks)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False, dtype=dtype)

        # K normalization — LayerNorm with bias (NOT RMSNorm, NOT Hadamard)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Per-head weights for scoring: dim -> index_n_heads
        if tensor_model_parallel_group is not None and can_shard:
            self.weights_proj = ColumnParallelLinear(
                self.dim, self.n_heads,
                bias=False, gather_output=False, dtype=torch.float32,
                tensor_model_parallel_group=tensor_model_parallel_group,
            )
        else:
            self.weights_proj = nn.Linear(
                self.dim, self.n_heads, bias=False, dtype=torch.float32,
            )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        is_prefill: bool,
        past_indexer_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k indices for sparse attention masking.

        Args:
            x: hidden_states (bsz, seqlen, dim)
            qr: compressed query from MLA's q_a_proj + q_a_layernorm (bsz, seqlen, q_lora_rank)
            position_ids: (bsz, seqlen)
            cos_cache, sin_cache: precomputed RoPE tables
            is_prefill: True for context encoding, False for token generation
            past_indexer_k: (bsz, prior_len, index_head_dim) from KV cache, decode only

        Returns:
            topk_indices: (bsz, seqlen, topk) positions selected by the indexer
            processed_k: (bsz, seqlen, index_head_dim) to store in KV cache
        """
        bsz, seqlen, _ = x.size()

        # --- Q path ---
        q = self.wq_b(qr)  # (bsz, seqlen, n_local_heads * head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # Split into rope and non-rope parts, apply RoPE (BSHD layout)
        q_pe = q[..., : self.rope_head_dim]
        q_nope = q[..., self.rope_head_dim :]
        q_pe = apply_rotary_pos_emb_bshd(q_pe, cos_cache, sin_cache, position_ids)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # --- K path ---
        k = self.wk(x)  # (bsz, seqlen, head_dim)
        k = self.k_norm(k)  # LayerNorm (not Hadamard transform)
        k_pe = k[..., : self.rope_head_dim]
        k_nope = k[..., self.rope_head_dim :]
        k_pe = apply_rotary_pos_emb_bshd(
            k_pe.unsqueeze(2), cos_cache, sin_cache, position_ids
        ).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Save processed k for KV cache (returned to caller)
        processed_k = k  # (bsz, seqlen, head_dim)

        # --- Assemble keys for scoring ---
        if is_prefill:
            k_for_scoring = k  # (bsz, seqlen, head_dim)
        else:
            # Decode: concatenate cached prior keys with current key
            k_for_scoring = torch.cat([past_indexer_k, k], dim=1)  # (bsz, end_pos, head_dim)

        # --- Compute per-head weights ---
        weights = self.weights_proj(x.float()) * (self.n_heads ** -0.5)  # (bsz, seqlen, n_local_heads)

        # --- BF16 indexer scoring ---
        # score[b,s,h,t] = sum_d(q[b,s,h,d] * k[b,t,d]) * softmax_scale
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_for_scoring.float()) * self.softmax_scale
        # Weighted sum across local heads (FP32 for precision)
        index_score = torch.einsum("bsht,bsh->bst", scores, weights)  # (bsz, seqlen, kv_len)

        # --- All-reduce across TP ranks (sum partial head contributions) ---
        # Only needed when indexer heads are actually sharded across ranks.
        if self.indexer_sharded and self.tensor_model_parallel_group is not None and self.tp_degree > 1:
            torch.distributed.all_reduce(
                index_score,
                op=torch.distributed.ReduceOp.SUM,
                group=self.tensor_model_parallel_group,
            )

        # --- Mask invalid positions ---
        if is_prefill:
            # Causal mask: can't attend to future positions
            causal_mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=x.device, dtype=index_score.dtype,
            ).triu_(1)
            index_score = index_score + causal_mask.unsqueeze(0)

        # --- Top-k selection ---
        kv_len = index_score.shape[-1]
        k_val = min(self.index_topk, kv_len)
        if k_val >= kv_len:
            # All positions selected — skip topk
            topk_indices = torch.arange(kv_len, device=x.device).unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1)
        else:
            topk_indices = index_score.topk(k_val, dim=-1)[1]  # (bsz, seqlen, topk)

        return topk_indices, processed_k


# ---------------------------------------------------------------------------
# MLA Attention (Multi-head Latent Attention with DSA)
# ---------------------------------------------------------------------------

class Glm5Attention(nn.Module):
    """
    Multi-head Latent Attention (MLA) for GLM-5.1 with DSA indexer integration.

    MLA architecture:
      Query:  x → q_a_proj → RMSNorm → q_b_proj → split(q_nope[192], q_pe[64]) → RoPE(q_pe)
      KV:     x → kv_a_proj → split(compressed[512], k_pe[64])
              → RMSNorm(compressed) → kv_b_proj → split(k_nope[192], v[256])
              → RoPE(k_pe) → expand k_pe to all heads
      Cache: compressed [k_pe | kv_compressed | indexer_k] format
      Indexer: selects top-k tokens via DSA
    """

    def __init__(self, config: Glm5InferenceConfig, layer_idx: Optional[int] = None, tensor_model_parallel_group=None):
        super().__init__()

        # Config
        self.config = config
        self.neuron_config = config.neuron_config

        # Tensor parallelism
        self.tp_degree = config.neuron_config.tp_degree
        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        else:
            try:
                from neuronx_distributed.parallel_layers import parallel_state
                self.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            except Exception:
                self.tensor_model_parallel_group = None
        self.rank_util = SPMDRank(world_size=self.tp_degree)

        # Data types
        self.torch_dtype = getattr(config.neuron_config, "attention_dtype", None) or config.neuron_config.torch_dtype
        self.rpl_reduce_dtype = getattr(config.neuron_config, "rpl_reduce_dtype", None)

        # Sequence parallelism
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # RoPE — standard Llama/NeoX, no YaRN
        self.rotary_emb = Glm5RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_parameters", {}).get("rope_theta", 1000000.0)
            if isinstance(getattr(config, "rope_parameters", None), dict)
            else getattr(config, "rope_theta", 1000000.0),
        )
        self.bias = getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        assert layer_idx is not None, "Please make sure to provide a `layer_idx` when creating this class."

        self.attention_dropout = config.attention_dropout
        self.num_total_heads = config.num_attention_heads
        assert self.num_attention_heads % self.tp_degree == 0, "Number of attention heads must be a multiple of tp degree."
        if cpu_mode():
            self.num_heads = self.num_total_heads
        else:
            self.num_heads = self.num_total_heads // self.tp_degree

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 256

        self.head_dim = self.v_head_dim  # 256 — for output projection sizing

        self.is_causal = True
        self.init_mla_properties()

        self.softmax_scale = self.q_head_dim ** (-0.5)

        # DSA Indexer — always present for GLM-5.1
        self.has_indexer = True
        self.indexer = Glm5Indexer(
            config, tensor_model_parallel_group=self.tensor_model_parallel_group,
        )
        self.index_head_dim = config.index_head_dim

    def init_mla_properties(self):
        config = self.config
        dtype = self.torch_dtype

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, self.num_total_heads * self.q_head_dim, bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )
        else:
            if self.tensor_model_parallel_group is not None:
                self.q_a_proj = ColumnParallelLinear(
                    self.hidden_size, config.q_lora_rank,
                    bias=config.attention_bias,
                    gather_output=True,
                    dtype=dtype,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                )
            else:
                self.q_a_proj = nn.Linear(
                    self.hidden_size, config.q_lora_rank, bias=config.attention_bias, dtype=dtype
                )
            self.q_a_layernorm = get_rmsnorm_cls()(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank, self.num_total_heads * self.q_head_dim, bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )

        if self.tensor_model_parallel_group is not None:
            self.kv_a_proj_with_mqa = ColumnParallelLinear(
                self.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
                gather_output=True,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
        else:
            self.kv_a_proj_with_mqa = nn.Linear(
                self.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
                dtype=dtype
            )
        self.kv_a_layernorm = get_rmsnorm_cls()(config.kv_lora_rank)
        if self.tensor_model_parallel_group is not None:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                self.num_total_heads
                * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )
        else:
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_total_heads
                * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )

        if self.tensor_model_parallel_group is not None:
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=True,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
                reduce_dtype=self.rpl_reduce_dtype,
            )
        else:
            self.o_proj = nn.Linear(
                self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: torch.Tensor = None,
            active_mask: Optional[torch.LongTensor] = None,
            adapter_ids=None,
            cos_cache: Optional[torch.Tensor] = None,
            sin_cache: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """Forward pass for GLM-5.1 MLA attention with DSA sparse masking."""
        # On decode, past_key_value comes from KVCacheManager as [k_cache, v_cache]
        if past_key_value is not None and isinstance(past_key_value, (list, tuple)):
            combined = past_key_value[0].squeeze(1)  # (bsz, seq_len, combined_dim)
            past_key_value = combined

        if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        bsz, q_len, _ = hidden_states.size()

        # Weight matrix absorption (kv_b_proj absorbed into attention computation)
        # kv_b_proj: kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim)
        # Per-head layout: [qk_nope_192 | v_256]
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)

        out_absorb = wkv_b[:, self.qk_nope_head_dim:, :]  # (H, v_head_dim, kv_lora_rank)

        # Compute compressed query (qr) — needed for both MLA Q and Indexer
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
            qr = None
        else:
            qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(qr)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        q_nope, q_pe = torch.tensor_split(
            q, (self.qk_nope_head_dim,), dim=-1
        )
        compressed_kv, k_pe = torch.tensor_split(
            compressed_kv, (self.kv_lora_rank,), dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # q_nope absorbing: combine q_nope with kv_b weight to avoid expanding kv
        q_absorb = wkv_b[:, :self.qk_nope_head_dim]
        q_nope = torch.einsum('hdc,bhqd->bhqc', q_absorb, q_nope)

        seq_len = self.neuron_config.seq_len
        if sin_cache is None and cos_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos_cache, sin_cache, position_ids)

        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
        active_scores *= self.softmax_scale

        # --- Prefill path ---
        if past_key_value is None:
            # DSA: compute indexer sparse mask
            if qr is not None:
                topk_indices, indexer_k = self.indexer(
                    hidden_states, qr, position_ids, cos_cache, sin_cache,
                    is_prefill=True,
                )
                # Create sparse mask: -inf everywhere except topk positions
                index_mask = torch.full(
                    (bsz, 1, q_len, q_len), float("-inf"),
                    device=hidden_states.device, dtype=active_scores.dtype,
                )
                index_mask.scatter_(-1, topk_indices.unsqueeze(1), 0.0)
                active_scores = active_scores + index_mask

            active_scores = torch.where(attention_mask, active_scores, torch.finfo(active_scores.dtype).min)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                k_pe.dtype
            )

            # Attention result with V absorb
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

        # --- Decode path ---
        else:
            # Split prior cache into MLA components and indexer k
            mla_dim = self.qk_rope_head_dim + self.kv_lora_rank
            k_pe_prior = past_key_value[..., : self.qk_rope_head_dim]
            compressed_kv_prior = past_key_value[..., self.qk_rope_head_dim : mla_dim]
            indexer_k_prior = past_key_value[..., mla_dim :]

            k_pe_prior = k_pe_prior.reshape(bsz, 1, compressed_kv_prior.shape[1], self.qk_rope_head_dim)

            # Scores against prior tokens
            prior_scores = torch.matmul(q_pe, k_pe_prior.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv_prior)
            prior_scores *= self.softmax_scale

            # DSA: compute indexer sparse mask over all positions
            if qr is not None:
                topk_indices, indexer_k = self.indexer(
                    hidden_states, qr, position_ids, cos_cache, sin_cache,
                    is_prefill=False, past_indexer_k=indexer_k_prior,
                )
                # Build mask over all positions [prior | active]
                prior_len = prior_scores.shape[-1]
                end_pos = prior_len + 1
                full_mask = torch.full(
                    (bsz, 1, 1, end_pos), float("-inf"),
                    device=hidden_states.device, dtype=prior_scores.dtype,
                )
                full_mask.scatter_(-1, topk_indices.unsqueeze(1), 0.0)
                # Split mask for prior and active
                prior_scores = prior_scores + full_mask[..., :prior_len]
                active_scores = active_scores + full_mask[..., prior_len:]

            prior_scores = torch.where(
                attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
            )
            prior_scores = prior_scores.to(torch.float32)

            softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation=False)
            softmax_prior, softmax_active = softmax_prior.to(k_pe.dtype), softmax_active.to(k_pe.dtype)

            # Attention result with V absorb
            x = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
            attn_active = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            x = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
            attn_prior = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            attn_output = attn_prior + attn_active

        # Transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # Concatenate k_pe and compressed_kv into combined format for KVCacheManager.
        # Format: [k_pe | compressed_kv | indexer_k]
        cache_parts = [k_pe.squeeze(1), compressed_kv]
        if qr is not None:
            cache_parts.append(indexer_k)
        combined = torch.cat(cache_parts, dim=-1).unsqueeze(1)
        past_key_value = (combined, combined)

        return attn_output, past_key_value, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronGlm5DecoderLayer(nn.Module):
    """
    GLM-5.1 decoder layer with MLA attention and Dense MLP or MoE.

    Layers 0 through first_k_dense_replace-1 use a dense MLP;
    remaining layers use Mixture-of-Experts (MoE).
    """

    def __init__(self, config: Glm5InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx < getattr(config, "first_k_dense_replace", 3)

        self.self_attn = Glm5Attention(config=config, layer_idx=layer_idx)
        self.moe_fused_nki_kernel_enabled = getattr(config.neuron_config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if self.is_dense_layer:
            self.mlp = Glm5DenseMLP(config)
        elif self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        # Swap in Glm5Router (GroupLimitedRouter + routed_scaling_factor)
        if not self.is_dense_layer:
            self.mlp.router = Glm5Router(
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 2.5),
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                n_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                dtype=config.neuron_config.router_config.dtype,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=1,
            )

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Please use `attention_mask` instead."
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP (Dense for first_k_dense_replace layers, MoE for rest)
        residual = hidden_states
        if self.is_dense_layer:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            if not self.moe_fused_nki_kernel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            is_speculative_decoding = self.config.neuron_config.enable_fused_speculation and (not self.config.neuron_config.is_prefill_stage)
            hidden_states = self.mlp(hidden_states, padding_mask, is_speculative_decoding=is_speculative_decoding)[0]
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class NeuronGlm5Model(NeuronBaseModel):
    """
    NeuronGlm5Model extends the GLM-5.1 model to be traceable.
    The forward function of this class is traced for Neuron compilation.
    """

    def setup_attr_for_model(self, config: Glm5InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Glm5InferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronGlm5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------

class NeuronGlm5ForCausalLM(NeuronBaseForCausalLM):
    """
    GLM-5.1 CausalLM for NeuronX Distributed Inference.
    """

    _model_cls = NeuronGlm5Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return Glm5InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Glm5InferenceConfig) -> dict:
        return convert_glm5_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        args = custom_compiler_args()
        args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Glm5InferenceConfig",
    "Glm5NeuronConfig",
    "Glm5Attention",
    "Glm5DenseMLP",
    "Glm5Router",
    "Glm5Indexer",
    "NeuronGlm5DecoderLayer",
    "NeuronGlm5Model",
    "NeuronGlm5ForCausalLM",
]
