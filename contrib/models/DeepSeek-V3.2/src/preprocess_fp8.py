# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Preprocess DeepSeek V3.2 FP8 checkpoints for Neuron inference.

The HuggingFace FP8 checkpoint uses OCP e4m3fn format with block-wise scales
(128x128 blocks). Neuron uses IEEE-754 FP8 E4M3 (range ±240 vs OCP's ±448).

This script:
1. Rescales FP8 weights from OCP → Neuron range (factor 448/240)
2. Converts block-wise scales to per-tensor scales for the NKI MoE kernel
3. Fuses gate_proj + up_proj into gate_up_proj for each expert
4. Saves the preprocessed checkpoint

Usage:
    python preprocess_fp8.py --input-path /path/to/hf/DeepSeek-V3.2 --output-path /path/to/output
"""

import argparse
import gc
import json
import os

import torch

from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    save_state_dict_safetensors,
)

FP8_SCALING_FACTOR = 448.0 / 240.0
W_DTYPE = torch.float8_e4m3fn
S_DTYPE = torch.float32


def blockwise_to_per_tensor_scale(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Convert block-wise scale_inv to a single per-tensor scale for the NKI kernel.

    The NKI MoE kernel expects one scale value per output row (for gate_up) or
    per output column (for down). We compute the max scale across blocks per row
    and rescale the weight so a single per-row scale is correct.

    Returns (rescaled_weight_fp8, per_row_scale).
    """
    M, N = weight.shape
    # scale_inv shape: (ceil(M/block_size), ceil(N/block_size))
    # Each block's actual value = weight_fp8 * scale_inv[block_row, block_col]

    # Compute per-row max scale (across column blocks)
    # This is the scale we'll use for the whole row
    per_row_scale = scale_inv.max(dim=1, keepdim=True).values  # (ceil(M/bs), 1)

    # Rescale weight: for each block, multiply by (block_scale / row_scale)
    # to normalize so a single row_scale works
    weight_f32 = weight.to(torch.float32)
    rows_b, cols_b = scale_inv.shape

    for rb in range(rows_b):
        r_start = rb * block_size
        r_end = min(r_start + block_size, M)
        row_s = per_row_scale[rb, 0]
        for cb in range(cols_b):
            c_start = cb * block_size
            c_end = min(c_start + block_size, N)
            block_s = scale_inv[rb, cb]
            # Adjust: new_val = old_fp8 * (block_scale / row_scale)
            ratio = block_s / row_s
            weight_f32[r_start:r_end, c_start:c_end] *= ratio

    # Expand per_row_scale to match weight rows
    per_row_scale_expanded = per_row_scale.repeat_interleave(block_size, dim=0)[:M, 0]

    return weight_f32.to(W_DTYPE), per_row_scale_expanded.to(S_DTYPE)


def rescale_ocp_to_neuron(weight_fp8: torch.Tensor, scale: torch.Tensor):
    """Rescale from OCP e4m3fn range (±448) to Neuron E4M3 range (±240)."""
    w_bf16 = weight_fp8.to(torch.bfloat16) / FP8_SCALING_FACTOR
    new_scale = scale * FP8_SCALING_FACTOR
    return w_bf16.to(W_DTYPE), new_scale


def main():
    parser = argparse.ArgumentParser(description="Preprocess DeepSeek V3.2 FP8 weights for Neuron")
    parser.add_argument("--input-path", required=True, help="Path to HF FP8 checkpoint")
    parser.add_argument("--output-path", required=True, help="Path to save preprocessed checkpoint")
    args = parser.parse_args()

    with open(os.path.join(args.input_path, "config.json")) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    num_experts = config["n_routed_experts"]
    hidden_size = config["hidden_size"]
    moe_intermediate_size = config["moe_intermediate_size"]
    dense_intermediate_size = config["intermediate_size"]
    first_k_dense = config["first_k_dense_replace"]
    block_size = config.get("quantization_config", {}).get("weight_block_size", [128, 128])[0]

    print(f"Loading state dict from {args.input_path}...")
    state_dict = load_state_dict(args.input_path)
    keys = set(state_dict.keys())

    for layer_n in range(num_layers):
        prefix = f"model.layers.{layer_n}."
        is_moe = layer_n >= first_k_dense

        if is_moe:
            # --- Process routed experts: fuse gate/up, convert scales ---
            gate_up_weights = []
            gate_up_scales = []
            down_weights = []
            down_scales = []

            for e in range(num_experts):
                ep = f"{prefix}mlp.experts.{e}."
                print(f"  layer {layer_n} expert {e}", end="\r")

                # Gate proj
                gw = state_dict.pop(f"{ep}gate_proj.weight")
                gs = state_dict.pop(f"{ep}gate_proj.weight_scale_inv")
                gw, gs = blockwise_to_per_tensor_scale(gw, gs, block_size)
                gw, gs = rescale_ocp_to_neuron(gw, gs)

                # Up proj
                uw = state_dict.pop(f"{ep}up_proj.weight")
                us = state_dict.pop(f"{ep}up_proj.weight_scale_inv")
                uw, us = blockwise_to_per_tensor_scale(uw, us, block_size)
                uw, us = rescale_ocp_to_neuron(uw, us)

                # Fuse gate + up: (hidden_size, 2*moe_intermediate_size)
                # gate/up weights are (moe_intermediate_size, hidden_size), transpose to (hidden_size, moe_intermediate_size)
                fused_w = torch.cat([gw.T, uw.T], dim=1)  # (hidden_size, 2*moe_intermediate_size)
                fused_s = torch.cat([gs, us], dim=0)  # (2*moe_intermediate_size,)
                gate_up_weights.append(fused_w)
                gate_up_scales.append(fused_s)

                # Down proj
                dw = state_dict.pop(f"{ep}down_proj.weight")
                ds = state_dict.pop(f"{ep}down_proj.weight_scale_inv")
                dw, ds = blockwise_to_per_tensor_scale(dw, ds, block_size)
                dw, ds = rescale_ocp_to_neuron(dw, ds)
                down_weights.append(dw.T)  # (moe_intermediate_size, hidden_size)
                down_scales.append(ds)

                gc.collect()

            print(f"  layer {layer_n}: stacking {num_experts} experts")

            # Stack: (num_experts, hidden_size, 2*moe_intermediate_size)
            state_dict[f"{prefix}mlp.experts.gate_up_proj.weight"] = torch.stack(gate_up_weights)
            state_dict[f"{prefix}mlp.experts.gate_up_proj.scale"] = torch.stack(gate_up_scales)

            # Stack: (num_experts, moe_intermediate_size, hidden_size)
            state_dict[f"{prefix}mlp.experts.down_proj.weight"] = torch.stack(down_weights)
            state_dict[f"{prefix}mlp.experts.down_proj.scale"] = torch.stack(down_scales)

            del gate_up_weights, gate_up_scales, down_weights, down_scales
            gc.collect()

            # --- Process shared experts: dequantize to BF16 (not FP8) ---
            # The NKI kernel doesn't support FP8 for shared experts
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wk = f"{prefix}mlp.shared_experts.{proj}.weight"
                sk = f"{prefix}mlp.shared_experts.{proj}.weight_scale_inv"
                if wk in state_dict and sk in state_dict:
                    w = state_dict[wk]
                    s = state_dict.pop(sk)
                    # Dequantize block-wise to BF16
                    M, N = w.shape
                    s_exp = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
                    state_dict[wk] = (w.to(torch.float32) * s_exp[:M, :N]).to(torch.bfloat16)

        else:
            # --- Dense layers: dequantize FP8 to BF16 ---
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wk = f"{prefix}mlp.{proj}.weight"
                sk = f"{prefix}mlp.{proj}.weight_scale_inv"
                if wk in state_dict and sk in state_dict:
                    w = state_dict[wk]
                    s = state_dict.pop(sk)
                    M, N = w.shape
                    s_exp = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
                    state_dict[wk] = (w.to(torch.float32) * s_exp[:M, :N]).to(torch.bfloat16)

        # --- Attention weights: always dequantize to BF16 ---
        for attn_proj in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
            wk = f"{prefix}self_attn.{attn_proj}.weight"
            sk = f"{prefix}self_attn.{attn_proj}.weight_scale_inv"
            if wk in state_dict and sk in state_dict:
                w = state_dict[wk]
                s = state_dict.pop(sk)
                M, N = w.shape
                s_exp = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
                state_dict[wk] = (w.to(torch.float32) * s_exp[:M, :N]).to(torch.bfloat16)

        gc.collect()

    # Remove any remaining orphan scale_inv keys
    orphan_scales = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    for k in orphan_scales:
        del state_dict[k]

    print(f"Saving preprocessed checkpoint to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    save_state_dict_safetensors(state_dict, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
