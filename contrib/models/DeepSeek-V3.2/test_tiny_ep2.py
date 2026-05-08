#!/usr/bin/env python3
"""
Tiny model EP=2 test for DeepSeek V3.2 on trn2.3xlarge.

Uses kathywu95/deepseek-v3.2-small-random-fp8-64-experts (7.4GB, 64 experts,
2 layers) to debug Expert Parallelism routing on 2 NeuronCores.

Config: TP=2, EP=2, 32 experts/rank, intermediate=2048 (128-multiple ✓)
First test: BF16 (dequant FP8→BF16) to isolate EP routing from FP8 issues.
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def patch_tkg_disable_mega_kernel():
    """Disable fused mega-kernel (doesn't support shared experts / EP correctly)."""
    from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG

    _orig = MoEFusedTKG._can_use_nki_kernel

    def _patched(self, kernel_type, hidden_states):
        if kernel_type == "moe_fused":
            return False
        return _orig(self, kernel_type, hidden_states)

    MoEFusedTKG._can_use_nki_kernel = _patched
    print("Patched: mega-kernel disabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to tiny HF model (kathywu95/deepseek-v3.2-small-random-fp8-64-experts)")
    parser.add_argument("--compiled-path", default="/tmp/deepseek_tiny_ep2")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--ep-degree", type=int, default=2)
    parser.add_argument("--fp8", action="store_true", help="Use FP8 (default: dequant to BF16)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-tkg", action="store_true", help="Disable TKG module")
    args = parser.parse_args()

    # FP8 compiler flag
    if args.fp8:
        os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + \
            " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"

    # Patch BEFORE model imports
    if not args.no_tkg:
        patch_tkg_disable_mega_kernel()

    from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
    from src.modeling_deepseek import (
        DeepseekV3InferenceConfig,
        NeuronDeepseekV3ForCausalLM,
    )

    use_tkg = not args.no_tkg

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        moe_ep_degree=args.ep_degree,
        moe_tp_degree=args.tp_degree // args.ep_degree,
        # TKG for correct EP all-reduce path
        moe_fused_nki_kernel_enabled=use_tkg,
        save_sharded_checkpoint=True,
    )

    with open(os.path.join(args.model_path, "config.json")) as f:
        hf_config = json.load(f)

    def load_config(config):
        for k, v in hf_config.items():
            if k.startswith("_") or k in ("torch_dtype", "transformers_version"):
                continue
            setattr(config, k, v)
        config._name_or_path = args.model_path

    inf_config = DeepseekV3InferenceConfig(neuron_config, load_config=load_config)

    if not args.fp8:
        inf_config._is_fp8 = False  # Force BF16 dequant path

    print(f"Config: TP={args.tp_degree}, EP={inf_config.neuron_config.moe_ep_degree}, "
          f"TP_MoE={inf_config.neuron_config.moe_tp_degree}, "
          f"intermediate={inf_config.intermediate_size}, "
          f"n_experts={inf_config.num_local_experts}, "
          f"experts/rank={inf_config.num_local_experts // args.ep_degree}, "
          f"is_fp8={inf_config._is_fp8}, TKG={use_tkg}, "
          f"has_indexer={inf_config.has_indexer}")

    neff_path = os.path.join(args.compiled_path, "model.pt")

    # Step 1: Compile (also shards weights)
    if not os.path.exists(neff_path):
        print(f"\n=== COMPILING to {args.compiled_path} ===")
        t0 = time.time()
        model = NeuronDeepseekV3ForCausalLM(args.model_path, inf_config)
        model.compile(args.compiled_path)
        print(f"Compilation took {time.time()-t0:.0f}s")
        del model
        gc.collect()
    else:
        print(f"Found compiled model at {neff_path}")

    if args.compile_only:
        print("Compile-only mode, exiting.")
        return

    # Step 3: Load and generate
    print(f"\n=== LOADING from {args.compiled_path} ===")
    t0 = time.time()
    model = NeuronDeepseekV3ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    print(f"Load took {time.time()-t0:.0f}s")

    from transformers import AutoTokenizer, GenerationConfig
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    inputs = tokenizer(args.prompt, padding=True, return_tensors="pt")
    gen_model = HuggingFaceGenerationAdapter(model)

    print("Warmup...")
    _ = gen_model.generate(
        inputs.input_ids, generation_config=gen_config,
        attention_mask=inputs.attention_mask, max_new_tokens=1,
    )

    print(f"Generating (prompt='{args.prompt}', max_new_tokens={args.max_new_tokens})...")
    t0 = time.time()
    outputs = gen_model.generate(
        inputs.input_ids, generation_config=gen_config,
        attention_mask=inputs.attention_mask, max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_len = inputs.input_ids.shape[1]
    new_tokens = len(outputs[0]) - input_len

    print(f"\n{'='*60}")
    mode = "FP8" if args.fp8 else "BF16"
    print(f"Tiny Model {mode} EP={args.ep_degree} Results:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Output: {text}")
    print(f"  New tokens: {new_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    if new_tokens > 1:
        print(f"  Throughput: {new_tokens/elapsed:.1f} tok/s")
    print(f"{'='*60}")

    # Note: random weights → output won't be coherent, but should be
    # non-degenerate (not all same token, not NaN/inf logits)
    generated = text[len(args.prompt):].strip()
    tokens = outputs[0][input_len:]
    unique_tokens = len(set(tokens.tolist()))
    print(f"\nDiagnostics:")
    print(f"  Unique tokens generated: {unique_tokens}/{new_tokens}")
    print(f"  First 10 token IDs: {tokens[:10].tolist()}")
    if unique_tokens <= 2 and new_tokens > 5:
        print("  ⚠️  WARNING: Degenerate output (likely garbage/stuck)")
    elif unique_tokens > 2:
        print("  ✓ Output has variety (EP routing likely working)")


if __name__ == "__main__":
    main()
