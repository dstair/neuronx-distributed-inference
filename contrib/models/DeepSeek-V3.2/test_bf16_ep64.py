#!/usr/bin/env python3
"""
BF16 EP=64 test for DeepSeek V3.2 on trn2.48xlarge.

Uses TKG module (for correct EP all-reduce path) but disables the fused
mega-kernel (which doesn't support shared experts). The flat compiler flow
within TKG handles router + experts + shared experts as separate ops.
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
    """Monkey-patch MoEFusedTKG._can_use_nki_kernel to disable the fused mega-kernel.
    Individual kernels are already disabled by the framework. This ensures the
    flat compiler path is always used within TKG."""
    from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
    import types

    _orig = MoEFusedTKG._can_use_nki_kernel

    def _patched(self, kernel_type, hidden_states):
        if kernel_type == "moe_fused":
            return False
        return _orig(self, kernel_type, hidden_states)

    MoEFusedTKG._can_use_nki_kernel = _patched
    print("Patched MoEFusedTKG: mega-kernel disabled, using flat compiler flow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", default="/scratch/deepseek_v32_bf16_ep64")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--shard-only", action="store_true")
    args = parser.parse_args()

    # Patch BEFORE any model imports that might trigger TKG creation
    patch_tkg_disable_mega_kernel()

    from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
    from src.modeling_deepseek import (
        DeepseekV3InferenceConfig,
        NeuronDeepseekV3ForCausalLM,
    )

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        moe_ep_degree=64,
        moe_tp_degree=1,
        # TKG enabled (needed for correct EP all-reduce path)
        # mega-kernel disabled via monkey-patch above
        moe_fused_nki_kernel_enabled=True,
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
    inf_config._is_fp8 = False  # Force BF16

    print(f"Config: EP={inf_config.neuron_config.moe_ep_degree}, "
          f"TP_MoE={inf_config.neuron_config.moe_tp_degree}, "
          f"intermediate={inf_config.intermediate_size}, "
          f"is_fp8={inf_config._is_fp8}, "
          f"TKG={inf_config.neuron_config.moe_fused_nki_kernel_enabled}")

    neff_path = os.path.join(args.compiled_path, "model.pt")
    weights_path = os.path.join(args.compiled_path, "weights")

    # Step 1: Compile
    if not os.path.exists(neff_path):
        print(f"Compiling to {args.compiled_path}...")
        t0 = time.time()
        model = NeuronDeepseekV3ForCausalLM(args.model_path, inf_config)
        model.compile(args.compiled_path)
        print(f"Compilation took {time.time()-t0:.0f}s")
        del model
        gc.collect()

    if args.compile_only:
        print("Compile-only mode, exiting.")
        return

    # Step 2: Shard weights (if not already done)
    existing_shards = len([f for f in os.listdir(weights_path) if f.endswith(".safetensors")]) if os.path.exists(weights_path) else 0
    if existing_shards < 64:
        print(f"Sharding weights ({existing_shards} existing shards, need 64)...")
        t0 = time.time()

        from neuronx_distributed_inference.modules.checkpoint import load_state_dict
        from neuronx_distributed.trace.functions import shard_checkpoint
        from neuronx_distributed.parallel_layers import parallel_state

        # Initialize parallel state for sharding
        if not parallel_state.model_parallel_is_initialized():
            if not torch.distributed.is_initialized():
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "29500")
                torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
            from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
            initialize_model_parallel(
                tensor_model_parallel_size=64,
                expert_model_parallel_size=64,
                skip_collective_init=True,
            )

        # Load and convert weights
        print("Loading state dict...")
        model_sd = load_state_dict(args.model_path)
        print(f"Loaded {len(model_sd)} keys")

        # Convert HF -> Neuron format
        from src.modeling_deepseek import convert_deepseek_v3_hf_to_neuron_state_dict
        model_sd = convert_deepseek_v3_hf_to_neuron_state_dict(model_sd, inf_config)
        print(f"Converted to {len(model_sd)} Neuron keys")

        # Create model for sharding (need the module structure for preshard hooks)
        model = NeuronDeepseekV3ForCausalLM(args.compiled_path)
        neuron_model = model.models[0].model_cls(inf_config)
        neuron_model.bfloat16()

        # Shard and save — resume from existing_shards
        os.makedirs(weights_path, exist_ok=True)
        print(f"Sharding ranks {existing_shards}..63...")
        shard_checkpoint(
            checkpoint=model_sd,
            model=neuron_model,
            start_rank=existing_shards,
            serialize_path=weights_path,
        )
        print(f"Sharding took {time.time()-t0:.0f}s")
        del model_sd, neuron_model, model
        gc.collect()
    else:
        print(f"Found {existing_shards} sharded weight files, skipping sharding")

    if args.shard_only:
        print("Shard-only mode, exiting.")
        return

    # Step 3: Load and generate
    print(f"Loading from {args.compiled_path}...")
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
    print(f"BF16 EP=64 Results:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Output: {text}")
    print(f"  New tokens: {new_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    if new_tokens > 1:
        print(f"  Throughput: {new_tokens/elapsed:.1f} tok/s")
    print(f"{'='*60}")

    generated = text[len(args.prompt):].strip()
    words = generated.split()
    if len(words) < 2:
        print("WARNING: Very short output — possible garbage")
    elif len(set(words)) == 1 and len(words) > 3:
        print("WARNING: Repetitive output — possible garbage")
    else:
        print("Output looks coherent!")


if __name__ == "__main__":
    main()
