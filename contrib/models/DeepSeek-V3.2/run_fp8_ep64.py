#!/usr/bin/env python3
"""
FP8 EP=64 inference for DeepSeek V3.2 (671B) on trn2.48xlarge.

Uses preprocessed FP8 weights with Expert Parallelism (EP=64, TP_MoE=1).
Each rank gets 4 full experts with intermediate=2048.
"""

import gc
import json
import os
import sys
import time

import torch

os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + \
    " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def patch_tkg_disable_mega_kernel():
    from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
    _orig = MoEFusedTKG._can_use_nki_kernel
    def _patched(self, kernel_type, hidden_states):
        if kernel_type == "moe_fused":
            return False
        return _orig(self, kernel_type, hidden_states)
    MoEFusedTKG._can_use_nki_kernel = _patched
    print("Patched: mega-kernel disabled")


# Patch BEFORE model imports
patch_tkg_disable_mega_kernel()

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from src.modeling_deepseek import DeepseekV3InferenceConfig, NeuronDeepseekV3ForCausalLM

MODEL_PATH = "/scratch/DeepSeek-V3.2-FP8-neuron"
COMPILED_PATH = "/scratch/deepseek_v32_fp8_ep64"
SEQ_LEN = 128
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30

neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=SEQ_LEN,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=False,
    flash_decoding_enabled=False,
    logical_nc_config=2,
    moe_ep_degree=64,
    moe_tp_degree=1,
    moe_fused_nki_kernel_enabled=True,
    save_sharded_checkpoint=True,
)

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    hf_config = json.load(f)

def load_config(config):
    for k, v in hf_config.items():
        if k.startswith("_") or k in ("torch_dtype", "transformers_version"):
            continue
        setattr(config, k, v)
    config._name_or_path = MODEL_PATH

inf_config = DeepseekV3InferenceConfig(neuron_config, load_config=load_config)
# _is_fp8 auto-detected from quantization_config

print(f"Config: TP=64, EP={inf_config.neuron_config.moe_ep_degree}, "
      f"TP_MoE={inf_config.neuron_config.moe_tp_degree}, "
      f"intermediate={inf_config.intermediate_size}, "
      f"is_fp8={inf_config._is_fp8}, "
      f"has_indexer={inf_config.has_indexer}")

neff_path = os.path.join(COMPILED_PATH, "model.pt")

# Step 1: Compile (includes weight sharding)
if not os.path.exists(neff_path):
    print(f"\n=== COMPILING to {COMPILED_PATH} ===")
    t0 = time.time()
    model = NeuronDeepseekV3ForCausalLM(MODEL_PATH, inf_config)
    model.compile(COMPILED_PATH)
    print(f"Compilation took {time.time()-t0:.0f}s")
    del model
    gc.collect()
else:
    print(f"Found compiled model at {neff_path}")

# Step 2: Load and generate
print(f"\n=== LOADING from {COMPILED_PATH} ===")
t0 = time.time()
model = NeuronDeepseekV3ForCausalLM(COMPILED_PATH)
model.load(COMPILED_PATH)
print(f"Load took {time.time()-t0:.0f}s")

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer(PROMPT, padding=True, return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)

print("Warmup...")
_ = gen_model.generate(
    inputs.input_ids, generation_config=gen_config,
    attention_mask=inputs.attention_mask, max_new_tokens=1,
)

print(f"Generating (prompt='{PROMPT}', max_new_tokens={MAX_NEW_TOKENS})...")
t0 = time.time()
outputs = gen_model.generate(
    inputs.input_ids, generation_config=gen_config,
    attention_mask=inputs.attention_mask, max_new_tokens=MAX_NEW_TOKENS,
)
elapsed = time.time() - t0

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
input_len = inputs.input_ids.shape[1]
new_tokens = len(outputs[0]) - input_len

print(f"\n{'='*60}")
print(f"FP8 EP=64 Results:")
print(f"  Prompt: {PROMPT}")
print(f"  Output: {text}")
print(f"  New tokens: {new_tokens}")
print(f"  Time: {elapsed:.2f}s")
if new_tokens > 1:
    print(f"  Throughput: {new_tokens/elapsed:.1f} tok/s")
print(f"{'='*60}")

tokens = outputs[0][input_len:]
unique_tokens = len(set(tokens.tolist()))
print(f"\nDiagnostics:")
print(f"  Unique tokens: {unique_tokens}/{new_tokens}")
if unique_tokens <= 2 and new_tokens > 5:
    print("  ⚠️  WARNING: Degenerate output")
else:
    print("  ✓ Output has variety")
