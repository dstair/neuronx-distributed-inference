#!/usr/bin/env python3
"""Shard BF16 EP=64 weights using NXDI's builder (correct EP parallel state)."""

import gc
import json
import os
import sys
import time

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

# Patch TKG mega-kernel before any imports
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
_orig_can_use = MoEFusedTKG._can_use_nki_kernel
def _no_mega(self, kernel_type, hidden_states):
    return False if kernel_type == "moe_fused" else _orig_can_use(self, kernel_type, hidden_states)
MoEFusedTKG._can_use_nki_kernel = _no_mega

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from src.modeling_deepseek import DeepseekV3InferenceConfig, NeuronDeepseekV3ForCausalLM

MODEL_PATH = "/scratch/DeepSeek-V3.2-FP8-neuron"
COMPILED_PATH = "/scratch/deepseek_v32_bf16_ep64"

neuron_config = MoENeuronConfig(
    tp_degree=64, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
    seq_len=128, torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=False, flash_decoding_enabled=False, logical_nc_config=2,
    moe_ep_degree=64, moe_tp_degree=1,
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
inf_config._is_fp8 = False

print(f"EP={inf_config.neuron_config.moe_ep_degree}, TP_MoE={inf_config.neuron_config.moe_tp_degree}")

# Use the NXDI model's compile method which handles parallel state correctly
# The NEFF already exists, so compile() should detect it and only do weight sharding
# Actually, compile() always regenerates. Let's use the builder directly.

model = NeuronDeepseekV3ForCausalLM(MODEL_PATH, inf_config)

print(f"Starting weight sharding at {time.strftime('%H:%M:%S')}")
t0 = time.time()

# compile() does trace + compile + shard. Since we already have model.pt,
# we need to force recompile to get the sharding done.
# The simplest approach: just call compile() again - it will regenerate NEFF + shard
model.compile(COMPILED_PATH)

elapsed = time.time() - t0
print(f"Compile+shard took {elapsed:.0f}s")

# Verify
weights_dir = os.path.join(COMPILED_PATH, "weights")
n_shards = len([f for f in os.listdir(weights_dir) if f.endswith(".safetensors")])
print(f"Sharded weight files: {n_shards}")

if n_shards == 64:
    print("SUCCESS: All 64 shards created")
else:
    print(f"WARNING: Expected 64 shards, got {n_shards}")
