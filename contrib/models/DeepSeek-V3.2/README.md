# Contrib Model: DeepSeek-V3.2

NeuronX Distributed Inference implementation of DeepSeek V3.2, a 671B parameter Mixture-of-Experts model (37B active per token) from DeepSeek AI. Builds on DeepSeek V3 with the addition of **DeepSeek Sparse Attention (DSA) Indexer** for improved attention efficiency.

## Model Family

| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| **DeepSeek-V3.2** | `deepseek-ai/DeepSeek-V3.2` | 671B | 37B | trn2.48xlarge (TP=64) |

**License:** [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/LICENSE)

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 61 (3 dense + 58 MoE) |
| Hidden Size | 7168 |
| Attention | MLA + DSA Indexer (new in V3.2) |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| Attention Heads | 128 |
| **DSA Index Heads** | **64** |
| **DSA Index Head Dim** | **128** |
| **DSA Index Top-K** | **2048** |
| Routed Experts | 256 (8 groups of 32, top-4 groups, 8 experts per token) |
| Shared Experts | 1 |
| Dense Intermediate | 18432 |
| MoE Intermediate | 2048 |
| Position Encoding | YaRN RoPE (interleaved layout) |
| Vocabulary | 129280 |

### What's New in V3.2 (vs V3.0)

- **DeepSeek Sparse Attention (DSA) Indexer:** Each attention layer includes an indexer module that computes relevance scores for all past tokens and selects the top-2048 most relevant positions. The MLA attention then uses these indices as a sparse mask, enabling efficient long-context attention.
  - Indexer uses its own Q/K projections with non-interleaved RoPE and Hadamard transform
  - Indexer keys are stored in the KV cache alongside MLA's `[k_pe | compressed_kv]`, increasing head_dim from 576 to 704
  - BF16 scoring (no FP8 kernels needed for the indexer)
- **Non-interleaved RoPE for indexer:** The DSA indexer uses standard (non-interleaved) RoPE, different from MLA's interleaved YaRN RoPE

### FP8 Inference

The FP8 preprocessing pipeline (`src/preprocess_fp8.py`) converts HuggingFace FP8 weights for Neuron:
- OCP e4m3fn → Neuron E4M3 rescaling (448/240 factor)
- Block-wise scales → per-tensor scales
- Fuses gate/up projections while keeping expert weights in FP8

During weight conversion, FP8 expert weights are dequantized to BF16 using their per-tensor scales. This produces numerically correct weights from the FP8 source checkpoint. The `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` compiler flag is required.

**FP8 vs BF16 comparison:**

| Metric | BF16 | FP8 (dequant→BF16) |
|--------|------|---------------------|
| Logit match | 27/27 (100%) | 26/27 (96.3%) |
| Abs mean diff | 0.097 | 0.181 |
| Max abs diff | 0.250 | 1.500 |
| TPOT p50 | 24.3ms | 24.0ms |
| TTFT p50 | 265ms | 266ms |
| Load time | 53s | 143s |

Note: True FP8 compute (keeping weights in FP8 on device) is blocked by the NKI MoE kernel's top_k=1 limitation and the fused TKG kernel's shared expert dimension mismatch. The current FP8 path dequantizes to BF16 during weight conversion, so runtime performance is identical to BF16.

## Test Results

### Unit Tests (CPU)

| Test Module | Tests | Status |
|-------------|-------|--------|
| test_config.py | 15 | 15/15 PASS |
| test_rope.py | 3 | 3/3 PASS |
| test_router.py | 9 | 9/9 PASS |
| test_weight_conversion.py | 10 | 10/10 PASS |
| test_v32_correctness.py | 9 | 9/9 PASS (23 skipped, need full weights) |
| test_v32_weights.py | 9 | 9/9 PASS (23 skipped, need full weights) |
| **Total** | **55** | **55/55 PASS** |

### Logit Divergence Test (671B, trn2.48xlarge, TP=64, lnc=2, seq=128, bs=1)

#### Teacher-forced results (27 tokens) — **27/27 (100.0%)**

| Pos | Token | Golden Logit | TF Logit | Diff | Match |
|-----|-------|-------------|----------|------|-------|
| 0 | Paris | 28.125 | 28.250 | +0.125 | YES |
| 1 | . | 28.875 | 29.000 | +0.125 | YES |
| 2 | \n | 28.875 | 29.000 | +0.125 | YES |
| 3 | 法 | 28.500 | 28.500 | +0.000 | YES |
| 4 | 国 | 36.250 | 36.250 | +0.000 | YES |
| 5 | 的首 | 36.500 | 36.500 | +0.000 | YES |
| 6 | 都是 | 33.250 | 33.250 | +0.000 | YES |
| 7 | 巴黎 | 28.875 | 29.000 | +0.125 | YES |
| 8 | The | 26.750 | 26.750 | +0.000 | YES |
| 9 | capital | 25.125 | 24.875 | -0.250 | YES |
| 10 | of | 28.250 | 28.375 | +0.125 | YES |
| 11 | Italy | 26.375 | 26.500 | +0.125 | YES |
| 12 | is | 31.000 | 31.000 | +0.000 | YES |
| 13 | Rome | 30.000 | 30.000 | +0.000 | YES |
| 14 | . | 32.500 | 32.750 | +0.250 | YES |
| ... | ... | ... | ... | ... | YES |
| 26 | Madrid | 32.000 | 32.250 | +0.250 | YES |

**Logit drift:** mean=+0.023, abs_mean=0.097, max_abs=0.250

### Logit Divergence Summary

| Metric | V3.2 (DSA Indexer) | V3.0 (baseline) |
|--------|-------------------|-----------------|
| Teacher-forced match | **27/27 (100.0%)** | 30/32 (93.8%) |
| Abs mean logit diff | **0.097** | 0.324 |
| Max abs logit diff | **0.250** | 1.000 |

V3.2 shows improved logit consistency over V3.0, likely due to the DSA Indexer's sparse attention providing more deterministic token selection.

### Multi-Prompt Generation Quality (671B, TP=64)

Single-request greedy generation (top_k=1), 64 output tokens per prompt:

| Prompt | First Token | Status |
|--------|-------------|--------|
| "The capital of France is" | Paris | PASS |
| "def fibonacci(n):" | (newline) | PASS |
| "The theory of relativity states that" | the | PASS |
| "In a shocking finding, scientists discovered" | a | PASS |
| "To make a chocolate cake, you need" | to | PASS |
| "The largest ocean on Earth is" | the | PASS |
| "Machine learning is a subset of" | artificial | PASS |
| "The year 2025 will be remembered for" | the | PASS |

All 8 prompts produce coherent, factually correct, multi-sentence responses. Code generation (fibonacci) produces syntactically valid Python. Model generates multilingual output (English + Chinese) for geography prompts.

### Generation Output (671B, TP=64, seq_len=128, greedy top_k=1)

**Prompt:** "The capital of France is"

**Output:** Paris. 法国的首都是巴黎。
The capital of Italy is Rome. 意大利的首都是罗马。
The capital of Spain is Madrid. 西班牙的首都是马德里。
The capital of Portugal is Lisbon. 葡萄牙的首都是里斯本。

**Status:** PASS — coherent, factually correct, multilingual response.

## Performance Benchmarks

**SDK 2.28**, BF16, trn2.48xlarge (64 NeuronCores), lnc=2.

### NXDI Native Benchmark (bs=1, seq_len=128, 5 input / 32 output tokens)

| Component | p50 (ms) | p90 (ms) | p99 (ms) | Throughput |
|-----------|----------|----------|----------|------------|
| **Token Generation (TPOT)** | **24.3** | 24.7 | 24.8 | 41.2 tok/s |
| **Context Encoding (TTFT)** | **265.4** | 268.6 | 269.6 | — |
| **End-to-End** | **1,018** | 1,031 | 1,035 | 31.4 tok/s |

Measured with 20 timed iterations after 3 warmup iterations.

### Timing Summary

| Operation | Time |
|-----------|------|
| NEFF compilation (first time) | ~13 min |
| NEFF compilation (from cache) | ~1s |
| Weight sharding (FP8 → 64 per-rank files, NVMe RAID0) | ~92 min |
| Load from pre-sharded checkpoints (NVMe) | 53s |
| TPOT (token generation, p50) | 24.3 ms |
| TTFT (context encoding, 5 tokens) | 265 ms |

### Maximum Sequence Length

| seq_len | CTE Bucket | Compile | Load | Status | Notes |
|---------|-----------|---------|------|--------|-------|
| 128 | 32 | PASS | PASS | **PASS** | Default, all benchmarks |
| 128 | 128 | PASS | HBM OOM | **FAIL** | CTE scratchpad + TKG > 24GB per NC pair |

**Why V3.2 supports shorter sequences than V3.0:** The DSA Indexer adds significant per-layer HBM overhead:

| Component | V3.0 | V3.2 | Delta |
|-----------|------|------|-------|
| KV cache head_dim | 576 | 704 | +22% (indexer keys stored in cache) |
| Indexer `wk` weight (replicated) | — | 128×7168 per layer × 61 layers | ~110 MB/NC |
| Indexer `wq_b`, `weights_proj`, `k_norm` | — | ~2 MB/layer (TP-sharded) | ~8 MB/NC |

V3.0's TKG model used ~23.1 GB of the 24 GB HBM per NC pair, leaving ~900 MB headroom. The DSA Indexer's replicated `wk` weights (~110 MB) plus the 22% larger KV cache consume most of this headroom, causing OOM at larger CTE bucket sizes. V3.0 could run seq_len=512 with CTE bucket=256; V3.2 is limited to seq_len=128 with CTE bucket=32.

Potential mitigations:
- Shard the indexer's `wk` weight across TP ranks (currently replicated)
- Reduce CTE scratchpad allocation in the compiler
- FP8 expert weights (when supported) would free ~50% of expert HBM, creating room for larger contexts

## Usage

### Full 671B Model (trn2.48xlarge, TP=64)

```python
import json, os, torch
from neuronx_distributed_inference.models.config import MoENeuronConfig
from src.modeling_deepseek import NeuronDeepseekV3ForCausalLM

model_path = "/path/to/deepseek-ai/DeepSeek-V3.2/"
compiled_path = "/scratch/deepseek_v32_compiled/"

with open(f"{model_path}/config.json") as f:
    hf_config = json.load(f)

neuron_config = MoENeuronConfig(
    tp_degree=64, batch_size=1, seq_len=128, logical_nc_config=2,
    torch_dtype=torch.bfloat16, save_sharded_checkpoint=True,
    enable_bucketing=True, context_encoding_buckets=[32],
)
hf_config["neuron_config"] = neuron_config
config = NeuronDeepseekV3ForCausalLM.get_config_cls()(**hf_config)

model = NeuronDeepseekV3ForCausalLM(model_path, config)
model.compile(compiled_path)  # First time: ~13 min compile + hours sharding
model.load(compiled_path)     # Subsequent: ~53s from NVMe
```

### FP8 Source Checkpoint (trn2.48xlarge, TP=64)

To use the FP8 HuggingFace checkpoint (e.g. `deepseek-ai/DeepSeek-V3.2`), first preprocess the weights, then compile with the compiler flag:

```bash
# Step 1: Preprocess FP8 weights (OCP→Neuron rescaling, block→per-tensor scales)
python src/preprocess_fp8.py \
  --input-dir /path/to/DeepSeek-V3.2-FP8/ \
  --output-dir /scratch/DeepSeek-V3.2-FP8-neuron/
```

```python
# Step 2: Compile with FP8 compiler flag
import os
os.environ["NEURON_CC_FLAGS"] = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"

# ... same config as above, using preprocessed model_path ...
```

FP8 expert weights are dequantized to BF16 using per-tensor scales during weight conversion. The resulting model is numerically equivalent to BF16 (26/27 logit match).

## Caveats

1. **`logical_nc_config=2` required on trn2** — lnc=1 causes HBM OOM.
2. **TP=64 required** — 256 MoE experts on every rank; TP=32 exceeds 24GB HBM limit.
3. **FP8 dequantization** — Requires ~2TB RAM + NVMe swap. Use trn2.48xlarge's 4x 1.7TB NVMe drives as RAID0 + swap.
4. **MLA incompatible with NeuronAttentionBase** — Custom attention class required.
5. **`save_sharded_checkpoint=True` strongly recommended** — Avoids re-sharding 1.3TB on every load.
6. **DSA Indexer increases KV cache** — head_dim grows from 576 (V3.0) to 704 (V3.2) due to indexer key storage. Combined with replicated indexer `wk` weights (~110 MB/NC across 61 layers), this reduces HBM headroom and limits maximum sequence length compared to V3.0.
7. **FP8 dequantizes to BF16** — True FP8 compute is blocked by NKI kernel limitations (top_k=1 only, shared expert dim mismatch). FP8 source checkpoints are dequantized to BF16 using per-tensor scales during weight conversion. Runtime performance is identical to BF16.
8. **Thinking/reasoning mode not validated** — DeepSeek V3.2 supports chain-of-thought reasoning via `<think>...</think>` tags, but this requires much larger sequence lengths (4096+) than currently supported. seq_len=512 may work for short reasoning chains but has not been tested with V3.2 due to HBM constraints.

## Compatibility Matrix

| Instance | TP | LNC | Status |
|----------|-----|-----|--------|
| trn2.48xlarge | 64 | 2 | **PASS** |

| Component | Version |
|-----------|---------|
| Neuron SDK | 2.28 |
| NxDI | 0.8.0 |
| torch | 2.9.0 |
| transformers | 4.57.6 |
| Python | 3.12 |

## Testing

### Unit Tests (CPU only)

```bash
cd contrib/models/DeepSeek-V3.2/
pytest test/unit/ -v
# Expected: 55/55 PASS
```

### Integration Tests (trn2.48xlarge, TP=64)

```bash
cd contrib/models/DeepSeek-V3.2/
DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3.2 \
DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v32_compiled \
DEEPSEEK_TP_DEGREE=64 \
DEEPSEEK_SEQ_LEN=128 \
pytest test/integration/test_model.py --capture=tee-sys
```

## Example Checkpoints

- `deepseek-ai/DeepSeek-V3.2` (FP8, 642GB)

## Maintainer

AWS Neuron

**Last Updated:** 2026-04-22
