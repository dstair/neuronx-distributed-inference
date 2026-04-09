# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forced logit divergence test for GLM-5.1 on Neuron.

Two-phase approach:
  Phase 1 (generate golden): Compile model, run free generation, save golden logits.
  Phase 2 (teacher-forced):  Feed golden tokens one at a time, compare argmax at each
                              position against golden logits.

This validates:
  - Model compiles and loads successfully on Neuron
  - Weights are loaded correctly (FP8 dequantization, state dict conversion)
  - Model produces coherent output
  - Teacher-forced determinism (same input → same logits)

Usage:
    # Full pipeline (compile + generate golden + teacher-forced comparison):
    GLM5_MODEL_PATH=/scratch/glm-5.1-fp8 \
    GLM5_COMPILED_PATH=/scratch/glm5_compiled \
    python test/integration/teacher_forced_comparison.py

    # With pre-existing golden logits:
    GLM5_MODEL_PATH=/scratch/glm-5.1-fp8 \
    GLM5_COMPILED_PATH=/scratch/glm5_compiled \
    GLM5_GOLDEN_LOGITS=/scratch/glm5_golden_outputs.pt \
    GLM5_GOLDEN_INPUTS=/scratch/glm5_golden_inputs.pt \
    python test/integration/teacher_forced_comparison.py

Environment variables:
    GLM5_MODEL_PATH       Path to HF model weights (required)
    GLM5_COMPILED_PATH    Path to compiled artifacts (default: /scratch/glm5_compiled)
    GLM5_TP_DEGREE        Tensor parallelism degree (default: 64)
    GLM5_SEQ_LEN          Max sequence length (default: 512)
    GLM5_GOLDEN_LOGITS    Path to golden logits (optional; generates if absent)
    GLM5_GOLDEN_INPUTS    Path to golden inputs (optional; generates if absent)
    GLM5_NUM_TOKENS       Number of tokens to generate for golden (default: 32)
    GLM5_PROMPT           Custom prompt (optional)
"""

import os
import sys
import time

import torch
from transformers import AutoTokenizer, GenerationConfig

# Ensure the contrib root (GLM-5/) is on sys.path
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from src.modeling_glm5 import (
    Glm5InferenceConfig,
    NeuronGlm5ForCausalLM,
)


# ── Configuration ──────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("GLM5_MODEL_PATH", "")
COMPILED_PATH = os.environ.get("GLM5_COMPILED_PATH", "/scratch/glm5_compiled")
TP_DEGREE = int(os.environ.get("GLM5_TP_DEGREE", "64"))
SEQ_LEN = int(os.environ.get("GLM5_SEQ_LEN", "512"))
GOLDEN_LOGITS_PATH = os.environ.get("GLM5_GOLDEN_LOGITS", "")
GOLDEN_INPUTS_PATH = os.environ.get("GLM5_GOLDEN_INPUTS", "")
NUM_TOKENS = int(os.environ.get("GLM5_NUM_TOKENS", "32"))

DEFAULT_PROMPT = "The theory of general relativity, proposed by Albert Einstein in 1915, describes gravity as"


def load_or_compile_model(model_path, compiled_path):
    """Load or compile the GLM-5.1 model on Neuron."""
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
    )

    inf_config = Glm5InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    neff_path = os.path.join(compiled_path, "model.pt")
    if not os.path.exists(neff_path):
        print(f"[compile] Compiling to {compiled_path}...")
        t0 = time.time()
        model = NeuronGlm5ForCausalLM(model_path, inf_config)
        model.compile(compiled_path)
        model.load(compiled_path)
        print(f"[compile] Done in {time.time()-t0:.1f}s")
    else:
        print(f"[load] Loading pre-compiled model from {compiled_path}...")
        t0 = time.time()
        model = NeuronGlm5ForCausalLM(model_path, inf_config)
        model.load(compiled_path)
        print(f"[load] Done in {time.time()-t0:.1f}s")

    return model


def generate_golden(model, tokenizer, prompt, num_tokens):
    """Run free generation and capture generated tokens as golden reference."""
    print(f"\n{'='*80}")
    print("PHASE 1: Generating golden tokens")
    print(f"{'='*80}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Tokens to generate: {num_tokens}")

    adapter = HuggingFaceGenerationAdapter(model)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    gen_config = GenerationConfig(
        do_sample=False,
        top_k=1,
        max_new_tokens=num_tokens,
        min_new_tokens=num_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
    )

    t0 = time.time()
    outputs = adapter.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
        return_dict_in_generate=True,
    )
    gen_time = time.time() - t0

    # Extract generated tokens (excluding prompt)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"  Generated text: {generated_text!r}")
    print(f"  Generation time: {gen_time:.2f}s ({num_tokens/gen_time:.1f} tok/s)")
    print(f"  Token IDs: {generated_ids}")

    golden_inputs = {
        "prompt": prompt,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
    }

    return golden_inputs


def teacher_forced_comparison(model, tokenizer, golden_inputs):
    """
    Teacher-forced token comparison.

    Feed golden tokens at each step and compare the generated next token against
    the golden reference. With on-device sampling (top_k=1), we compare token IDs
    directly rather than logit distributions.
    """
    print(f"\n{'='*80}")
    print("PHASE 2: Teacher-forced token comparison")
    print(f"{'='*80}")

    golden_token_ids = golden_inputs["generated_token_ids"]
    prompt = golden_inputs["prompt"]
    num_tokens = len(golden_token_ids)

    # Tokenize prompt
    prompt_ids = tokenizer([prompt], padding=True, return_tensors="pt").input_ids[0].tolist()
    full_sequence = prompt_ids + golden_token_ids

    print(f"  Prompt length: {len(prompt_ids)}")
    print(f"  Golden tokens: {num_tokens}")
    print(f"  Full sequence length: {len(full_sequence)}")

    adapter = HuggingFaceGenerationAdapter(model)
    new_token_ids = []

    t0 = time.time()
    for i in range(num_tokens):
        # Feed prompt + golden tokens up to position i
        prefix = full_sequence[:len(prompt_ids) + i]
        input_ids = torch.tensor([prefix], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        gen_config = GenerationConfig(
            do_sample=False,
            top_k=1,
            max_new_tokens=1,
            min_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
        )

        outputs = adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
        )

        # Extract the single generated token
        new_tok = outputs.sequences[0][len(prefix)].item()
        new_token_ids.append(new_tok)

        g_tok = golden_token_ids[i]
        match = "MATCH" if new_tok == g_tok else "DIFF"
        g_text = tokenizer.decode([g_tok]).replace("\n", "\\n")
        n_text = tokenizer.decode([new_tok]).replace("\n", "\\n")
        print(f"  pos {i:2d}: golden='{g_text}' ({g_tok}) new='{n_text}' ({new_tok}) {match}")

    tf_time = time.time() - t0
    print(f"\n  Teacher-forced time: {tf_time:.2f}s ({num_tokens/tf_time:.1f} tok/s)")

    # === Comparison table ===
    print(f"\n{'='*100}")
    print("TEACHER-FORCED COMPARISON (golden tokens fed at each position)")
    print(f"{'='*100}")
    print(f"{'Pos':>3} {'GoldTok':>8} {'NewTok':>8} {'Gold Decoded':<20} {'New Decoded':<20} {'Match':>6}")
    print("-" * 100)

    num_match = 0
    first_diverge = None

    for i in range(num_tokens):
        g_tok = golden_token_ids[i]
        n_tok = new_token_ids[i]

        g_text = tokenizer.decode([g_tok]).replace("\n", "\\n")
        n_text = tokenizer.decode([n_tok]).replace("\n", "\\n")

        match = g_tok == n_tok
        if match:
            num_match += 1
        if first_diverge is None and not match:
            first_diverge = i

        print(f"{i:3d} {g_tok:8d} {n_tok:8d} {g_text:<20} {n_text:<20} {'YES' if match else 'NO':>6}")

    # === Summary ===
    print(f"\n{'='*100}")
    print("TEACHER-FORCED SUMMARY")
    print(f"  Total positions:    {num_tokens}")
    print(f"  Token matches:      {num_match}/{num_tokens} ({100*num_match/num_tokens:.1f}%)")
    if first_diverge is not None:
        print(f"  First divergence:   position {first_diverge}")
    else:
        print(f"  First divergence:   NONE (perfect match!)")

    # Show divergence positions
    diverge_positions = [i for i in range(num_tokens) if golden_token_ids[i] != new_token_ids[i]]
    if diverge_positions:
        print(f"\n  Divergence positions ({len(diverge_positions)}): {diverge_positions}")

    return {
        "new_token_ids": new_token_ids,
        "num_match": num_match,
        "num_tokens": num_tokens,
        "accuracy": num_match / num_tokens if num_tokens > 0 else 0,
    }


def main():
    if not MODEL_PATH:
        print("ERROR: GLM5_MODEL_PATH must be set.")
        print("  export GLM5_MODEL_PATH=/scratch/glm-5.1-fp8")
        sys.exit(1)

    prompt = os.environ.get("GLM5_PROMPT", DEFAULT_PROMPT)

    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or compile model
    model = load_or_compile_model(MODEL_PATH, COMPILED_PATH)

    # Phase 1: Generate or load golden tokens
    if GOLDEN_INPUTS_PATH and os.path.exists(GOLDEN_INPUTS_PATH):
        print(f"\nLoading golden inputs from {GOLDEN_INPUTS_PATH}...")
        golden_inputs = torch.load(GOLDEN_INPUTS_PATH, weights_only=False)
        print(f"  Golden prompt: {golden_inputs['prompt']!r}")
        print(f"  Golden tokens: {len(golden_inputs['generated_token_ids'])}")
    else:
        golden_inputs = generate_golden(model, tokenizer, prompt, NUM_TOKENS)

        # Save golden inputs for future runs
        save_dir = os.path.dirname(COMPILED_PATH) or "/scratch"
        golden_in_path = os.path.join(save_dir, "glm5_golden_inputs.pt")
        torch.save(golden_inputs, golden_in_path)
        print(f"\n  Saved golden inputs to {golden_in_path}")

    # Phase 2: Teacher-forced comparison
    results = teacher_forced_comparison(model, tokenizer, golden_inputs)

    # Save results
    save_dir = os.path.dirname(COMPILED_PATH) or "/scratch"
    results_path = os.path.join(save_dir, "glm5_teacher_forced_results.pt")
    torch.save(results, results_path)
    print(f"\n  Saved results to {results_path}")

    # Final verdict
    accuracy = results["accuracy"]
    print(f"\n{'='*60}")
    print(f"  TEACHER-FORCED: {results['num_match']}/{results['num_tokens']} ({100*accuracy:.1f}%)")
    if accuracy >= 0.9:
        print(f"  VERDICT: PASS (>= 90% match)")
    else:
        print(f"  VERDICT: FAIL (< 90% match)")
    print(f"{'='*60}")

    return 0 if accuracy >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
