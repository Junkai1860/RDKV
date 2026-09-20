#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) evaluation for RDKV.

Supports 3 methods:
  - fullkv:      Full KV cache (upper bound)
  - ada_snapkv:  Adaptive SnapKV baseline
  - obkv:        OBKV with knapsack_split (our method)

Aligned with ReST-KV NIAH benchmark protocol.

Usage:
  python pred_niah_eval.py \
      --method obkv \
      --haystack-dir data/PaulGrahamEssays \
      --output-dir results_needle/results \
      --method-version obkv_B1024 \
      --s-len 2000 --e-len 32001 --step 400 \
      --token-budget 1024 --k-budget-ratio 0.5 \
      --shard 0 --num-shards 4 --seed 42
"""

import argparse
import gc
import glob
import json
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timezone

# ── Imports from obkv_fast library ───────────────────────────────────────
from obkv_fast import (
    DEFAULT_MODEL_PATH,
    greedy_decode_pt as greedy_decode,
    load_model_and_tokenizer,
    prefill_last_token,
    run_obkv,
    set_determinism,
)
from knapsack_solver import (
    DEFAULT_EPSILON_K,
    DEFAULT_EPSILON_V,
    load_epsilon_kv_from_calibration,
)


# ── Constants (aligned with ReST-KV) ────────────────────────────────────

NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
RETRIEVAL_QUESTION = "The best thing to do in San Francisco is: "
# Match ReST-KV: np.linspace(0, 100, num=10, endpoint=True) → 10 points
DOCUMENT_DEPTH_PERCENTS = np.round(
    np.linspace(0, 100, num=10, endpoint=True)
).astype(int).tolist()
# Result: [0, 11, 22, 33, 44, 56, 67, 78, 89, 100]
MAX_NEW_TOKENS = 30
FINAL_CONTEXT_LENGTH_BUFFER = 200
PERIOD_TOKENS = [13]  # LLaMA3

# Expected answer for scoring (the completion part of the needle, not the full needle)
EXPECTED_ANSWER_WORDS = set(
    "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
)


# ── NIAH functions (standalone, from ReST-KV) ───────────────────────────

def read_context_files(haystack_dir, tokenizer, max_context_length):
    """Read Paul Graham essays until we have enough tokens."""
    context = ""
    while len(tokenizer.encode(context, add_special_tokens=False)) < max_context_length:
        for filepath in sorted(glob.glob(os.path.join(haystack_dir, "*.txt"))):
            with open(filepath, "r") as f:
                context += f.read()
    return context


def encode_and_trim(tokenizer, context, context_length):
    """Tokenize context and truncate to context_length tokens."""
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) > context_length:
        context = tokenizer.decode(tokens[:context_length], skip_special_tokens=True)
    return context


def insert_needle(tokenizer, context, needle, depth_percent, context_length,
                  buffer=FINAL_CONTEXT_LENGTH_BUFFER):
    """Insert needle into context at the specified depth percentage."""
    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    context_length -= buffer
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]

        # Back up to nearest period token
        while tokens_new_context and tokens_new_context[-1] not in PERIOD_TOKENS:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        print("insertion at %d" % insertion_point)
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    new_context = tokenizer.decode(tokens_new_context, skip_special_tokens=True)
    return new_context


def generate_prompt(context, retrieval_question):
    """Generate prompt in ReST-KV format (NOT chat template)."""
    return (
        f"<|im_start|> This is a very long story book: <book> {context} </book>.\n"
        f" Based on the content of the book, Question: {retrieval_question}\n"
        f"Answer:"
    )


def result_exists(results_dir, method_version, context_length, depth_percent):
    """Check if result JSON already exists for this (length, depth) pair."""
    filename = f"{method_version}_len_{context_length}_depth_{int(depth_percent * 100)}_results.json"
    filepath = os.path.join(results_dir, method_version, filename)
    return os.path.exists(filepath)


def score_response(response):
    """Score response using word-set overlap with expected answer (0-1 scale).

    Matches ReST-KV visualize.py: intersection of response words with
    expected_answer words, divided by number of expected_answer words.
    """
    if not response:
        return 0.0
    response_words = set(response.lower().split())
    return len(response_words.intersection(EXPECTED_ANSWER_WORDS)) / len(EXPECTED_ANSWER_WORDS)


def save_result(results_dir, method_version, context_length, depth_percent, result):
    """Save a single NIAH result JSON file."""
    version_dir = os.path.join(results_dir, method_version)
    os.makedirs(version_dir, exist_ok=True)
    filename = f"{method_version}_len_{context_length}_depth_{int(depth_percent * 100)}_results.json"
    filepath = os.path.join(version_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"  Saved: {filepath}")


# ── Argument parsing ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NIAH evaluation for RDKV.")
    p.add_argument("--method", required=True, choices=["fullkv", "snapkv", "ada_snapkv", "obkv"])
    p.add_argument("--haystack-dir", required=True, help="Dir with Paul Graham essay .txt files")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-dir", required=True, help="Base results directory")
    p.add_argument("--method-version", required=True,
                    help="Version string for result path (e.g. fullkv, obkv_B1024)")
    p.add_argument("--s-len", type=int, default=2000, help="Start context length")
    p.add_argument("--e-len", type=int, default=32001, help="End context length (exclusive)")
    p.add_argument("--step", type=int, default=400, help="Context length step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    # SnapKV / Ada-SnapKV params
    p.add_argument("--token-budget", type=int, default=1024)
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--alpha-safeguard", type=float, default=0.2)
    # OBKV params
    p.add_argument("--k-budget-ratio", type=float, default=0.5)
    p.add_argument("--epsilon-path", default=None)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--score-pooling", type=int, default=0,
                    help="Avg pooling kernel size for token scores "
                         "(0=disabled, 5=SnapKV-style)")
    p.add_argument("--streaming", action="store_true", default=True,
                    help="Layer-streaming per-head prefill "
                         "(run_obkv). Always on — kept "
                         "for backwards-compat with the legacy CLI.")
    p.add_argument("--eviction-mode", choices=["joint"], default="joint",
                    help="Per-head joint knapsack over {0,2,4,8,16}.")
    p.add_argument("--v-score-type", choices=["attn_linear"], default="attn_linear",
                    help="V-side per-token score (sum_tau a over the recent window).")
    p.add_argument("--pool-padding", type=str, default="reflect",
                    choices=["reflect", "zero"])
    p.add_argument("--v-bit-options", type=str, default="0,2,4,8,16")
    p.add_argument("--k-bit-options", type=str, default="0,2,4,8,16")
    p.add_argument("--fp16-topk", type=int, default=0)
    return p.parse_args()


# ── GPU memory diagnostics ───────────────────────────────────────────────

def _print_gpu_memory(label):
    n = torch.cuda.device_count()
    print(f"  [GPU MEM] {label} -- {n} device(s)")
    for i in range(n):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB")


# ── Method: Full KV (using model.generate, matching ReST-KV) ─────────────

def run_generate(model, input_ids, max_new_tokens, eos_token_ids):
    """Run model.generate() — used by fullkv and ada_snapkv to match ReST-KV."""
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            output_attentions=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            eos_token_id=eos_token_ids,
        )
    # Return only the generated token IDs (exclude prompt)
    return output_ids[0][input_ids.shape[1]:].tolist()


def run_fullkv(model, input_ids, max_new_tokens, eos_token_ids, device):
    return run_generate(model, input_ids, max_new_tokens, eos_token_ids)


# ── Method: Ada-SnapKV (using model.generate with kvpress) ───────────────

def run_kvpress(model, input_ids, args, prompt_len, max_new_tokens, eos_token_ids, device):
    """Run inference with kvpress-based compression (SnapKV or Ada-SnapKV)."""
    from pred_kvpress_baseline import compression_ratio_for_budget
    from kvpress import AdaKVPress, SnapKVPress

    compression_ratio = compression_ratio_for_budget(prompt_len, args.token_budget)

    base_press = SnapKVPress(
        compression_ratio=compression_ratio,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
    )
    if args.method == "ada_snapkv":
        press = AdaKVPress(press=base_press, alpha_safeguard=args.alpha_safeguard)
    else:
        press = base_press

    with press(model):
        return run_generate(model, input_ids, max_new_tokens, eos_token_ids)


# ── Method: OBKV (thin wrapper around obkv_fast.run_obkv) ───────────────

def _parse_bit_options(spec):
    if not spec:
        return None
    bits = sorted({int(b.strip()) for b in spec.split(",") if b.strip()})
    if 0 not in bits:
        bits.insert(0, 0)
    return torch.tensor(bits, dtype=torch.float32)


def run_obkv_method(model, input_ids, args, epsilon_K, epsilon_V, max_new_tokens,
                    eos_token_ids, device):
    v_bit_options = _parse_bit_options(getattr(args, "v_bit_options", None))
    k_bit_options = _parse_bit_options(getattr(args, "k_bit_options", None))
    pool_kernel = (
        getattr(args, "score_pooling", 0)
        if getattr(args, "score_pooling", 0) > 0
        else 1
    )
    entry = run_obkv
    kwargs = dict(
        model=model,
        input_ids=input_ids,
        token_budget=args.token_budget,
        k_budget_ratio=args.k_budget_ratio,
        epsilon_K=epsilon_K,
        epsilon_V=epsilon_V,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        chunk_size=args.chunk_size,
        obs_window=args.obs_window,
        pool_kernel_size=pool_kernel,
        pool_padding=getattr(args, "pool_padding", "reflect"),
        v_bit_options=v_bit_options,
        k_bit_options=k_bit_options,
        fp16_topk=getattr(args, "fp16_topk", 0),
        device=device,
    )
    kwargs["eviction_mode"] = getattr(args, "eviction_mode", "joint")
    kwargs["v_score_type"] = getattr(args, "v_score_type", "attn_linear")
    return entry(**kwargs)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_determinism(args.seed)

    print("=== Needle-in-a-Haystack Evaluation ===")
    print(f"  Method:         {args.method}")
    print(f"  Method version: {args.method_version}")
    print(f"  Length range:   {args.s_len} - {args.e_len} (step={args.step})")
    print(f"  Depth percents: {DOCUMENT_DEPTH_PERCENTS}")
    print(f"  Shard:          {args.shard}/{args.num_shards}")
    print(f"  Token budget:   {args.token_budget}")
    if args.method == "obkv":
        print(f"  K ratio:        {args.k_budget_ratio}")
        print(f"  Obs window:     {args.obs_window}")
        print(f"  Score pool:     {args.score_pooling}")
    print(f"  Output:         {args.output_dir}")
    print()

    # ── Build context length list and shard ───────────────────────────
    all_lengths = list(range(args.s_len, args.e_len, args.step))
    # Shard by index
    shard_size = len(all_lengths) // args.num_shards
    start = args.shard * shard_size
    if args.shard < args.num_shards - 1:
        end = (args.shard + 1) * shard_size
    else:
        end = len(all_lengths)
    context_lengths = all_lengths[start:end]

    total_samples = len(context_lengths) * len(DOCUMENT_DEPTH_PERCENTS)
    print(f"  Shard {args.shard}: {len(context_lengths)} lengths "
          f"({context_lengths[0]}-{context_lengths[-1]}), "
          f"{total_samples} total samples")
    print()

    # ── Load model ───────────────────────────────────────────────────
    model_mode = "value_residual_probe" if args.method == "obkv" else "none"
    model, tokenizer, primary_device = load_model_and_tokenizer(
        args.model_path, model_mode, args.attn_implementation,
    )

    # NIAH EOS: eos_token_id + newline (matching ReST-KV)
    eos_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.encode("\n", add_special_tokens=False)[-1],
    ]

    # ── Load OBKV epsilon calibration ────────────────────────────────
    epsilon_K, epsilon_V = None, None
    if args.method == "obkv" and args.epsilon_path:
        epsilon_K, epsilon_V = load_epsilon_kv_from_calibration(args.epsilon_path)
        print(f"[OBKV] Loaded epsilon from {args.epsilon_path}")

    # ── Pre-read haystack text ───────────────────────────────────────
    max_context_length = max(context_lengths)
    print(f"[Haystack] Reading from {args.haystack_dir} (need {max_context_length} tokens)...")
    haystack_text = read_context_files(args.haystack_dir, tokenizer, max_context_length)
    haystack_tokens = len(tokenizer.encode(haystack_text, add_special_tokens=False))
    print(f"[Haystack] Loaded {haystack_tokens} tokens")
    print()

    # ── Main evaluation loop ─────────────────────────────────────────
    completed = 0
    skipped = 0
    oom_count = 0
    scores_all = []
    t_start = time.time()

    for context_length in context_lengths:
        for depth_percent in DOCUMENT_DEPTH_PERCENTS:
            # Checkpoint resume
            if result_exists(args.output_dir, args.method_version,
                             context_length, depth_percent):
                skipped += 1
                print(f"  [SKIP] len={context_length} depth={depth_percent}%")
                continue

            print(f"[{completed + skipped + 1}/{total_samples}] "
                  f"len={context_length} depth={depth_percent}%")

            # 1. Generate context with needle
            context = encode_and_trim(tokenizer, haystack_text, context_length)
            context = insert_needle(tokenizer, context, NEEDLE,
                                    depth_percent, context_length)

            # 2. Generate prompt (ReST-KV format, NOT chat template)
            prompt = generate_prompt(context, RETRIEVAL_QUESTION)

            # 3. Tokenize (with BOS, matching ReST-KV)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(primary_device)
            prompt_len = input_ids.shape[1]
            print(f"  prompt_len={prompt_len}")

            # 4. Inference
            test_start = time.time()
            generated_ids = []
            error = None

            try:
                if args.method == "fullkv":
                    generated_ids = run_fullkv(
                        model, input_ids, MAX_NEW_TOKENS,
                        eos_token_ids, primary_device,
                    )
                elif args.method in ("snapkv", "ada_snapkv"):
                    generated_ids = run_kvpress(
                        model, input_ids, args, prompt_len,
                        MAX_NEW_TOKENS, eos_token_ids, primary_device,
                    )
                elif args.method == "obkv":
                    generated_ids = run_obkv_method(
                        model, input_ids, args, epsilon_K, epsilon_V,
                        MAX_NEW_TOKENS, eos_token_ids, primary_device,
                    )
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                error = "OOM"
                print(f"  [OOM] len={context_length} depth={depth_percent}%")
                traceback.print_exc()
                _print_gpu_memory("at OOM")
                gc.collect()
                torch.cuda.empty_cache()

            test_elapsed = time.time() - test_start

            # 5. Decode response
            response = tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip() if generated_ids else ""

            # 6. Score
            score = score_response(response)
            scores_all.append(score)

            print(f"  Response: {response}")
            print(f"  Score: {score:.4f}  Duration: {test_elapsed:.1f}s")

            # 7. Save result
            result = {
                "model": args.model_path,
                "context_length": int(context_length),
                "depth_percent": float(depth_percent),
                "version": 1,
                "needle": NEEDLE,
                "model_response": response,
                "score": score,
                "test_duration_seconds": test_elapsed,
                "test_timestamp_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S%z"
                ),
            }
            if error:
                result["error"] = error
            save_result(args.output_dir, args.method_version,
                        context_length, depth_percent, result)

            completed += 1

            # Free memory
            del input_ids, generated_ids
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    avg_score = sum(scores_all) / len(scores_all) if scores_all else 0.0
    print(f"\n=== Summary ===")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  OOM:       {oom_count}")
    print(f"  Avg score: {avg_score:.4f}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
