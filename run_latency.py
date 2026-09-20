#!/usr/bin/env python3
"""benchmark_latency_v3.py

Single-GPU latency benchmark for Full KV vs OBKV (B=1024, accel config).
Paper efficiency section: measures TTFT, Decode Total, TPOT, Peak Memory.

Changes from v2:
  - Single-GPU only (no TP / device_map="auto")
  - Removed obkv_evict mode; only full_kv and obkv (accel packed decode)
  - B=1024, kr=0.5, K{0,4} V{0,2,4,8,16} explicit bit options
  - 1024 generated tokens, median of 3 repeats
  - Direct eviction + packing (no fake_quantize step)
  - V=16 tokens handled via DualZoneCache new-zone (no Triton kernel changes)
  - Peak memory tracking, TPOT monotonicity check
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from knapsack_solver import (
    DEFAULT_EPSILON_K,
    DEFAULT_EPSILON_V,
    knapsack_bit_allocation,
)
from obkv_accel.fast_decode import greedy_decode_fast
from obkv_accel.packing import build_packed_cache, DualZoneCache, TriZoneCache
from obkv_fast import (
    _perhead_select_and_pack,
    _streaming_prefill_and_pack,
    apply_score_pooling,
    detach_past_key_values,
    evict_past_key_values,
    greedy_decode_pt,
    greedy_decode_pt_with_perhead_masks,
    load_epsilon_kv_from_calibration,
    load_model_and_tokenizer,
    patch_llama31_rope_compat,
    prepare_past_key_values_for_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU latency benchmark v3: Full KV vs OBKV (B=1024, accel config).",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HF id.",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="Context lengths to benchmark.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help="Number of tokens to generate (including first token from prefill).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1024,
        help="Token budget B for OBKV eviction.",
    )
    parser.add_argument(
        "--k-budget-ratio",
        type=float,
        default=0.5,
        help="K-side budget ratio.",
    )
    parser.add_argument(
        "--k-options",
        nargs="+",
        type=int,
        default=[0, 2, 4, 8, 16],
        help="K-side bit-width options for knapsack.",
    )
    parser.add_argument(
        "--v-options",
        nargs="+",
        type=int,
        default=[0, 2, 4, 8, 16],
        help="V-side bit-width options for knapsack.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Chunk size for value_residual_probe scoring (legacy mode).",
    )
    parser.add_argument(
        "--obs-window",
        type=int,
        default=32,
        help="Observation window for two-phase probe (default 32).",
    )
    parser.add_argument(
        "--legacy-probe",
        action="store_true",
        help="Use legacy single-pass chunked probe (slow but verified).",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="Chunk size for chunked prefill (e.g. 8192). None = original single-pass. "
             "If OOM with 8192, try 4096.",
    )
    parser.add_argument(
        "--save-artifact",
        type=str,
        default=None,
        help="Save OBKV artifacts (packed cache) to this path for later reuse.",
    )
    parser.add_argument(
        "--load-artifact",
        type=str,
        default=None,
        help="Load pre-built OBKV artifacts from this path (skip prefill, fix T_eff).",
    )
    parser.add_argument(
        "--skip-full-kv",
        action="store_true",
        help="Skip full-KV baseline measurement (only measure OBKV).",
    )
    parser.add_argument(
        "--per-head",
        action="store_true",
        help="Use per-layer per-head OBKV artifact (each KV head selects its "
             "own top-k tokens before the shared V/K knapsack). Forces "
             "single-pass value_residual_probe prefill; --legacy-probe and "
             "--prefill-chunk-size are ignored for the OBKV measurement.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use Plan-A layer-streaming per-head prefill. Scores and packs "
             "each layer immediately after its forward pass, keeping only one "
             "layer's FP16 KV in memory at a time. Lower peak memory than "
             "--per-head.",
    )
    parser.add_argument(
        "--pool-kernel-size",
        type=int,
        default=5,
        help="Avg pooling kernel size applied to per-head scores (per-head mode).",
    )
    parser.add_argument(
        "--pool-padding",
        choices=["reflect", "zero"],
        default="reflect",
        help="Padding mode for score pooling. Matches run_longbench.py.",
    )
    parser.add_argument(
        "--v-score-type",
        choices=["attn_linear", "h_t", "v_block"],
        default="attn_linear",
        help="V-side per-token score form. 'attn_linear' (default, paper "
             "config: sum_tau a) is the only one supported by the cleaned "
             "supplementary code; 'v_block' / 'h_t' work only against the "
             "full unstripped code base.",
    )
    parser.add_argument(
        "--epsilon-path",
        type=str,
        default=None,
        help="Optional calibration artifact with per-layer K/V epsilons.",
    )
    parser.add_argument(
        "--eviction-mode",
        choices=["joint", "topk"],
        default="joint",
        help="Per-head eviction strategy: 'joint' (default) or 'topk' "
             "(per-head two-stage: top-k + per-head knapsack over {2,4,8,16}). "
             "Only effective with --per-head or --streaming.",
    )
    parser.add_argument(
        "--n-kept-multiplier",
        type=float,
        default=5.0,
        help="Coarse top-k multiplier used when --eviction-mode=topk.",
    )
    parser.add_argument(
        "--decode-blockwise-rdkv",
        action="store_true",
        help="Use Blockwise RDKV for generated decode KV under the same "
             "synthetic latency protocol.",
    )
    parser.add_argument(
        "--decode-block-size",
        type=int,
        default=128,
        help="Decode block size for --decode-blockwise-rdkv.",
    )
    parser.add_argument(
        "--decode-block-budget-tokens",
        type=int,
        default=1,
        help="Per-layer/per-head FP16-equivalent token budget per decode block.",
    )
    parser.add_argument(
        "--decode-correctness-check",
        action="store_true",
        help="Enable expensive packed-vs-decompressed checks in Blockwise RDKV.",
    )
    args = parser.parse_args()

    # --- Validation ---
    valid_bits = {0, 2, 4, 8, 16}
    for name, opts in [("--k-options", args.k_options), ("--v-options", args.v_options)]:
        invalid = [b for b in opts if b not in valid_bits]
        if invalid:
            parser.error(f"{name} contains invalid bit-widths {invalid}. Allowed: {sorted(valid_bits)}")
        if 0 not in opts:
            opts.insert(0, 0)

    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive.")
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.decode_blockwise_rdkv:
        if args.decode_block_size <= 0:
            parser.error("--decode-block-size must be positive.")
        if args.decode_block_budget_tokens <= 0:
            parser.error("--decode-block-budget-tokens must be positive.")
        if args.decode_block_budget_tokens > args.decode_block_size:
            parser.error("--decode-block-budget-tokens cannot exceed --decode-block-size.")

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def median_ms(values: Sequence[float]) -> float:
    return float(statistics.median(values) * 1000.0)


def build_random_input_ids(
    ctx_len: int, device: torch.device, bos_token_id: int,
) -> torch.Tensor:
    ids = torch.randint(1, 32000, (1, ctx_len), device=device, dtype=torch.long)
    ids[0, 0] = bos_token_id
    return ids


# ---------------------------------------------------------------------------
# Chunked prefill helpers
# ---------------------------------------------------------------------------

def _cache_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[2])


def chunked_prefill_full_kv(
    model,
    input_ids: torch.Tensor,
    prefill_chunk_size: int,
) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...], int]:
    """Process input_ids in chunks via model.model(), building KV cache incrementally.

    Returns (next_token_logits, detached_past_key_values, original_seq_len).
    """
    past_key_values = None
    last_hidden_state = None

    with torch.inference_mode():
        for chunk_start in range(0, input_ids.shape[1], prefill_chunk_size):
            chunk_end = min(chunk_start + prefill_chunk_size, input_ids.shape[1])
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            start = _cache_seq_length(past_key_values)
            cache_position = torch.arange(
                start,
                start + chunk_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            position_ids = cache_position.unsqueeze(0)
            outputs = model.model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_ids=position_ids,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden_state = outputs.last_hidden_state[:, -1, :]

    if last_hidden_state is None or past_key_values is None:
        raise RuntimeError("Chunked prefill produced no cache.")

    next_token_logits = model.lm_head(last_hidden_state).float()
    detached_past_key_values = detach_past_key_values(past_key_values)
    original_seq_len = int(input_ids.shape[1])
    return next_token_logits, detached_past_key_values, original_seq_len


def greedy_decode_after_first_token(
    model,
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    first_token_id: int,
    max_remaining_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: int,
) -> List[int]:
    """Greedy decode loop for Full KV after chunked prefill."""
    eos_set = {int(token_id) for token_id in eos_token_ids if token_id is not None}
    generated_ids = [first_token_id]
    if first_token_id in eos_set or max_remaining_tokens <= 0:
        return generated_ids

    next_token = torch.tensor([[first_token_id]], device=primary_device, dtype=torch.long)
    current_past_key_values = prepare_past_key_values_for_model(past_key_values)

    for step in range(max_remaining_tokens):
        model_kwargs = {
            "input_ids": next_token,
            "past_key_values": current_past_key_values,
            "use_cache": True,
            "return_dict": True,
            "position_ids": torch.tensor(
                [[decode_position_start + step]],
                device=primary_device,
                dtype=torch.long,
            ),
        }
        with torch.inference_mode():
            outputs = model(**model_kwargs)
        current_past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        if token_id in eos_set:
            break
        generated_ids.append(token_id)

    return generated_ids


def _cache_shapes(past_key_values) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    return [
        (tuple(key_states.shape), tuple(value_states.shape))
        for key_states, value_states in past_key_values
    ]


def validate_chunked_prefill_equivalence(
    model,
    input_ids: torch.Tensor,
    prefill_chunk_size: int,
) -> None:
    """Verify chunked prefill produces identical next-token as single-pass (Full KV)."""
    with torch.inference_mode():
        reference_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
    reference_logits = reference_outputs.logits[:, -1, :]
    reference_past_key_values = detach_past_key_values(reference_outputs.past_key_values)
    del reference_outputs

    clear_memory()

    chunked_logits, chunked_past_key_values, _ = chunked_prefill_full_kv(
        model=model,
        input_ids=input_ids,
        prefill_chunk_size=prefill_chunk_size,
    )

    reference_shapes = _cache_shapes(reference_past_key_values)
    chunked_shapes = _cache_shapes(chunked_past_key_values)
    if reference_shapes != chunked_shapes:
        raise AssertionError(
            "Chunked prefill cache shape mismatch: "
            f"reference={reference_shapes}, chunked={chunked_shapes}, "
            f"seq_len={input_ids.shape[1]}, chunk_size={prefill_chunk_size}"
        )

    reference_argmax = int(reference_logits.argmax(dim=-1).item())
    chunked_argmax = int(chunked_logits.argmax(dim=-1).item())
    if reference_argmax != chunked_argmax:
        raise AssertionError(
            "Chunked prefill next-token mismatch: "
            f"reference_argmax={reference_argmax}, chunked_argmax={chunked_argmax}, "
            f"seq_len={input_ids.shape[1]}, chunk_size={prefill_chunk_size}"
        )
    print(f"  [validation] Full KV chunked prefill equivalence OK (seq_len={input_ids.shape[1]})")

    del reference_past_key_values, chunked_past_key_values
    clear_memory()


def validate_obkv_chunked_equivalence(
    model,
    input_ids: torch.Tensor,
    prefill_chunk_size: int,
    obs_window: int,
    compute_device: torch.device,
    token_budget: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
) -> None:
    """Verify OBKV chunked Phase 1 produces equivalent artifact as single-pass."""
    common_kwargs = dict(
        model=model,
        input_ids=input_ids,
        obs_window=obs_window,
        compute_device=compute_device,
        token_budget=token_budget,
        k_budget_ratio=k_budget_ratio,
        k_bit_options=k_bit_options,
        v_bit_options=v_bit_options,
    )

    ref_artifact = build_obkv_artifact_twophase(**common_kwargs, prefill_chunk_size=None)
    ref_t_eff = ref_artifact["t_eff"]
    ref_argmax = int(ref_artifact["next_token_logits"].argmax(dim=-1).item())
    del ref_artifact
    clear_memory()

    chunked_artifact = build_obkv_artifact_twophase(**common_kwargs, prefill_chunk_size=prefill_chunk_size)
    chunked_t_eff = chunked_artifact["t_eff"]
    chunked_argmax = int(chunked_artifact["next_token_logits"].argmax(dim=-1).item())
    del chunked_artifact
    clear_memory()

    t_eff_diff = abs(ref_t_eff - chunked_t_eff)
    t_eff_rel = t_eff_diff / max(ref_t_eff, 1)
    if t_eff_rel > 0.01:
        print(
            f"  [validation] WARNING: OBKV t_eff mismatch beyond 1%: "
            f"reference={ref_t_eff}, chunked={chunked_t_eff} (diff={t_eff_diff}, {t_eff_rel:.2%})"
        )
    elif t_eff_diff > 0:
        print(
            f"  [validation] OBKV t_eff minor diff (numerical noise): "
            f"reference={ref_t_eff}, chunked={chunked_t_eff} (diff={t_eff_diff}, {t_eff_rel:.2%})"
        )
    if ref_argmax != chunked_argmax:
        print(
            f"  [validation] WARNING: OBKV next-token mismatch: "
            f"reference={ref_argmax}, chunked={chunked_argmax}"
        )
    print(f"  [validation] OBKV chunked prefill validation done (seq_len={input_ids.shape[1]})")


def generation_config_variants(tokenizer, max_new_tokens: int, vocab_size: int):
    common = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": None,
    }
    return [
        ("eos_empty_list", GenerationConfig(eos_token_id=[], **common)),
        ("eos_impossible_id", GenerationConfig(eos_token_id=vocab_size + 1024, **common)),
    ]


def run_generate_checked(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    expected_generated: int,
    config_variant: Optional[Tuple[str, GenerationConfig]] = None,
):
    variants = (
        [config_variant]
        if config_variant is not None
        else generation_config_variants(tokenizer, max_new_tokens, model.config.vocab_size)
    )
    errors: List[str] = []
    for variant_name, generation_config in variants:
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )
            generated = int(outputs.shape[1] - input_ids.shape[1])
            if generated != expected_generated:
                errors.append(
                    f"{variant_name}: generated {generated} tokens, expected {expected_generated}"
                )
                continue
            return outputs, variant_name
        except Exception as exc:
            errors.append(f"{variant_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def resolve_generation_variant(
    model, tokenizer, input_ids, max_new_tokens, expected_generated,
):
    for variant_name, generation_config in generation_config_variants(
        tokenizer, max_new_tokens, model.config.vocab_size,
    ):
        try:
            run_generate_checked(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                expected_generated=expected_generated,
                config_variant=(variant_name, generation_config),
            )
            return variant_name, generation_config
        except Exception:
            continue
    raise RuntimeError("No generation config variant produced the requested number of tokens.")


# ---------------------------------------------------------------------------
# Bit allocation
# ---------------------------------------------------------------------------

def compute_obkv_bit_allocation(
    token_scores: torch.Tensor,
    channel_scores: torch.Tensor,
    token_budget: int,
    head_dim: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_budget = 2 * token_budget * head_dim * 16
    effective_bpt = total_budget / (token_scores.shape[0] * head_dim * 2)
    v_avg = (1.0 - k_budget_ratio) * effective_bpt * 2
    v_bits = knapsack_bit_allocation(
        token_scores,
        v_avg,
        epsilon=DEFAULT_EPSILON_V,
        bit_options=v_bit_options,
    )
    # Mirror run_obkv's method-B ledger fix: v=16 tokens go to the FP16 new
    # zone at D*16 bits each, so deduct that from k_budget_bits and divide
    # by the compressed-zone size only.
    t_eff = int((v_bits > 0).sum().item())
    n_fp16 = int((v_bits == 16).sum().item())
    t_eff_compressed = t_eff - n_fp16
    if t_eff == 0:
        k_bits = torch.full((head_dim,), 16, dtype=torch.long)
    else:
        k_budget_bits = k_budget_ratio * total_budget
        k_budget_remaining = k_budget_bits - n_fp16 * head_dim * 16
        if t_eff_compressed == 0:
            k_bits = torch.full((head_dim,), 16, dtype=torch.long)
        elif k_budget_remaining <= 0:
            k_bits = torch.zeros(head_dim, dtype=torch.long)
        else:
            k_avg_sync = k_budget_remaining / (head_dim * t_eff_compressed)
            k_bits = knapsack_bit_allocation(
                channel_scores,
                k_avg_sync,
                epsilon=DEFAULT_EPSILON_K,
                bit_options=k_bit_options,
            )
    return k_bits, v_bits


# ---------------------------------------------------------------------------
# OBKV artifact builder (with V=16 new-zone handling)
# ---------------------------------------------------------------------------

def build_obkv_artifact(
    model,
    input_ids: torch.Tensor,
    chunk_size: int,
    compute_device: torch.device,
    token_budget: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
) -> Dict[str, object]:
    next_token_logits, past_key_values, selector_scores, prefill_meta = run_prefill(
        model=model,
        input_ids=input_ids,
        mode="value_residual_probe",
        chunk_size=chunk_size,
        compute_device=compute_device,
        sample_idx=1,
        task_name="latency_benchmark_v3",
        compute_channel_scores=True,
        probe_ids_override=None,
    )

    # Aggregate token scores (V-side) and channel scores (K-side)
    token_scores = torch.zeros(input_ids.shape[1], dtype=torch.float32)
    for layer_scores in selector_scores.values():
        token_scores += layer_scores.float().cpu().sum(dim=0)

    # SnapKV-style reflect pooling on token scores before knapsack.
    token_scores = apply_score_pooling(
        token_scores, pool_type="avg", kernel_size=7, padding_mode="reflect",
    )

    channel_scores_per_layer = prefill_meta["channel_scores_per_layer"]
    first_ch = next(iter(channel_scores_per_layer.values()))
    head_dim = int(first_ch.shape[-1])
    channel_scores = torch.zeros(head_dim, dtype=torch.float32)
    for ch_scores in channel_scores_per_layer.values():
        channel_scores += ch_scores.float().cpu()

    # Knapsack allocation
    k_bits, v_bits = compute_obkv_bit_allocation(
        token_scores=token_scores,
        channel_scores=channel_scores,
        token_budget=token_budget,
        head_dim=head_dim,
        k_budget_ratio=k_budget_ratio,
        k_bit_options=k_bit_options,
        v_bit_options=v_bit_options,
    )

    # Evict v=0 tokens
    keep_mask = v_bits > 0
    keep_ids = keep_mask.nonzero(as_tuple=True)[0].sort().values
    t_eff = int(keep_ids.shape[0])
    if t_eff == 0:
        raise RuntimeError("OBKV kept zero tokens after eviction.")

    past_kv_kept = evict_past_key_values(past_key_values, keep_ids)
    kept_v_bits = v_bits[keep_ids]
    # TriZone: v=16 K goes into packed, v=16 V goes into new_v_only.
    # ``build_packed_cache`` handles both internally — no external split.
    n_fp16 = int((kept_v_bits == 16).sum().item())  # diagnostic only

    packed_cache = build_packed_cache(
        past_kv_kept, kept_v_bits, k_bits=k_bits,
        original_seq_len=int(input_ids.shape[1]),
    )

    return {
        "packed_cache": packed_cache,
        "next_token_logits": next_token_logits,
        "original_seq_len": int(input_ids.shape[1]),
        "t_eff": t_eff,
        "n_fp16": n_fp16,
    }


def _score_one_layer(
    aw: torch.Tensor,         # [B, H_kv, q_per_kv, obs_window, T] attention weights
    q: torch.Tensor,          # [B, H_kv, q_per_kv, obs_window, D] post-RoPE query
    o: torch.Tensor,          # [B, H_kv, q_per_kv, obs_window, D] attn output (from hook)
    v: torch.Tensor,          # [B, H_kv, T, D] values
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-layer token and channel scores.

    Attention output `o` is provided directly from o_proj hook (no aw@V recomputation).
    Uses in-place ops for residual_norm_sq to minimize temporaries.
    """
    # --- ||q||^2 / d ---
    q_norm = q.square().sum(dim=-1, keepdim=True) / head_dim
    # [B, H_kv, q_per_kv, obs_window, 1]

    # --- ||v-o||^2 via in-place on o_dot_v buffer ---
    v_norm_sq = v.square().sum(dim=-1, keepdim=True).unsqueeze(2).transpose(-1, -2)
    # [B, H_kv, 1, 1, T]
    o_norm_sq = o.square().sum(dim=-1, keepdim=True)
    # [B, H_kv, q_per_kv, obs_window, 1]
    residual = torch.matmul(o, v.unsqueeze(2).transpose(-1, -2))
    # [B, H_kv, q_per_kv, obs_window, T]
    residual.mul_(-2.0).add_(v_norm_sq).add_(o_norm_sq).clamp_min_(0.0)
    del v_norm_sq, o_norm_sq

    # --- contrib = a^2 * (||q||^2/d) * ||v-o||^2 ---
    contrib = aw.square() * q_norm * residual
    del residual

    # --- Token scores ---
    layer_token = contrib.sum(dim=(2, 3)).float()  # [B, H_kv, T] fp32

    # --- Channel scores (depends on base_t from contrib, cannot be separated) ---
    q_norm_sq = q_norm * head_dim
    base_t = contrib.sum(dim=-1, keepdim=True) / q_norm_sq.clamp(min=1e-10)
    del contrib
    q_sq = q.float().square()
    layer_channel = (q_sq * base_t.float()).sum(dim=(1, 2, 3))  # [B, D] fp32
    del q_sq, base_t

    return layer_token, layer_channel


def _compute_hook_based_scores(
    captured_q: dict,
    past_key_values,
    num_layers: int,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    obs_window: int,
    T: int,
    attentions: tuple = (),
    captured_o: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Post-forward score computation using model-returned attention weights
    and hook-captured attention outputs.

    - Attention weights from output_attentions=True (Q@K^T+softmax done by model)
    - Attention output `o` from o_proj hook (aw@V done by model)
    - Only remaining matmul: o@V^T for residual norm

    Returns (token_scores [T], channel_scores [head_dim]) on CPU, float32.
    """
    dev = past_key_values[0][0].device
    attentions_list = list(attentions)

    # GPU accumulators — no CPU sync until the very end
    token_scores_gpu = torch.zeros(T, device=dev, dtype=torch.float32)
    channel_scores_gpu = torch.zeros(head_dim, device=dev, dtype=torch.float32)

    for layer_idx in range(num_layers):
        aw_raw = attentions_list[layer_idx]
        attentions_list[layer_idx] = None  # free ~250MB at 128K

        q_raw = captured_q.pop(layer_idx)              # [B, H_q, obs_window, D]
        o_raw = captured_o.pop(layer_idx)              # [B, H_q, obs_window, D]
        v = past_key_values[layer_idx][1]              # [B, H_kv, T, D]
        B = q_raw.shape[0]

        # Reshape to GQA groups: [B, H_kv, q_per_kv, obs_window, ...]
        aw = aw_raw.view(B, num_kv_heads, q_per_kv, obs_window, T)
        q = q_raw.view(B, num_kv_heads, q_per_kv, obs_window, head_dim)
        o = o_raw.view(B, num_kv_heads, q_per_kv, obs_window, head_dim)

        layer_token, layer_channel = _score_one_layer(aw, q, o, v, head_dim)
        del aw, q, o

        token_scores_gpu += layer_token.sum(dim=(0, 1))   # [T]
        channel_scores_gpu += layer_channel.sum(dim=0)     # [D]

    # Single CPU transfer at the end
    token_scores = token_scores_gpu.cpu()
    channel_scores = channel_scores_gpu.cpu()
    del token_scores_gpu, channel_scores_gpu

    # Normalization for tail-only probes: position n is seen by
    # min(obs_window, T-n) probes due to causal masking.
    correct_norm = torch.full((T,), float(obs_window), dtype=torch.float32)
    for i in range(obs_window):
        correct_norm[T - obs_window + i] = float(obs_window - i)
    token_scores /= correct_norm

    return token_scores, channel_scores


def build_obkv_artifact_twophase(
    model,
    input_ids: torch.Tensor,
    obs_window: int,
    compute_device: torch.device,
    token_budget: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
    prefill_chunk_size: Optional[int] = None,
) -> Dict[str, object]:
    """Two-phase prefill: fast SDPA for bulk, hook-based probe for last obs_window tokens."""
    T = input_ids.shape[1]
    device = input_ids.device

    # ---- Phase 1: fast prefill, NO patching (native SDPA) ----
    prefill_len = T - obs_window
    if prefill_chunk_size is not None and prefill_len > prefill_chunk_size:
        # Chunked Phase 1: use model.model() to skip lm_head on intermediate chunks
        past_kv = None
        with torch.inference_mode():
            for chunk_start in range(0, prefill_len, prefill_chunk_size):
                chunk_end = min(chunk_start + prefill_chunk_size, prefill_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                start = _cache_seq_length(past_kv)
                cache_position = torch.arange(
                    start,
                    start + chunk_ids.shape[1],
                    device=device,
                    dtype=torch.long,
                )
                position_ids = cache_position.unsqueeze(0)
                out1 = model.model(
                    input_ids=chunk_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    return_dict=True,
                )
                past_kv = out1.past_key_values
                del out1
    else:
        # Original single-pass Phase 1 (use model.model() to skip lm_head)
        with torch.inference_mode():
            out1 = model.model(
                input_ids=input_ids[:, :-obs_window],
                use_cache=True,
                return_dict=True,
            )
        past_kv = out1.past_key_values
        del out1

    # ---- Phase 2: hook-based probe (post-RoPE q capture + native forward) ----
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    q_per_kv = num_q_heads // num_kv_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_q_heads)
    num_layers = config.num_hidden_layers

    # Pre-compute RoPE cos/sin for positions [T-obs_window, ..., T-1]
    position_ids = torch.arange(T - obs_window, T, device=device).unsqueeze(0)
    rotary_emb_fn = getattr(model.model, "rotary_emb", None)
    if rotary_emb_fn is None:
        rotary_emb_fn = model.model.layers[0].self_attn.rotary_emb
    dummy_x = torch.zeros(1, obs_window, head_dim, device=device, dtype=model.dtype)
    sig = inspect.signature(rotary_emb_fn.forward)
    if "position_ids" in sig.parameters:
        rope_cos, rope_sin = rotary_emb_fn(dummy_x, position_ids)
    else:
        rope_cos, rope_sin = rotary_emb_fn(dummy_x, seq_len=T)
        rope_cos = rope_cos[position_ids.squeeze(0)]
        rope_sin = rope_sin[position_ids.squeeze(0)]

    # Register hooks: q_proj (post-RoPE query) and o_proj (attention output)
    captured_post_rope_q = {}
    captured_attn_output = {}

    def make_q_hook(layer_idx, cos, sin, n_heads, hdim):
        def hook_fn(module, input, output):
            B, S, _ = output.shape
            q = output.view(B, S, n_heads, hdim).transpose(1, 2)
            q_rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
            captured_post_rope_q[layer_idx] = q_rotated.detach()
        return hook_fn

    def make_o_hook(layer_idx, n_heads, hdim):
        def hook_fn(module, input, output):
            x = input[0]
            B, S, _ = x.shape
            captured_attn_output[layer_idx] = x.view(B, S, n_heads, hdim).transpose(1, 2).detach()
        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.q_proj.register_forward_hook(
            make_q_hook(layer_idx, rope_cos, rope_sin, num_q_heads, head_dim)
        )
        hooks.append(h)
        h = layer.self_attn.o_proj.register_forward_hook(
            make_o_hook(layer_idx, num_q_heads, head_dim)
        )
        hooks.append(h)

    # Temporarily switch to eager attention for output_attentions support
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn._attn_implementation = "eager"

    with torch.inference_mode():
        out2 = model(
            input_ids=input_ids[:, -obs_window:],
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
    for h in hooks:
        h.remove()

    # Restore SDPA for decode
    model.config._attn_implementation = "sdpa"
    for layer in model.model.layers:
        layer.self_attn._attn_implementation = "sdpa"

    next_token_logits = out2.logits[:, -1, :]
    past_key_values = out2.past_key_values

    # Score computation: attn weights from model, o from hook, only o@V^T remains
    token_scores, channel_scores = _compute_hook_based_scores(
        captured_q=captured_post_rope_q,
        past_key_values=past_key_values,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        q_per_kv=q_per_kv,
        head_dim=head_dim,
        obs_window=obs_window,
        T=T,
        attentions=out2.attentions,
        captured_o=captured_attn_output,
    )
    del out2

    # SnapKV-style reflect pooling on token scores before knapsack.
    token_scores = apply_score_pooling(
        token_scores, pool_type="avg", kernel_size=7, padding_mode="reflect",
    )

    # ---- Phase 3: knapsack + evict + pack ----
    k_bits, v_bits = compute_obkv_bit_allocation(
        token_scores=token_scores,
        channel_scores=channel_scores,
        token_budget=token_budget,
        head_dim=head_dim,
        k_budget_ratio=k_budget_ratio,
        k_bit_options=k_bit_options,
        v_bit_options=v_bit_options,
    )

    # Evict v=0 tokens
    keep_mask = v_bits > 0
    keep_ids = keep_mask.nonzero(as_tuple=True)[0].sort().values
    t_eff = int(keep_ids.shape[0])
    if t_eff == 0:
        raise RuntimeError("OBKV kept zero tokens after eviction.")

    past_kv_kept = evict_past_key_values(past_key_values, keep_ids)
    kept_v_bits = v_bits[keep_ids]

    # TriZone: single call handles v=16 split internally.
    n_fp16 = int((kept_v_bits == 16).sum().item())  # diagnostic only

    packed_cache = build_packed_cache(
        past_kv_kept, kept_v_bits, k_bits=k_bits,
        original_seq_len=T,
    )

    return {
        "packed_cache": packed_cache,
        "next_token_logits": next_token_logits,
        "original_seq_len": T,
        "t_eff": t_eff,
        "n_fp16": n_fp16,
    }


# ---------------------------------------------------------------------------
# Per-layer per-head artifact builder
# ---------------------------------------------------------------------------


def build_obkv_perhead_artifact(
    model,
    input_ids: torch.Tensor,
    chunk_size: int,
    compute_device: torch.device,
    token_budget: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
    obs_window: int = 32,
    pool_kernel_size: int = 7,
    pool_padding: str = "reflect",
    prefill_chunk_size: Optional[int] = None,
    eviction_mode: str = "joint",
    n_kept_multiplier: float = 5.0,
) -> Dict[str, object]:
    """Per-head artifact builder.

    Now uses the two-phase hook-based prefill scorer (bulk via native
    SDPA/FA2 + hook scoring on obs_window tail) which returns per-layer
    per-head scores directly. Much faster prefill vs single-pass patched
    value_residual_probe.
    """
    T = int(input_ids.shape[1])

    def _sync_now():
        sync_cuda()
        return time.perf_counter()

    t0 = _sync_now()

    use_twophase = obs_window > 0 and T > obs_window

    if use_twophase:
        next_token_logits, past_kv, scores_per_layer, channel_scores_per_layer, _head_dim = (
            _twophase_prefill_score(
                model=model,
                input_ids=input_ids,
                obs_window=obs_window,
                prefill_chunk_size=prefill_chunk_size,
                device=compute_device,
            )
        )
    else:
        probe_ids_override: Optional[torch.LongTensor] = None
        if obs_window > 0 and T > obs_window:
            probe_ids_override = torch.arange(T - obs_window, T, dtype=torch.long)
        next_token_logits, past_kv, scores_per_layer, prefill_meta = run_prefill(
            model=model,
            input_ids=input_ids,
            mode="value_residual_probe",
            chunk_size=chunk_size,
            compute_device=compute_device,
            compute_channel_scores=True,
            probe_ids_override=probe_ids_override,
        )
        channel_scores_per_layer = prefill_meta["channel_scores_per_layer"]
    num_layers = len(past_kv)

    t_after_prefill = _sync_now()

    past_kv = list(past_kv)
    packed_layers: List = []
    new_v_only_list: List[Optional[torch.Tensor]] = []
    per_layer_t_eff: List[int] = []
    n_v16_layers = 0

    for layer_idx in range(num_layers):
        K, V = past_kv[layer_idx]
        layer_scores = scores_per_layer[layer_idx]
        ch_scores = channel_scores_per_layer[layer_idx]

        packed_layer, new_v_only, t_eff = _perhead_select_and_pack(
            K, V, layer_scores, ch_scores,
            eviction_mode=eviction_mode,
            token_budget=token_budget,
            k_budget_ratio=k_budget_ratio,
            obs_window=obs_window,
            pool_kernel_size=pool_kernel_size,
            pool_padding=pool_padding,
            v_bit_options=v_bit_options,
            k_bit_options=k_bit_options,
            eps_V=DEFAULT_EPSILON_V,
            eps_K=DEFAULT_EPSILON_K,
            n_kept_multiplier=n_kept_multiplier,
        )

        packed_layers.append(packed_layer)
        new_v_only_list.append(new_v_only)
        per_layer_t_eff.append(t_eff)
        if new_v_only is not None:
            n_v16_layers += 1

        past_kv[layer_idx] = (None, None)
        del K, V

    t_after_pack = _sync_now()

    dual_cache = TriZoneCache(
        packed=tuple(packed_layers),
        new_v_only=list(new_v_only_list),
        new_both_k=[None] * num_layers,
        new_both_v=[None] * num_layers,
        original_seq_len=T,
    )

    t_after_cache_build = _sync_now()

    prefill_ms = (t_after_prefill - t0) * 1000.0
    layer_loop_ms = (t_after_pack - t_after_prefill) * 1000.0
    cache_build_ms = (t_after_cache_build - t_after_pack) * 1000.0
    ttft_ms = prefill_ms + layer_loop_ms + cache_build_ms
    mean_t_eff = (sum(per_layer_t_eff) / max(1, len(per_layer_t_eff))) if per_layer_t_eff else 0.0
    print(
        f"  [TIMING perhead_artifact] T={T} mean_T_eff={mean_t_eff:.1f} "
        f"prefill={prefill_ms:.0f}ms "
        f"layer_loop={layer_loop_ms:.0f}ms "
        f"cache_build={cache_build_ms:.0f}ms "
        f"(TTFT={ttft_ms:.0f}ms)",
        flush=True,
    )

    return {
        "packed_cache": dual_cache,
        "next_token_logits": next_token_logits,
        "original_seq_len": T,
        "t_eff": per_layer_t_eff[0] if per_layer_t_eff else 0,
        "n_fp16": n_v16_layers,
    }


def build_obkv_streaming_artifact(
    model,
    input_ids: torch.Tensor,
    compute_device: torch.device,
    token_budget: int,
    k_budget_ratio: float,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
    obs_window: int = 32,
    pool_kernel_size: int = 5,
    pool_padding: str = "reflect",
    eviction_mode: str = "joint",
    n_kept_multiplier: float = 5.0,
    epsilon_K: Optional[Dict[int, float]] = None,
    epsilon_V: Optional[Dict[int, float]] = None,
    v_score_type: str = "v_block",
    **_ignored,
) -> Dict[str, object]:
    """Streaming per-head artifact builder (Plan A).

    Forwards one layer at a time, scoring and packing immediately before
    releasing that layer's FP16 KV. Returns the same artifact dict as
    build_obkv_perhead_artifact so measure_obkv can use it unchanged.
    """
    T = int(input_ids.shape[1])

    torch.cuda.reset_peak_memory_stats(compute_device)
    sync_cuda()

    dual_cache, next_token_logits, per_layer_t_eff, td = _streaming_prefill_and_pack(
        model=model,
        input_ids=input_ids,
        token_budget=token_budget,
        k_budget_ratio=k_budget_ratio,
        epsilon_K=epsilon_K,
        epsilon_V=epsilon_V,
        obs_window=obs_window,
        pool_kernel_size=pool_kernel_size,
        pool_padding=pool_padding,
        v_bit_options=v_bit_options,
        k_bit_options=k_bit_options,
        device=compute_device,
        eviction_mode=eviction_mode,
        n_kept_multiplier=n_kept_multiplier,
        v_score_type=v_score_type,
    )

    sync_cuda()
    prefill_peak_gb = torch.cuda.max_memory_allocated(compute_device) / 1e9

    t_streaming = td["t_streaming"]
    t_cache_build = td["t_cache_build"]
    ttft_ms = (t_streaming + t_cache_build) * 1000.0
    mean_t_eff = sum(per_layer_t_eff) / max(1, len(per_layer_t_eff))
    print(
        f"  [TIMING streaming_artifact] T={T} mean_T_eff={mean_t_eff:.1f} "
        f"streaming={t_streaming*1000:.0f}ms "
        f"cache_build={t_cache_build*1000:.0f}ms "
        f"(TTFT={ttft_ms:.0f}ms) "
        f"prefill_peak={prefill_peak_gb:.2f}GB",
        flush=True,
    )

    return {
        "packed_cache": dual_cache,
        "next_token_logits": next_token_logits,
        "original_seq_len": T,
        "t_eff": per_layer_t_eff[0] if per_layer_t_eff else 0,
        "n_fp16": 0,
        "prefill_peak_gb": prefill_peak_gb,
        "ttft_ms": round(ttft_ms, 1),
    }


# ---------------------------------------------------------------------------
# Full KV measurement
# ---------------------------------------------------------------------------

def measure_full_kv(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_tokens: int,
    warmup: int,
    repeats: int,
    prefill_chunk_size: Optional[int] = None,
) -> Dict[str, object]:
    # --- Chunked prefill path ---
    if prefill_chunk_size is not None:
        decode_kwargs = dict(
            model=model,
            max_remaining_tokens=max(num_tokens - 1, 0),
            eos_token_ids=[],
            primary_device=input_ids.device,
        )

        # Warmup
        for _ in range(warmup):
            logits, pkv, seq_len = chunked_prefill_full_kv(model, input_ids, prefill_chunk_size)
            first_tid = int(logits.argmax(dim=-1).item())
            greedy_decode_after_first_token(
                past_key_values=pkv, first_token_id=first_tid,
                decode_position_start=seq_len, **decode_kwargs,
            )

        # Measurement
        prefill_times: List[float] = []
        decode_times: List[float] = []
        peak_memories: List[float] = []
        num_generated = num_tokens

        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats()

            sync_cuda()
            t0 = time.perf_counter()
            logits, pkv, seq_len = chunked_prefill_full_kv(model, input_ids, prefill_chunk_size)
            sync_cuda()
            t1 = time.perf_counter()
            prefill_times.append(t1 - t0)

            first_tid = int(logits.argmax(dim=-1).item())

            sync_cuda()
            t2 = time.perf_counter()
            generated_ids = greedy_decode_after_first_token(
                past_key_values=pkv, first_token_id=first_tid,
                decode_position_start=seq_len, **decode_kwargs,
            )
            sync_cuda()
            t3 = time.perf_counter()
            decode_times.append(t3 - t2)
            num_generated = len(generated_ids)

            peak_memories.append(torch.cuda.max_memory_allocated() / 1e9)

        prefill_total_ms = median_ms(prefill_times)
        decode_total_ms = median_ms(decode_times)
        decode_tokens = max(num_generated - 1, 1)
        return {
            "status": "ok",
            "ttft_ms": round(prefill_total_ms, 1),
            "decode_total_ms": round(decode_total_ms, 1),
            "tpot_ms": round(decode_total_ms / decode_tokens, 2),
            "peak_memory_gb": round(max(peak_memories), 2),
            "prefill_impl": "chunked_model_forward",
            "decode_impl": "greedy_forward_loop",
        }

    # --- Original single-pass path ---
    ttft_variant = resolve_generation_variant(
        model, tokenizer, input_ids, max_new_tokens=1, expected_generated=1,
    )
    total_variant = resolve_generation_variant(
        model, tokenizer, input_ids, max_new_tokens=num_tokens, expected_generated=num_tokens,
    )

    # Warmup
    for _ in range(warmup):
        run_generate_checked(
            model, tokenizer, input_ids,
            max_new_tokens=1, expected_generated=1, config_variant=ttft_variant,
        )
        run_generate_checked(
            model, tokenizer, input_ids,
            max_new_tokens=num_tokens, expected_generated=num_tokens, config_variant=total_variant,
        )

    # Measurement
    ttft_times: List[float] = []
    total_times: List[float] = []
    peak_memories: List[float] = []
    num_generated = num_tokens

    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()

        sync_cuda()
        t0 = time.perf_counter()
        run_generate_checked(
            model, tokenizer, input_ids,
            max_new_tokens=1, expected_generated=1, config_variant=ttft_variant,
        )
        sync_cuda()
        t1 = time.perf_counter()
        ttft_times.append(t1 - t0)

        sync_cuda()
        t2 = time.perf_counter()
        outputs, _ = run_generate_checked(
            model, tokenizer, input_ids,
            max_new_tokens=num_tokens, expected_generated=num_tokens,
            config_variant=total_variant,
        )
        sync_cuda()
        t3 = time.perf_counter()
        total_times.append(t3 - t2)
        num_generated = int(outputs.shape[1] - input_ids.shape[1])

        peak_memories.append(torch.cuda.max_memory_allocated() / 1e9)

    ttft_ms = median_ms(ttft_times)
    total_ms = median_ms(total_times)
    decode_total_ms = max(total_ms - ttft_ms, 0.0)
    decode_tokens = max(num_generated - 1, 1)
    return {
        "status": "ok",
        "ttft_ms": round(ttft_ms, 1),
        "decode_total_ms": round(decode_total_ms, 1),
        "tpot_ms": round(decode_total_ms / decode_tokens, 2),
        "peak_memory_gb": round(max(peak_memories), 2),
    }


# ---------------------------------------------------------------------------
# OBKV measurement
# ---------------------------------------------------------------------------


def _run_decompress_decode(
    *,
    model,
    artifact: Dict[str, object],
    max_new_tokens: int,
    eos_token_ids: List[int],
    primary_device: torch.device,
) -> List[int]:
    """Mirror obkv_fast.run_obkv_perhead_streaming's OBKV_DECOMPRESS_DECODE
    branch: dequantise the TriZoneCache to fp16 past_kv and run plain HF
    model.forward decode. Used for the trizone-vs-no-trizone ablation.
    """
    from obkv_accel.trizone_decompress import unpack_trizone_to_past_kv

    dual_cache = artifact["packed_cache"]
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    H_q = cfg.num_attention_heads

    real_devs = {p.device for p in model.parameters() if p.device.type != "meta"}
    multi_dev = len(real_devs) > 1
    decompress_dev = None if multi_dev else primary_device

    past_kv, per_layer_masks = unpack_trizone_to_past_kv(
        dual_cache,
        head_dim=head_dim,
        device=decompress_dev,
        return_masks=True,
        H_q=H_q,
        max_new_tokens=max_new_tokens,
    )

    decode_position_start = artifact["original_seq_len"]
    next_token_logits = artifact["next_token_logits"]

    if any(m is not None for m in per_layer_masks):
        assert all(m is not None for m in per_layer_masks), (
            "mixed shared / per-head packed layers in one cache — bug in "
            "unpack_trizone_to_past_kv dispatcher or upstream packing"
        )
        per_layer_base_lens = [
            int(pl.T_eff_k if pl.T_eff_k > 0 else pl.T_eff)
            for pl in dual_cache.packed
        ]
        orig_cfg = (
            hasattr(model.config, "_attn_implementation"),
            getattr(model.config, "_attn_implementation", None),
        )
        orig_layer = [
            (
                hasattr(layer.self_attn, "_attn_implementation"),
                getattr(layer.self_attn, "_attn_implementation", None),
            )
            for layer in model.model.layers
        ]
        try:
            model.config._attn_implementation = "sdpa"
            for layer in model.model.layers:
                layer.self_attn._attn_implementation = "sdpa"
            return greedy_decode_pt_with_perhead_masks(
                model=model,
                past_key_values=past_kv,
                next_token_logits=next_token_logits,
                per_layer_masks=per_layer_masks,
                per_layer_base_lens=per_layer_base_lens,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                primary_device=primary_device,
                decode_position_start=decode_position_start,
            )
        finally:
            had, val = orig_cfg
            if had:
                model.config._attn_implementation = val
            else:
                try:
                    delattr(model.config, "_attn_implementation")
                except AttributeError:
                    pass
            for layer, (had_l, val_l) in zip(model.model.layers, orig_layer):
                if had_l:
                    layer.self_attn._attn_implementation = val_l
                else:
                    try:
                        delattr(layer.self_attn, "_attn_implementation")
                    except AttributeError:
                        pass
    return greedy_decode_pt(
        model=model,
        past_key_values=past_kv,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        primary_device=primary_device,
        decode_position_start=decode_position_start,
    )


def measure_obkv(
    model,
    input_ids: torch.Tensor,
    num_tokens: int,
    chunk_size: int,
    token_budget: int,
    k_budget_ratio: float,
    warmup: int,
    repeats: int,
    compute_device: torch.device,
    k_bit_options: torch.Tensor,
    v_bit_options: torch.Tensor,
    obs_window: int = 32,
    legacy_probe: bool = False,
    prefill_chunk_size: Optional[int] = None,
    prebuilt_artifact: Optional[Dict[str, object]] = None,
    per_head: bool = False,
    pool_kernel_size: int = 5,
    streaming: bool = False,
    pool_padding: str = "reflect",
    eviction_mode: str = "joint",
    n_kept_multiplier: float = 5.0,
    v_score_type: str = "v_block",
    epsilon_K: Optional[Dict[int, float]] = None,
    epsilon_V: Optional[Dict[int, float]] = None,
    decode_blockwise_rdkv: bool = False,
    decode_block_size: int = 128,
    decode_block_budget_tokens: int = 1,
    decode_correctness_check: bool = False,
) -> Dict[str, object]:
    T = input_ids.shape[1]
    use_twophase = not legacy_probe and obs_window < T

    if streaming:
        artifact_builder = build_obkv_streaming_artifact
        artifact_kwargs = dict(
            model=model,
            input_ids=input_ids,
            compute_device=compute_device,
            token_budget=token_budget,
            k_budget_ratio=k_budget_ratio,
            k_bit_options=k_bit_options,
            v_bit_options=v_bit_options,
            obs_window=obs_window,
            pool_kernel_size=pool_kernel_size,
            pool_padding=pool_padding,
            eviction_mode=eviction_mode,
            n_kept_multiplier=n_kept_multiplier,
            v_score_type=v_score_type,
            epsilon_K=epsilon_K,
            epsilon_V=epsilon_V,
        )
    elif per_head:
        # Per-head now uses two-phase hook-based prefill (same path as global
        # run_obkv when prefill_chunk_size set, but keeps per-head scores).
        artifact_builder = build_obkv_perhead_artifact
        artifact_kwargs = dict(
            model=model,
            input_ids=input_ids,
            chunk_size=chunk_size,
            compute_device=compute_device,
            token_budget=token_budget,
            k_budget_ratio=k_budget_ratio,
            k_bit_options=k_bit_options,
            v_bit_options=v_bit_options,
            obs_window=obs_window,
            pool_kernel_size=pool_kernel_size,
            pool_padding="reflect",
            prefill_chunk_size=prefill_chunk_size,
        )
    elif use_twophase:
        artifact_builder = build_obkv_artifact_twophase
        artifact_kwargs = dict(
            model=model,
            input_ids=input_ids,
            obs_window=obs_window,
            compute_device=compute_device,
            token_budget=token_budget,
            k_budget_ratio=k_budget_ratio,
            k_bit_options=k_bit_options,
            v_bit_options=v_bit_options,
            prefill_chunk_size=prefill_chunk_size,
        )
    else:
        artifact_builder = build_obkv_artifact
        artifact_kwargs = dict(
            model=model,
            input_ids=input_ids,
            chunk_size=chunk_size,
            compute_device=compute_device,
            token_budget=token_budget,
            k_budget_ratio=k_budget_ratio,
            k_bit_options=k_bit_options,
            v_bit_options=v_bit_options,
        )

    def _get_artifact():
        if prebuilt_artifact is not None:
            return prebuilt_artifact
        return artifact_builder(**artifact_kwargs)

    # run_latency: CUDA Graph defaults to ON (capture once, replay across
    # warmup+repeats — capture cost is fully amortised). Set
    # OBKV_CUDA_GRAPH=0 to disable for ablation.
    # OBKV_DECOMPRESS_DECODE=1 forces the trizone artifact to be dequantised
    # back to fp16 past_kv before decode, replacing greedy_decode_fast with
    # plain HF model.forward (greedy_decode_pt). This is the trizone-vs-no-
    # trizone ablation path — incompatible with CUDA Graph (HF forward is
    # not graph-captured here). The TriZoneCache itself is still reusable
    # across repeats because each unpack creates a fresh fp16 past_kv.
    _decompress = os.environ.get("OBKV_DECOMPRESS_DECODE", "0") == "1"
    _use_cg = (not _decompress) and (os.environ.get("OBKV_CUDA_GRAPH", "1") == "1")
    decode_kwargs_base = dict(
        model=model,
        max_new_tokens=num_tokens - 1,
        eos_token_ids=[],
        primary_device=compute_device,
        use_cuda_graph=_use_cg,
    )

    # When CUDA Graph is on, pre-build artifact once so graphs persist
    # across warmup + measurement calls (same packed_cache object → same graphs).
    # Measure TTFT once from a fresh build, then reuse for decode.
    # Capture the prefill peak BEFORE the per-repeat reset so end-to-end peak
    # includes full-KV staging during prefill (the dominant footprint).
    _cg_ttft = None
    prefill_peak_gb = 0.0
    # Decompress mode also benefits from building the artifact once and
    # reusing it across warmup+repeats: each `_run_decompress_decode` call
    # unpacks a fresh fp16 past_kv from the same TriZoneCache, so reuse is
    # safe and avoids 4 redundant prefill passes per context.
    if (_use_cg or _decompress) and prebuilt_artifact is None:
        if os.environ.get("OBKV_PREFILL_WARMUP", "0") == "1":
            _warm_artifact = artifact_builder(**artifact_kwargs)
            del _warm_artifact
            gc.collect()
            torch.cuda.empty_cache()
            sync_cuda()
        _ttft_samples: List[float] = []
        _prefill_peak_samples: List[float] = []
        for _i in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            sync_cuda()
            _t0 = time.perf_counter()
            _candidate = artifact_builder(**artifact_kwargs)
            sync_cuda()
            _ttft_samples.append(time.perf_counter() - _t0)
            _prefill_peak_samples.append(torch.cuda.max_memory_allocated() / 1e9)
            if _i < repeats - 1:
                del _candidate
                gc.collect()
                torch.cuda.empty_cache()
                sync_cuda()
            else:
                prebuilt_artifact = _candidate
        _cg_ttft = statistics.median(_ttft_samples)
        prefill_peak_gb = max(_prefill_peak_samples)
        def _get_artifact():
            return prebuilt_artifact

    # Warmup
    for _ in range(warmup):
        artifact = _get_artifact()
        if _decompress:
            _run_decompress_decode(
                model=model,
                artifact=artifact,
                max_new_tokens=num_tokens - 1,
                eos_token_ids=[],
                primary_device=compute_device,
            )
        elif decode_blockwise_rdkv:
            from obkv_accel.blockwise_decode import (
                BlockwiseRDKVConfig,
                greedy_decode_blockwise,
            )
            greedy_decode_blockwise(
                model=model,
                dual_cache=artifact["packed_cache"],
                next_token_logits=artifact["next_token_logits"],
                max_new_tokens=num_tokens,
                eos_token_ids=[],
                primary_device=compute_device,
                decode_position_start=artifact["original_seq_len"],
                use_cuda_graph=_use_cg,
                config=BlockwiseRDKVConfig(
                    block_size=decode_block_size,
                    block_budget_tokens=decode_block_budget_tokens,
                    k_budget_ratio=k_budget_ratio,
                    obs_window=obs_window,
                    pool_kernel_size=pool_kernel_size,
                    pool_padding=pool_padding,
                    v_bit_options=v_bit_options,
                    k_bit_options=k_bit_options,
                    epsilon_v=epsilon_V,
                    epsilon_k=epsilon_K,
                    correctness_check=decode_correctness_check,
                ),
            )
        else:
            greedy_decode_fast(
                dual_cache=artifact["packed_cache"],
                next_token_logits=artifact["next_token_logits"],
                decode_position_start=artifact["original_seq_len"],
                **decode_kwargs_base,
            )

    # Measurement
    ttft_times: List[float] = []
    decode_times: List[float] = []
    peak_memories: List[float] = []
    generated_digests: List[str] = []
    generated_sequences: List[List[int]] = []
    t_eff = 0
    num_generated = num_tokens

    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()

        sync_cuda()
        t0 = time.perf_counter()
        artifact = _get_artifact()
        sync_cuda()
        t1 = time.perf_counter()
        ttft_times.append(_cg_ttft if _cg_ttft is not None else (t1 - t0))

        sync_cuda()
        t2 = time.perf_counter()
        if _decompress:
            generated_tail = _run_decompress_decode(
                model=model,
                artifact=artifact,
                max_new_tokens=num_tokens - 1,
                eos_token_ids=[],
                primary_device=compute_device,
            )
        elif decode_blockwise_rdkv:
            from obkv_accel.blockwise_decode import (
                BlockwiseRDKVConfig,
                greedy_decode_blockwise,
            )
            generated_tail, blockwise_report = greedy_decode_blockwise(
                model=model,
                dual_cache=artifact["packed_cache"],
                next_token_logits=artifact["next_token_logits"],
                max_new_tokens=num_tokens,
                eos_token_ids=[],
                primary_device=compute_device,
                decode_position_start=artifact["original_seq_len"],
                use_cuda_graph=_use_cg,
                config=BlockwiseRDKVConfig(
                    block_size=decode_block_size,
                    block_budget_tokens=decode_block_budget_tokens,
                    k_budget_ratio=k_budget_ratio,
                    obs_window=obs_window,
                    pool_kernel_size=pool_kernel_size,
                    pool_padding=pool_padding,
                    v_bit_options=v_bit_options,
                    k_bit_options=k_bit_options,
                    epsilon_v=epsilon_V,
                    epsilon_k=epsilon_K,
                    correctness_check=decode_correctness_check,
                ),
            )
        else:
            generated_tail = greedy_decode_fast(
                dual_cache=artifact["packed_cache"],
                next_token_logits=artifact["next_token_logits"],
                decode_position_start=artifact["original_seq_len"],
                **decode_kwargs_base,
            )
        sync_cuda()
        t3 = time.perf_counter()
        if decode_blockwise_rdkv:
            # Match the original ~18 ms protocol: time the whole decoder call.
            # Graph capture is already amortized by warmup/cache reuse, while
            # slot reset, replay-side updates, block packing and final flush
            # remain part of the measured blockwise overhead.
            decode_times.append(t3 - t2)
            num_generated = int(blockwise_report["generated_tokens"])
            generated_digests.append(
                hashlib.sha256(
                    json.dumps(
                        generated_tail, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
            )
            generated_sequences.append([int(token) for token in generated_tail])
        else:
            decode_times.append(t3 - t2)
            num_generated = 1 + len(generated_tail)
        t_eff = artifact["t_eff"]

        peak_memories.append(torch.cuda.max_memory_allocated() / 1e9)

    ttft_ms = median_ms(ttft_times)
    decode_total_ms = median_ms(decode_times)
    decode_tokens = max(
        num_generated if decode_blockwise_rdkv else num_generated - 1,
        1,
    )
    decode_peak_gb = max(peak_memories) if peak_memories else 0.0
    end_to_end_peak_gb = max(prefill_peak_gb, decode_peak_gb)
    result = {
        "status": "ok",
        "ttft_ms": round(ttft_ms, 1),
        "decode_total_ms": round(decode_total_ms, 1),
        "tpot_ms": round(decode_total_ms / decode_tokens, 2),
        "peak_memory_gb": round(end_to_end_peak_gb, 2),
        "decode_peak_gb": round(decode_peak_gb, 2),
        "prefill_peak_gb": round(prefill_peak_gb, 2),
        "t_eff": t_eff,
    }
    if decode_blockwise_rdkv:
        result.update(
            {
                "decode_blockwise_rdkv": True,
                "decode_cuda_graph": bool(
                    blockwise_report["decode_cuda_graph"]
                ),
                "decode_cuda_graph_active_slot_variants": list(
                    blockwise_report[
                        "decode_cuda_graph_active_slot_variants"
                    ]
                ),
                "decode_block_size": decode_block_size,
                "decode_block_budget_tokens": decode_block_budget_tokens,
                "decode_compression_ratio": (
                    decode_block_size / max(1, decode_block_budget_tokens)
                ),
                "num_compressed_decode_blocks": int(
                    blockwise_report["num_compressed_decode_blocks"]
                ),
                "decode_slot_bank_enabled": bool(
                    blockwise_report["decode_slot_bank_enabled"]
                ),
                "decode_slot_bank_slots": int(
                    blockwise_report["decode_slot_bank_slots"]
                ),
                "decode_slot_bank_physical_mb": round(
                    float(blockwise_report["decode_slot_bank_physical_bytes"])
                    / 1e6,
                    3,
                ),
                "generated_token_sha256": generated_digests[-1],
                "generated_token_sha256_per_repeat": generated_digests,
                "generated_token_deterministic_across_repeats": (
                    len(set(generated_digests)) == 1
                ),
                "generated_token_first_mismatch": next(
                    (
                        {
                            "index": token_idx,
                            "tokens": [
                                sequence[token_idx]
                                for sequence in generated_sequences
                            ],
                        }
                        for token_idx in range(
                            min(map(len, generated_sequences))
                        )
                        if len(
                            {
                                sequence[token_idx]
                                for sequence in generated_sequences
                            }
                        )
                        > 1
                    ),
                    None,
                ),
                "decode_excluding_final_flush_ms": round(
                    float(blockwise_report["decode_wall_time_ms_excluding_final_flush"]),
                    1,
                ),
                "final_flush_ms": round(
                    float(blockwise_report["final_flush_time_ms"]),
                    1,
                ),
                "decode_cache_peak_fp16_equivalent_tokens": round(
                    float(blockwise_report["decode_cache_peak_fp16_equivalent_tokens"]),
                    3,
                ),
                "decode_cache_final_fp16_equivalent_tokens": round(
                    float(blockwise_report["decode_cache_final_fp16_equivalent_tokens"]),
                    3,
                ),
                "decode_cache_peak_physical_mb": round(
                    float(blockwise_report["decode_cache_peak_physical_bytes"]) / 1e6,
                    3,
                ),
                "decode_cache_final_physical_mb": round(
                    float(blockwise_report["decode_cache_final_physical_bytes"]) / 1e6,
                    3,
                ),
                "prompt_compressed_kv_physical_mb": round(
                    float(blockwise_report["prompt_compressed_kv_physical_bytes"]) / 1e6,
                    3,
                ),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    patch_llama31_rope_compat()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_latency_v3.py")

    # Convert bit options to tensors
    k_bit_options = torch.tensor(sorted(set(args.k_options)), dtype=torch.float32)
    v_bit_options = torch.tensor(sorted(set(args.v_options)), dtype=torch.float32)

    # Load epsilon calibration (matches run_longbench.py streaming path)
    if args.epsilon_path:
        epsilon_K, epsilon_V = load_epsilon_kv_from_calibration(args.epsilon_path)
    else:
        epsilon_K, epsilon_V = DEFAULT_EPSILON_K, DEFAULT_EPSILON_V

    # Load model (single GPU)
    model, tokenizer, primary_device = load_model_and_tokenizer(
        args.model,
        "none",
        attn_implementation=args.attn_implementation,
    )

    bos_token_id = tokenizer.bos_token_id or 128000
    gpu_name = torch.cuda.get_device_name(0)

    # Output metadata
    payload = {
        "model": args.model,
        "gpu": gpu_name,
        "num_gpus": 1,
        "token_budget": args.token_budget,
        "k_budget_ratio": args.k_budget_ratio,
        "k_options": sorted(set(args.k_options)),
        "v_options": sorted(set(args.v_options)),
        "num_generate_tokens": args.num_tokens,
        "attn_implementation": args.attn_implementation,
        "prefill_chunk_size": args.prefill_chunk_size,
        "per_head": args.per_head,
        "streaming": args.streaming,
        "pool_kernel_size": args.pool_kernel_size,
        "pool_padding": args.pool_padding,
        "v_score_type": args.v_score_type,
        "epsilon_path": args.epsilon_path,
        "eviction_mode": args.eviction_mode,
        "n_kept_multiplier": args.n_kept_multiplier,
        "decode_blockwise_rdkv": args.decode_blockwise_rdkv,
        "decode_block_size": args.decode_block_size,
        "decode_block_budget_tokens": args.decode_block_budget_tokens,
        "decode_correctness_check": args.decode_correctness_check,
        "obkv_k_avg_mode": os.environ.get("OBKV_K_AVG_MODE", "perhead"),
        "obs_window": args.obs_window,
        "results": [],
    }

    print(f"Model: {args.model}")
    print(f"GPU: {gpu_name}")
    print(f"Budget: B={args.token_budget}, kr={args.k_budget_ratio}")
    print(f"K options: {sorted(set(args.k_options))}, V options: {sorted(set(args.v_options))}")
    print(f"Generate: {args.num_tokens} tokens, warmup={args.warmup}, repeats={args.repeats}")
    print(f"Context lengths: {args.context_lengths}")
    if args.prefill_chunk_size:
        print(f"Prefill chunk size: {args.prefill_chunk_size}")
    print()

    # --- Chunked prefill validation (32K) ---
    if args.prefill_chunk_size is not None:
        validation_ctx = min(32768, max(args.context_lengths))
        print(f"Validating chunked prefill equivalence at {validation_ctx} tokens...")
        val_ids = build_random_input_ids(validation_ctx, primary_device, bos_token_id)

        validate_chunked_prefill_equivalence(model, val_ids, args.prefill_chunk_size)
        clear_memory()

        validate_obkv_chunked_equivalence(
            model=model,
            input_ids=val_ids,
            prefill_chunk_size=args.prefill_chunk_size,
            obs_window=args.obs_window,
            compute_device=primary_device,
            token_budget=args.token_budget,
            k_budget_ratio=args.k_budget_ratio,
            k_bit_options=k_bit_options,
            v_bit_options=v_bit_options,
        )

        del val_ids
        clear_memory()
        print()

    for ctx_len in args.context_lengths:
        ctx_result: Dict[str, object] = {"context_length": ctx_len}

        # --- Full KV ---
        if args.skip_full_kv:
            full_kv = {"status": "skipped"}
        else:
            clear_memory()
            input_ids = build_random_input_ids(ctx_len, primary_device, bos_token_id)
            try:
                full_kv = measure_full_kv(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    num_tokens=args.num_tokens,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            except torch.cuda.OutOfMemoryError:
                clear_memory()
                full_kv = {"status": "oom"}
            except Exception as exc:
                full_kv = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        ctx_result["full_kv"] = full_kv

        status_str = full_kv["status"]
        if status_str == "ok":
            print(
                f"[{ctx_len} | full_kv] TTFT={full_kv['ttft_ms']}ms  "
                f"Decode={full_kv['decode_total_ms']}ms  "
                f"TPOT={full_kv['tpot_ms']}ms/tok  "
                f"Mem={full_kv['peak_memory_gb']}GB",
                flush=True,
            )
        elif status_str != "skipped":
            reason = full_kv.get("reason", "")
            print(f"[{ctx_len} | full_kv] {status_str.upper()} {reason}", flush=True)

        # --- OBKV ---
        clear_memory()
        input_ids = build_random_input_ids(ctx_len, primary_device, bos_token_id)

        # Load or build artifact (for fixed T_eff across ablation configs)
        prebuilt_artifact = None
        if args.load_artifact:
            artifact_path = Path(args.load_artifact) / f"ctx{ctx_len}.pt"
            if artifact_path.exists():
                prebuilt_artifact = torch.load(artifact_path, map_location=primary_device,
                                               weights_only=False)
                print(f"  [loaded artifact] {artifact_path} (T_eff={prebuilt_artifact['t_eff']})",
                      flush=True)
        elif args.save_artifact:
            # Build artifact once, save for reuse, then use as prebuilt
            T = input_ids.shape[1]
            use_twophase = not args.legacy_probe and args.obs_window < T
            if args.streaming:
                artifact = build_obkv_streaming_artifact(
                    model=model, input_ids=input_ids,
                    compute_device=primary_device, token_budget=args.token_budget,
                    k_budget_ratio=args.k_budget_ratio,
                    k_bit_options=k_bit_options, v_bit_options=v_bit_options,
                    obs_window=args.obs_window, pool_kernel_size=args.pool_kernel_size,
                    pool_padding=args.pool_padding,
                    eviction_mode=args.eviction_mode,
                    n_kept_multiplier=args.n_kept_multiplier,
                    v_score_type=args.v_score_type,
                    epsilon_K=epsilon_K, epsilon_V=epsilon_V,
                )
            elif args.per_head:
                artifact = build_obkv_perhead_artifact(
                    model=model, input_ids=input_ids, chunk_size=args.chunk_size,
                    compute_device=primary_device, token_budget=args.token_budget,
                    k_budget_ratio=args.k_budget_ratio,
                    k_bit_options=k_bit_options, v_bit_options=v_bit_options,
                    obs_window=args.obs_window, pool_kernel_size=args.pool_kernel_size,
                    eviction_mode=args.eviction_mode,
                    n_kept_multiplier=args.n_kept_multiplier,
                )
            elif use_twophase:
                artifact = build_obkv_artifact_twophase(
                    model=model, input_ids=input_ids, obs_window=args.obs_window,
                    compute_device=primary_device, token_budget=args.token_budget,
                    k_budget_ratio=args.k_budget_ratio,
                    k_bit_options=k_bit_options, v_bit_options=v_bit_options,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            else:
                artifact = build_obkv_artifact(
                    model=model, input_ids=input_ids, chunk_size=args.chunk_size,
                    compute_device=primary_device, token_budget=args.token_budget,
                    k_budget_ratio=args.k_budget_ratio,
                    k_bit_options=k_bit_options, v_bit_options=v_bit_options,
                )
            save_dir = Path(args.save_artifact)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"ctx{ctx_len}.pt"
            torch.save(artifact, save_path)
            print(f"  [saved artifact] {save_path} (T_eff={artifact['t_eff']})", flush=True)
            prebuilt_artifact = artifact

        try:
            obkv = measure_obkv(
                model=model,
                input_ids=input_ids,
                num_tokens=args.num_tokens,
                chunk_size=args.chunk_size,
                token_budget=args.token_budget,
                k_budget_ratio=args.k_budget_ratio,
                warmup=args.warmup,
                repeats=args.repeats,
                compute_device=primary_device,
                k_bit_options=k_bit_options,
                v_bit_options=v_bit_options,
                obs_window=args.obs_window,
                legacy_probe=args.legacy_probe,
                prefill_chunk_size=args.prefill_chunk_size,
                prebuilt_artifact=prebuilt_artifact,
                per_head=args.per_head,
                pool_kernel_size=args.pool_kernel_size,
                streaming=args.streaming,
                pool_padding=args.pool_padding,
                eviction_mode=args.eviction_mode,
                n_kept_multiplier=args.n_kept_multiplier,
                v_score_type=args.v_score_type,
                epsilon_K=epsilon_K,
                epsilon_V=epsilon_V,
                decode_blockwise_rdkv=args.decode_blockwise_rdkv,
                decode_block_size=args.decode_block_size,
                decode_block_budget_tokens=args.decode_block_budget_tokens,
                decode_correctness_check=args.decode_correctness_check,
            )
        except torch.cuda.OutOfMemoryError:
            clear_memory()
            obkv = {"status": "oom"}
        except Exception as exc:
            obkv = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        ctx_result["obkv"] = obkv

        status_str = obkv["status"]
        if status_str == "ok":
            print(
                f"[{ctx_len} | obkv] TTFT={obkv['ttft_ms']}ms  "
                f"Decode={obkv['decode_total_ms']}ms  "
                f"TPOT={obkv['tpot_ms']}ms/tok  "
                f"Mem={obkv['peak_memory_gb']}GB"
                f" (prefill={obkv.get('prefill_peak_gb', 0)}, decode={obkv.get('decode_peak_gb', 0)})  "
                f"T_eff={obkv['t_eff']}",
                flush=True,
            )
        else:
            reason = obkv.get("reason", "")
            print(f"[{ctx_len} | obkv] {status_str.upper()} {reason}", flush=True)

        payload["results"].append(ctx_result)
        print()

    # --- TPOT Monotonicity Check ---
    full_kv_tpots = [
        (r["context_length"], r["full_kv"]["tpot_ms"])
        for r in payload["results"]
        if r["full_kv"].get("status") == "ok"
    ]
    for i in range(len(full_kv_tpots) - 1):
        ctx_a, tpot_a = full_kv_tpots[i]
        ctx_b, tpot_b = full_kv_tpots[i + 1]
        if tpot_a > tpot_b:
            print(
                f"WARNING: Full KV TPOT monotonicity violated: "
                f"{ctx_a}={tpot_a:.1f}ms > {ctx_b}={tpot_b:.1f}ms"
            )

    # --- Save output ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[saved] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
