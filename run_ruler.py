#!/usr/bin/env python3
"""
RULER benchmark evaluation for RDKV.

Supports 4 methods:
  - fullkv:      Full KV cache (upper bound)
  - snapkv:      SnapKV pruning baseline
  - ada_snapkv:  Adaptive SnapKV baseline
  - obkv:        OBKV with knapsack_split (our method)

Usage:
  python pred_ruler_eval.py \
      --method obkv \
      --seq-length 4096 \
      --ruler-data-dir $SCR/ruler_data/llama31_8b_instruct \
      --output-dir results/ruler/obkv/4096/shard_0 \
      --shard 0 --num-shards 4 \
      --token-budget 1024 --k-budget-ratio 0.5 \
      --max-new-tokens 128 --seed 42
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# ── Imports from obkv_fast library ───────────────────────────────────────
from obkv_fast import (
    DEFAULT_MODEL_PATH,
    detach_past_key_values,
    greedy_decode_pt as greedy_decode,
    load_model_and_tokenizer,
    prefill_last_token,
    prepare_past_key_values_for_model,
    run_obkv,
    set_determinism,
)
from knapsack_solver import (
    DEFAULT_EPSILON_K,
    DEFAULT_EPSILON_V,
    load_epsilon_kv_from_calibration,
)
from ruler_dataset import (
    RULER_SHARD_SPLIT,
    RULER_TASKS,
    evaluate_sample,
    evaluate_task,
    load_ruler_shard,
    load_ruler_tasks,
)

# Unified max_new_tokens = 64 for all tasks (matches ReST-KV protocol)
RULER_UNIFIED_MAX_NEW_TOKENS = 64


# ── Argument parsing ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RULER benchmark evaluation for RDKV.")
    p.add_argument("--method", required=True, choices=["fullkv", "snapkv", "ada_snapkv", "pyramidkv", "obkv", "hqekv"])
    p.add_argument("--seq-length", type=int, required=True, help="RULER sequence length (e.g. 4096)")
    p.add_argument("--ruler-data-dir", required=True, help="Root dir of pre-generated RULER data")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=RULER_UNIFIED_MAX_NEW_TOKENS,
                    help=f"Max generation tokens (default: {RULER_UNIFIED_MAX_NEW_TOKENS}, matches ReST-KV)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=None,
                    help="Limit samples per task (for smoke testing)")
    p.add_argument("--task-filter", default=None,
                    help="Comma-separated task list (overrides shard split)")
    p.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa"])
    p.add_argument("--device-map", default="auto",
                    help="Device map for model loading (default: auto, use balanced for multi-GPU)")
    # SnapKV / Ada-SnapKV params
    p.add_argument("--token-budget", type=int, default=1024,
                    help="Token budget B (default: 1024)")
    p.add_argument("--window-size", type=int, default=32,
                    help="SnapKV observation window size")
    p.add_argument("--kernel-size", type=int, default=7,
                    help="SnapKV pooling kernel size")
    p.add_argument("--alpha-safeguard", type=float, default=0.2,
                    help="Ada-SnapKV safeguard coefficient")
    # OBKV params
    p.add_argument("--k-budget-ratio", type=float, default=0.5,
                    help="Fraction of budget for K-side in knapsack_split")
    p.add_argument("--epsilon-path", default=None,
                    help="Calibrated epsilon JSON for OBKV")
    p.add_argument("--chunk-size", type=int, default=128,
                    help="Chunk size for value_residual_probe prefill (single-pass path)")
    p.add_argument("--prefill-chunk-size", type=int, default=None,
                    help="If set (e.g. 8192), use two-phase chunked prefill — fits "
                         "128K context on a single A100-64GB GPU.")
    p.add_argument("--obs-window", type=int, default=32,
                    help="Observation window size for OBKV scoring "
                         "(use last N tokens as probe positions, 0=disabled)")
    p.add_argument("--score-pooling", type=int, default=5,
                    help="Avg pooling kernel size for token scores "
                         "(0=disabled, 5=SnapKV-style)")
    p.add_argument("--fp16-topk", type=int, default=0,
                    help="Number of top-importance V tokens to force FP16 before knapsack. "
                         "0 = disabled.")
    p.add_argument("--v-bit-options", type=str, default=None,
                    help="Comma-separated V-side bit options (e.g. '0,4,8,16' to drop 2-bit). "
                         "Default: 0,2,4,8,16")
    p.add_argument("--v-score-type", choices=["attn_linear"], default="attn_linear",
                    help="V-side per-token score (sum_tau a over the recent window).")
    p.add_argument("--k-score-type", choices=["think"], default="think",
                    help="K-side per-channel score "
                         "(ThinK: mean_tau(q_c^2) * mean_t(k_c^2)).")
    p.add_argument("--streaming", action="store_true", default=True,
                    help="Layer-streaming per-head prefill "
                         "(run_obkv). Always on — kept "
                         "for backwards-compat with the legacy CLI.")
    p.add_argument("--pool-type", choices=["avg", "max"], default="avg",
                    help="Per-head score pooling op: avg (default) or max.")
    p.add_argument("--eviction-mode", choices=["joint"], default="joint",
                    help="Per-head joint knapsack over {0,2,4,8,16}.")
    # HqeKV params
    p.add_argument("--hqekv-repo", default=None,
                    help="Path to HqeKV repo (defaults to ../HqeKV from this script).")
    p.add_argument("--hqekv-ratios-path", default=None,
                    help="Override path to ratios.json (default: <hqekv-repo>/config/ratios.json).")
    p.add_argument("--hqekv-model-key", default=None,
                    help="Top-level key in ratios.json to look up (default: derived from --model-path basename).")
    p.add_argument("--hqekv-strategy", default="high_uniform_group_low_normal_group",
                    help="HqeKV config.quant_strategy (matches pred_long_bench_hq.py default).")
    p.add_argument("--hqekv-times-range", action=argparse.BooleanOptionalAction, default=True,
                    help="HqeKV config.times_range flag.")
    p.add_argument("--hqekv-avg-bit", type=float, default=None,
                    help="Override avg_bit; default = 16 * token_budget / seq_length, clipped to [0, 4].")
    p.add_argument("--hqekv-gpu-id", type=int, default=0,
                    help="Single GPU ordinal (only used when --hqekv-device-map is unset).")
    p.add_argument("--hqekv-device-map", default=None,
                    help="HF device_map: 'auto' / 'balanced' / 'sequential' for multi-GPU "
                         "sharding, or 'cuda:N' / int for single GPU. If unset, falls back "
                         "to cuda:<hqekv-gpu-id>.")
    return p.parse_args()


# ── Tokenization ─────────────────────────────────────────────────────────

def tokenize_ruler_input(
    input_text: str,
    tokenizer,
    device: torch.device,
    answer_prefix: str = "",
) -> Tuple[torch.Tensor, int]:
    """Tokenize a RULER input string + answer_prefix.

    The RULER data was generated with model_template_type=meta-llama3, so the
    input text already includes the Llama-3 chat tokens as literal strings
    (e.g. <|begin_of_text|>, <|start_header_id|>, etc.). We tokenize directly
    without applying chat template again.

    The answer_prefix is appended to the input and included in the prefill
    (not counted as generated tokens). This guides the model to output in
    the expected format (matching RULER's official evaluation protocol).

    Returns:
        (input_ids, prompt_len) where prompt_len = len(input_text tokens)
        and input_ids includes both input_text + answer_prefix.
    """
    full_text = input_text + answer_prefix
    input_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
    # prompt_len includes answer_prefix (all of it is prefilled)
    prompt_len = input_ids.shape[1]
    return input_ids.to(device), prompt_len


# ── Method: Full KV ──────────────────────────────────────────────────────

def _print_gpu_memory(label: str):
    """Print per-GPU memory usage for all visible devices."""
    n = torch.cuda.device_count()
    print(f"  [GPU MEM] {label} — {n} device(s)")
    for i in range(n):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB")


def run_fullkv(model, input_ids, max_new_tokens, eos_token_ids, device):
    _print_gpu_memory("before prefill")
    with torch.inference_mode():
        next_token_logits, past_key_values = prefill_last_token(model, input_ids)
    _print_gpu_memory("after prefill")
    generated_ids = greedy_decode(
        model=model,
        past_key_values=past_key_values,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        primary_device=device,
    )
    _print_gpu_memory("after decode")
    return generated_ids


# ── Method: SnapKV / Ada-SnapKV ──────────────────────────────────────────

def run_kvpress(model, input_ids, args, prompt_len, max_new_tokens, eos_token_ids, device):
    """Run inference with kvpress-based compression (SnapKV, Ada-SnapKV, PyramidKV)."""
    from pred_kvpress_baseline import compression_ratio_for_budget

    compression_ratio = compression_ratio_for_budget(prompt_len, args.token_budget)

    # Build press object inline
    from kvpress import AdaKVPress, PyramidKVPress, SnapKVPress

    if args.method == "pyramidkv":
        press = PyramidKVPress(
            compression_ratio=compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    elif args.method == "ada_snapkv":
        press = AdaKVPress(
            press=SnapKVPress(
                compression_ratio=compression_ratio,
                window_size=args.window_size,
                kernel_size=args.kernel_size,
            ),
            alpha_safeguard=args.alpha_safeguard,
        )
    else:
        press = SnapKVPress(
            compression_ratio=compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )

    with press(model):
        with torch.inference_mode():
            next_token_logits, past_key_values = prefill_last_token(model, input_ids)
        # Determine if eviction happened
        cache_len = past_key_values[0][0].shape[2]
        generated_ids = greedy_decode(
            model=model,
            past_key_values=past_key_values,
            next_token_logits=next_token_logits,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            primary_device=device,
            decode_position_start=prompt_len if cache_len < prompt_len else None,
        )
    return generated_ids


# ── Method: OBKV (thin wrapper around obkv_fast.run_obkv) ───────────────

def _parse_bit_options(spec):
    if not spec:
        return None
    bits = sorted({int(b.strip()) for b in spec.split(",") if b.strip()})
    if 0 not in bits:
        bits.insert(0, 0)
    return torch.tensor(bits, dtype=torch.float32)


def run_obkv_method(model, input_ids, args, epsilon_K, epsilon_V, max_new_tokens,
                    eos_token_ids, device, task_name, sample_id):
    v_bit_options = _parse_bit_options(args.v_bit_options)
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
        pool_kernel_size=args.score_pooling if args.score_pooling > 0 else 1,
        pool_padding="reflect",
        v_bit_options=v_bit_options,
        k_bit_options=None,
        fp16_topk=args.fp16_topk,
        device=device,
        prefill_chunk_size=getattr(args, "prefill_chunk_size", None),
    )
    kwargs["eviction_mode"] = getattr(args, "eviction_mode", "joint")
    kwargs["v_score_type"] = getattr(args, "v_score_type", "attn_linear")
    kwargs["k_score_type"] = getattr(args, "k_score_type", "think")
    return entry(**kwargs)



def write_ruler_results(
    output_dir: Path,
    results_by_task: Dict[str, List[Dict]],
    summary: Dict,
):
    """Write per-task JSONL files and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, results in results_by_task.items():
        jsonl_path = output_dir / f"{task_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Output] Results written to {output_dir}")


def append_ruler_progress(output_dir: Path, row: Dict) -> None:
    """Append one JSONL progress row after a sample finishes.

    This is intentionally independent from the final result writer so we keep
    the existing end-of-run artifacts unchanged while exposing per-sample
    progress for long-running jobs.
    """
    progress_path = output_dir / "sample_progress.jsonl"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ── Method: HqeKV ────────────────────────────────────────────────────────

def _hqekv_avg_bit_key(v: float) -> str:
    """Match ratios.json formatting: '2' not '2.0', '3.2' not '3.20'."""
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def _hqekv_choose_bits(target: float, ratios_for_model: Dict[str, Dict[str, float]]
                       ) -> Tuple[float, Dict[str, float]]:
    """Pick (chosen_avg_bit, bits) from ratios for nearest-calibrated lookup.

    HqeKV bit set is {0, 1, 2, 4}; structural max avg is 4 (all blocks 4-bit).
    For target >= 4 we use trivial bits (no compression); otherwise nearest
    calibrated point from ratios.json (after clipping to [0, 4]).
    """
    target = max(0.0, min(4.0, target))
    if target >= 4.0:
        return 4.0, {"4": 1.0, "2": 0.0, "1": 0.0, "0": 0.0}
    keys = sorted(float(k) for k in ratios_for_model.keys())
    chosen = min(keys, key=lambda k: abs(k - target))
    return chosen, ratios_for_model[_hqekv_avg_bit_key(chosen)]


def _hqekv_derive_model_key(model_path: str) -> str:
    """Pull the model name from a path. Handles HF cache layout
    (`.../models--<vendor>--<name>/snapshots/<hash>/`) and direct dirs."""
    p = Path(model_path)
    for parent in [p, *p.parents]:
        name = parent.name
        if name.startswith("models--"):
            parts = name.split("--")
            if len(parts) >= 3:
                return parts[-1]
    return p.name


def _hqekv_install_loss_kwargs_shim():
    """transformers >=4.55 removed `LossKwargs` from `transformers.utils`.
    HqeKV model files (pinned to 4.52.4) still import it; inject an empty
    TypedDict fallback so HqeKV imports resolve under our overlay's tf 4.57."""
    import transformers.utils as _tu
    if not hasattr(_tu, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False): ...
        _tu.LossKwargs = LossKwargs


def load_hqekv_model_and_tokenizer(args, repo_root: Path):
    """Load HqeKV's LlamaForCausalLM_hqe with bit ratios chosen from --token-budget
    and --seq-length.  Returns (model, tokenizer, primary_device, info_dict)."""
    hqekv_repo = Path(args.hqekv_repo) if args.hqekv_repo else (repo_root / "HqeKV")
    if not hqekv_repo.exists():
        raise FileNotFoundError(f"HqeKV repo not found: {hqekv_repo}")
    ratios_path = Path(args.hqekv_ratios_path) if args.hqekv_ratios_path else (
        hqekv_repo / "config" / "ratios.json")
    if not ratios_path.exists():
        raise FileNotFoundError(f"HqeKV ratios.json not found: {ratios_path}")
    with open(ratios_path) as f:
        ratios = json.load(f)

    model_key = args.hqekv_model_key or _hqekv_derive_model_key(args.model_path)
    if model_key not in ratios:
        raise KeyError(
            f"Model key '{model_key}' not in ratios.json keys: {list(ratios.keys())}. "
            f"Pass --hqekv-model-key to override.")
    ratios_for_model = ratios[model_key]

    if args.hqekv_avg_bit is not None:
        target_avg_bit = float(args.hqekv_avg_bit)
    else:
        target_avg_bit = 16.0 * args.token_budget / args.seq_length

    chosen_avg_bit, bits = _hqekv_choose_bits(target_avg_bit, ratios_for_model)
    print(f"[HqeKV] target_avg_bit={target_avg_bit:.4f} (16 * {args.token_budget} / "
          f"{args.seq_length}) → calibrated={chosen_avg_bit} bits={bits}")

    # Make HqeKV importable
    sys.path.insert(0, str(hqekv_repo))
    _hqekv_install_loss_kwargs_shim()

    import torch as _torch
    from transformers import AutoTokenizer, LlamaConfig
    name_lower = args.model_path.lower()

    if "llama" in name_lower:
        config = LlamaConfig.from_pretrained(args.model_path)
        from model.llama_hqe import LlamaForCausalLM_hqe as ModelCls  # type: ignore
    elif "qwen" in name_lower:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # type: ignore
        config = Qwen3Config.from_pretrained(args.model_path)
        from model.qwen3_hqe import Qwen3ForCausalLM as ModelCls  # type: ignore
    else:
        raise NotImplementedError(
            f"HqeKV currently supports llama/qwen model paths only; got {args.model_path}")

    config.quant_strategy = args.hqekv_strategy
    config.bit_4 = float(bits.get("4", 0.0))
    config.bit_2 = float(bits.get("2", 0.0))
    config.bit_1 = float(bits.get("1", 0.0))
    config.bit_0 = float(bits.get("0", 0.0))
    config.times_range = bool(args.hqekv_times_range)

    dm_arg = args.hqekv_device_map
    if dm_arg in (None, ""):
        device_map_for_hf = _torch.device(f"cuda:{args.hqekv_gpu_id}")
        primary_device = device_map_for_hf
    elif dm_arg in ("auto", "balanced", "sequential", "balanced_low_0"):
        device_map_for_hf = dm_arg  # let HF shard across visible GPUs
        primary_device = _torch.device("cuda:0")  # input/embed land here under "auto"
    elif dm_arg.startswith("cuda:") or dm_arg.isdigit():
        device_map_for_hf = _torch.device(dm_arg if dm_arg.startswith("cuda:") else f"cuda:{dm_arg}")
        primary_device = device_map_for_hf
    else:
        raise ValueError(f"Unrecognised --hqekv-device-map: {dm_arg!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = ModelCls.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        torch_dtype=_torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map_for_hf,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    # HqeKV's _update_causal_mask still materialises a full (T, T) float16 mask
    # even when flash_attention_2 is active (which handles causality itself).
    # At T=131072 this is 32 GiB and OOMs on A100-64GB. Skip it for prefill
    # (q_len>1); keep the original path for decode (q_len=1) where qkv_matmul_hqe_2
    # consumes the mask.
    if args.attn_implementation == "flash_attention_2":
        _inner = getattr(model, "model", model)
        _orig_update = _inner._update_causal_mask
        def _patched_update_causal_mask(attention_mask, input_tensor, cache_position,
                                        past_key_values, output_attentions):
            if input_tensor.shape[1] > 1:
                return None
            return _orig_update(attention_mask, input_tensor, cache_position,
                                past_key_values, output_attentions)
        _inner._update_causal_mask = _patched_update_causal_mask
    info = {
        "target_avg_bit": target_avg_bit,
        "chosen_avg_bit": chosen_avg_bit,
        "bits": bits,
        "ratios_path": str(ratios_path),
        "model_key": model_key,
        "strategy": args.hqekv_strategy,
        "times_range": bool(args.hqekv_times_range),
    }
    return model, tokenizer, primary_device, info


def run_hqekv(model, tokenizer, input_ids, max_new_tokens, eos_token_ids):
    """Generate via HqeKV's custom `model.generate` (handles its tuple-cache internally).
    Returns the *generated* ids (input is sliced off)."""
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
        )[0]
    return output[input_ids.shape[1]:].tolist()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_determinism(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== RULER Benchmark Evaluation ===")
    print(f"  Method:       {args.method}")
    print(f"  Seq length:   {args.seq_length}")
    print(f"  Shard:        {args.shard}/{args.num_shards}")
    print(f"  Token budget: {args.token_budget}")
    if args.method == "obkv":
        print(f"  K ratio:      {args.k_budget_ratio}")
        print(f"  Obs window:   {args.obs_window}")
        print(f"  Score pool:   {args.score_pooling}")
        print(f"  FP16 top-k:   {args.fp16_topk}")
    print(f"  Max new tok:  {args.max_new_tokens}")
    print(f"  Output:       {output_dir}")
    print()

    # ── Load model ───────────────────────────────────────────────────
    hqekv_info = None
    if args.method == "hqekv":
        repo_root = Path(__file__).resolve().parent.parent
        model, tokenizer, primary_device, hqekv_info = load_hqekv_model_and_tokenizer(
            args, repo_root)
    else:
        # OBKV uses value_residual_probe; baselines use "none"
        model_mode = "value_residual_probe" if args.method == "obkv" else "none"
        model, tokenizer, primary_device = load_model_and_tokenizer(
            args.model_path,
            model_mode,
            args.attn_implementation,
            device_map_arg=args.device_map,
        )

    eos_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for special in ["<|eot_id|>", "<|end_of_text|>"]:
            tid = tokenizer.convert_tokens_to_ids(special)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos_token_ids.append(tid)
    eos_token_ids = list(set(eos_token_ids))

    # ── Load OBKV epsilon calibration ────────────────────────────────
    epsilon_K, epsilon_V = None, None
    if args.method == "obkv" and args.epsilon_path:
        epsilon_K, epsilon_V = load_epsilon_kv_from_calibration(args.epsilon_path)
        print(f"[OBKV] Loaded epsilon from {args.epsilon_path}")

    # ── Load RULER data ──────────────────────────────────────────────
    if args.task_filter:
        task_list = [t.strip() for t in args.task_filter.split(",")]
        work_items = load_ruler_tasks(
            args.ruler_data_dir, args.seq_length, task_filter=task_list,
            max_examples=args.max_examples,
        )
    else:
        work_items = load_ruler_shard(
            args.ruler_data_dir, args.seq_length, args.shard,
            max_examples=args.max_examples,
        )

    print(f"[Data] Loaded {len(work_items)} samples "
          f"({len(set(t for t, _ in work_items))} tasks)")

    # ── Inference loop ───────────────────────────────────────────────
    results_by_task: Dict[str, List[Dict]] = defaultdict(list)
    prompt_lens = []
    gen_lens = []
    oom_count = 0
    t_start = time.time()

    for idx, (task_name, sample) in enumerate(work_items, start=1):
        answer_prefix = sample.get("answer_prefix", "")
        input_ids, prompt_len = tokenize_ruler_input(
            sample["input"], tokenizer, primary_device,
            answer_prefix=answer_prefix,
        )
        prompt_lens.append(prompt_len)

        # Unified max_new_tokens (matches ReST-KV protocol)
        task_max_tokens = args.max_new_tokens

        print(f"[{idx}/{len(work_items)}] task={task_name} "
              f"sample={sample.get('index', '?')} prompt_len={prompt_len} "
              f"max_gen={task_max_tokens}"
              f"{' prefix=' + str(len(answer_prefix)) + 'ch' if answer_prefix else ''}")

        generated_ids = []
        error = None

        try:
            if args.method == "fullkv":
                generated_ids = run_fullkv(
                    model, input_ids, task_max_tokens, eos_token_ids, primary_device,
                )
            elif args.method in ("snapkv", "ada_snapkv", "pyramidkv"):
                generated_ids = run_kvpress(
                    model, input_ids, args, prompt_len,
                    task_max_tokens, eos_token_ids, primary_device,
                )
            elif args.method == "obkv":
                generated_ids = run_obkv_method(
                    model, input_ids, args, epsilon_K, epsilon_V,
                    task_max_tokens, eos_token_ids, primary_device,
                    task_name=task_name,
                    sample_id=sample.get("index", idx - 1),
                )
            elif args.method == "hqekv":
                generated_ids = run_hqekv(
                    model, tokenizer, input_ids,
                    task_max_tokens, eos_token_ids,
                )
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            error = "OOM"
            print(f"  [OOM] task={task_name} prompt_len={prompt_len}")
            print(f"  [OOM TRACEBACK]")
            traceback.print_exc()
            _print_gpu_memory("at OOM")
            for i in range(torch.cuda.device_count()):
                print(f"  [OOM] GPU {i} memory_summary:")
                print(torch.cuda.memory_summary(i, abbreviated=True))
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # allocator may be dirty from failed graph capture

        # Decode prediction
        prediction = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip() if generated_ids else ""

        gen_lens.append(len(generated_ids))

        # Score
        references = sample.get("outputs", [])
        score = evaluate_sample(task_name, prediction, references)

        result_row = {
            "task": task_name,
            "index": sample.get("index", idx - 1),
            "prediction": prediction,
            "outputs": references,
            "seq_length": args.seq_length,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated_ids),
            "score": score,
        }
        if error:
            result_row["error"] = error
        results_by_task[task_name].append(result_row)
        append_ruler_progress(
            output_dir,
            {
                "finished_at": time.time(),
                "completed": idx,
                "total": len(work_items),
                "task": task_name,
                "index": sample.get("index", idx - 1),
                "prompt_tokens": prompt_len,
                "generated_tokens": len(generated_ids),
                "score": score,
                "error": error,
            },
        )

        if idx % 10 == 0 or idx == len(work_items):
            elapsed = time.time() - t_start
            print(f"  Progress: {idx}/{len(work_items)} "
                  f"({elapsed:.0f}s, {elapsed/idx:.1f}s/sample)")

    # ── Per-task accuracy ────────────────────────────────────────────
    print("\n=== Per-Task Accuracy ===")
    task_accuracies = {}
    for task_name in sorted(results_by_task.keys()):
        acc = evaluate_task(task_name, results_by_task[task_name])
        task_accuracies[task_name] = acc
        n = len(results_by_task[task_name])
        print(f"  {task_name:25s}  {acc*100:6.2f}%  (n={n})")

    if task_accuracies:
        overall = sum(task_accuracies.values()) / len(task_accuracies)
        print(f"  {'AVERAGE':25s}  {overall*100:6.2f}%")

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "method": args.method,
        "model_path": args.model_path,
        "seq_length": args.seq_length,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "token_budget": args.token_budget,
        "fp16_topk": args.fp16_topk,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_samples": len(work_items),
        "oom_count": oom_count,
        "task_accuracies": {k: round(v, 4) for k, v in task_accuracies.items()},
        "overall_accuracy": round(overall, 4) if task_accuracies else None,
        "prompt_len_stats": {
            "min": min(prompt_lens) if prompt_lens else 0,
            "max": max(prompt_lens) if prompt_lens else 0,
            "avg": sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0,
        },
        "gen_len_stats": {
            "min": min(gen_lens) if gen_lens else 0,
            "max": max(gen_lens) if gen_lens else 0,
            "avg": sum(gen_lens) / len(gen_lens) if gen_lens else 0,
        },
    }
    if args.method == "obkv":
        summary["k_budget_ratio"] = args.k_budget_ratio
        summary["epsilon_path"] = args.epsilon_path
        summary["score_pooling"] = args.score_pooling
        summary["v_score_type"] = args.v_score_type
    if args.method == "hqekv" and hqekv_info is not None:
        summary["hqekv"] = hqekv_info

    write_ruler_results(output_dir, results_by_task, summary)
    print(f"\nTotal time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
