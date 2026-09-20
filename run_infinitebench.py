#!/usr/bin/env python3
"""
InfiniteBench evaluation for RDKV.

Supports 3 methods:
  - fullkv:      Full KV cache (upper bound)
  - ada_snapkv:  Adaptive SnapKV baseline
  - obkv:        OBKV with knapsack_split (our method)

Usage:
  python pred_infinitebench_eval.py \
      --method obkv \
      --task passkey,number_string \
      --output-dir results/infinitebench/obkv \
      --token-budget 1024 --k-budget-ratio 0.5 \
      --epsilon-path results/epsilon_calibration_16384.json \
      --seed 42
"""

import argparse
import gc
import json
import os
import re
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


# ── InfiniteBench constants (from ReST-KV) ──────────────────────────────

INFINITEBENCH_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "code_debug",
]

# Per-task max generation tokens (from ReST-KV run_infinite_bench.py)
DATASET2MAXLEN = {
    "passkey": 15,
    "number_string": 20,
    "kv_retrieval": 80,
    "longbook_sum_eng": 1200,
    "longbook_choice_eng": 40,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 3,
    "code_debug": 5,
}

# Task name → JSONL filename mapping
DATA_NAME_TO_PATH = {
    "passkey": "passkey.jsonl",
    "number_string": "number_string.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
    "longbook_sum_eng": "longbook_sum_eng.jsonl",
    "longbook_choice_eng": "longbook_choice_eng.jsonl",
    "longbook_qa_eng": "longbook_qa_eng.jsonl",
    "longbook_qa_chn": "longbook_qa_chn.jsonl",
    "longdialogue_qa_eng": "longdialogue_qa_eng.jsonl",
    "math_find": "math_find.jsonl",
    "code_debug": "code_debug.jsonl",
}

# Prompt templates (from ReST-KV run_infinite_bench.py lines 67-80)
DATA2PROMPT = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": 'Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely',  # noqa
}


# ── Prompt & data functions (from ReST-KV) ──────────────────────────────

def create_prompt(eg: dict, data_name: str) -> str:
    """Create prompt for a given example (from ReST-KV run_infinite_bench.py)."""
    template = DATA2PROMPT[data_name]

    if data_name in ["code_debug"]:
        code = eg["context"]
        return template.format(
            context=code,
            OPTION_A=eg["options"][0],
            OPTION_B=eg["options"][1],
            OPTION_C=eg["options"][2],
            OPTION_D=eg["options"][3],
        )
    elif data_name == "longdialogue_qa_eng":
        script = eg["context"]
        return template.format(context=script)
    elif data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["input"],
                context=book,
            )
        elif data_name == "longbook_sum_eng":
            return template.format(context=book)
        elif data_name == "longbook_qa_chn":
            return template.format(
                question=eg["input"],
                context=book,
            )
        else:
            raise ValueError
    elif data_name == "math_find":
        prompt = eg["input"]
        context = eg["context"]
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find the target number in {prompt}"
        target_number = find_result[0].lower()[:-3]
        prefix = f"What is {target_number} in the following list?"
        return template.format(
            prefix=prefix,
            context=context,
            input=prompt,
        )

    # Default: passkey, number_string, kv_retrieval
    if "content" in eg:
        content = eg["content"]
        del eg["content"]
        eg["context"] = content

    format_dict = {
        "context": eg["context"],
        "input": eg["input"],
    }
    return template.format(**format_dict)


def get_answer(eg: dict, data_name: str):
    """Extract ground truth answer (from ReST-KV run_infinite_bench.py)."""
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg["options"].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg["options"].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ["A", "B", "C", "D"]:
                ret = eg["answer"]
            else:
                raise ValueError
        else:
            raise ValueError
        return ret
    return eg["answer"]


# ── Data loading ─────────────────────────────────────────────────────────

def load_infinitebench_task(data_dir: str, task_name: str,
                            max_examples: Optional[int] = None) -> List[dict]:
    """Load InfiniteBench task data from JSONL file."""
    assert task_name in DATA_NAME_TO_PATH, f"Unknown task: {task_name}"
    fpath = os.path.join(data_dir, DATA_NAME_TO_PATH[task_name])
    assert os.path.exists(fpath), f"Data file not found: {fpath}"

    samples = []
    with open(fpath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip() == "":
                continue
            if max_examples is not None and len(samples) >= max_examples:
                break
            eg = json.loads(line)
            eg["id"] = eg.get("id", idx)
            samples.append(eg)
    return samples


# ── Tokenization ─────────────────────────────────────────────────────────

def truncate_input(input_ids: torch.Tensor, max_length: int = 127000) -> torch.Tensor:
    """Middle truncation: keep first half + last half."""
    if input_ids.shape[1] <= max_length:
        return input_ids
    half = max_length // 2
    return torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)


def tokenize_infinitebench_input(
    prompt: str,
    tokenizer,
    device: torch.device,
    max_length: int = 127000,
) -> Tuple[torch.Tensor, int]:
    """Tokenize InfiniteBench prompt with BOS, then middle-truncate.

    Uses add_special_tokens=True (adds BOS), matching ReST-KV behavior.
    No chat template applied — the prompt templates already contain
    task instructions and answer prefixes.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = truncate_input(input_ids, max_length)
    prompt_len = input_ids.shape[1]
    return input_ids.to(device), prompt_len


# ── Method: Full KV ──────────────────────────────────────────────────────

def run_fullkv(model, input_ids, max_new_tokens, eos_token_ids, device):
    with torch.inference_mode():
        next_token_logits, past_key_values = prefill_last_token(model, input_ids)
    generated_ids = greedy_decode(
        model=model,
        past_key_values=past_key_values,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        primary_device=device,
    )
    return generated_ids


# ── Method: Ada-SnapKV ───────────────────────────────────────────────────

def run_kvpress(model, input_ids, args, prompt_len, max_new_tokens, eos_token_ids, device):
    """Run inference with kvpress-based compression (Ada-SnapKV)."""
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
        with torch.inference_mode():
            next_token_logits, past_key_values = prefill_last_token(model, input_ids)
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


# ── Method: OBKV ─────────────────────────────────────────────────────────

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
    _kwargs = dict(
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
    _kwargs["eviction_mode"] = getattr(args, "eviction_mode", "joint")
    _kwargs["v_score_type"] = getattr(args, "v_score_type", "attn_linear")
    _kwargs["k_score_type"] = getattr(args, "k_score_type", "think")
    return entry(**_kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InfiniteBench evaluation for RDKV (refactor).")
    p.add_argument("--method", required=True, choices=["fullkv", "snapkv", "ada_snapkv", "obkv"])
    p.add_argument("--task", required=True,
                    help="Comma-separated task names (e.g. 'passkey,number_string')")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-dir", default="data/InfiniteBench/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-num-examples", type=int, default=None)
    p.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa"])
    p.add_argument("--token-budget", type=int, default=1024)
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--alpha-safeguard", type=float, default=0.2)
    p.add_argument("--k-budget-ratio", type=float, default=0.5)
    p.add_argument("--epsilon-path", default=None)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--score-pooling", type=int, default=5)
    p.add_argument("--fp16-topk", type=int, default=0)
    p.add_argument("--v-bit-options", type=str, default=None)
    p.add_argument("--v-score-type", choices=["attn_linear"], default="attn_linear",
                    help="V-side per-token score (sum_tau a over the recent window).")
    p.add_argument("--k-score-type", choices=["think"], default="think",
                    help="K-side per-channel score "
                         "(ThinK: mean_tau(q_c^2) * mean_t(k_c^2)).")
    p.add_argument("--prefill-chunk-size", type=int, default=None,
                    help="If set (e.g. 8192), use two-phase chunked prefill.")
    p.add_argument("--device-map", default="single",
                    choices=["single", "auto", "balanced"])
    p.add_argument("--streaming", action="store_true", default=True,
                    help="Layer-streaming per-head prefill "
                         "(run_obkv). Always on — kept "
                         "for backwards-compat with the legacy CLI.")
    p.add_argument("--pool-type", choices=["avg", "max"], default="avg",
                    help="Per-head score pooling op: avg (default) or max.")
    p.add_argument("--eviction-mode", choices=["joint"], default="joint",
                    help="Per-head joint knapsack over {0,2,4,8,16}.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_determinism(args.seed)

    # Parse task list
    task_list = [t.strip() for t in args.task.split(",")]
    for t in task_list:
        assert t in INFINITEBENCH_TASKS, \
            f"Unknown task '{t}'. Valid: {INFINITEBENCH_TASKS}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== InfiniteBench Evaluation ===")
    print(f"  Method:       {args.method}")
    print(f"  Tasks:        {task_list}")
    print(f"  Token budget: {args.token_budget}")
    if args.method == "obkv":
        print(f"  K ratio:      {args.k_budget_ratio}")
        print(f"  Obs window:   {args.obs_window}")
        print(f"  Score pool:   {args.score_pooling}")
        print(f"  Epsilon:      {args.epsilon_path}")
    print(f"  Output:       {output_dir}")
    if args.max_num_examples:
        print(f"  Max examples: {args.max_num_examples}")
    print()

    # ── Load model ───────────────────────────────────────────────────
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

    # ── Inference loop ───────────────────────────────────────────────
    total_samples = 0
    total_oom = 0
    t_start = time.time()

    for task_name in task_list:
        print(f"\n--- Task: {task_name} ---")
        samples = load_infinitebench_task(
            args.data_dir, task_name, args.max_num_examples,
        )
        print(f"  Loaded {len(samples)} samples")

        max_new_tokens = DATASET2MAXLEN[task_name]
        jsonl_path = output_dir / f"{task_name}.jsonl"
        fout = open(jsonl_path, "w", encoding="utf-8")

        for idx, eg in enumerate(samples):
            prompt = create_prompt(eg, task_name)
            ground_truth = get_answer(eg, task_name)

            input_ids, prompt_len = tokenize_infinitebench_input(
                prompt, tokenizer, primary_device,
            )

            print(f"  [{idx+1}/{len(samples)}] task={task_name} "
                  f"id={eg['id']} prompt_len={prompt_len} "
                  f"max_gen={max_new_tokens}")

            generated_ids = []
            try:
                if args.method == "fullkv":
                    generated_ids = run_fullkv(
                        model, input_ids, max_new_tokens,
                        eos_token_ids, primary_device,
                    )
                elif args.method in ("snapkv", "ada_snapkv"):
                    generated_ids = run_kvpress(
                        model, input_ids, args, prompt_len,
                        max_new_tokens, eos_token_ids, primary_device,
                    )
                elif args.method == "obkv":
                    generated_ids = run_obkv_method(
                        model, input_ids, args, epsilon_K, epsilon_V,
                        max_new_tokens, eos_token_ids, primary_device,
                        task_name=task_name,
                        sample_id=eg["id"],
                    )
            except torch.cuda.OutOfMemoryError:
                total_oom += 1
                print(f"  [OOM] task={task_name} id={eg['id']} "
                      f"prompt_len={prompt_len}")
                gc.collect()
                torch.cuda.empty_cache()

            prediction = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip() if generated_ids else ""

            result = {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "dataset": task_name,
                "id": eg["id"],
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            total_samples += 1
            if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
                elapsed = time.time() - t_start
                print(f"  Progress: {idx+1}/{len(samples)} "
                      f"({elapsed:.0f}s total)")

        fout.close()
        print(f"  Written: {jsonl_path}")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n=== Done ===")
    print(f"  Total samples: {total_samples}")
    print(f"  OOM count:     {total_oom}")
    print(f"  Time:          {elapsed:.0f}s ({elapsed/max(total_samples,1):.1f}s/sample)")
    print(f"  Output dir:    {output_dir}")

    summary = {
        "method": args.method,
        "tasks": task_list,
        "token_budget": args.token_budget,
        "total_samples": total_samples,
        "oom_count": total_oom,
        "elapsed_seconds": round(elapsed, 1),
    }
    if args.method == "obkv":
        summary["k_budget_ratio"] = args.k_budget_ratio
        summary["epsilon_path"] = args.epsilon_path
        summary["obs_window"] = args.obs_window
        summary["score_pooling"] = args.score_pooling
        summary["v_score_type"] = args.v_score_type

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
