#!/usr/bin/env python3
"""LongBench evaluation runner for OBKV.

Loads a LongBench shard manifest and runs obkv_fast.run_obkv() on each sample,
writing per-task JSONL predictions and a summary.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from obkv_fast import (
    DEFAULT_EPSILON_K,
    DEFAULT_EPSILON_V,
    DEFAULT_MODEL_PATH,
    ensure_dir,
    load_epsilon_kv_from_calibration,
    load_json,
    load_model_and_tokenizer,
    middle_truncate_token_ids,
    patch_llama31_rope_compat,
    run_obkv,
    set_determinism,
)


NO_CHAT_WRAP_TASKS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}


LONGBENCH_TASKS: Tuple[str, ...] = (
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
)



def resolve_longbench_root(repo_root: Path) -> Path:
    raw_candidates = []
    env_path = os.environ.get("LONGBENCH_REPO")
    if env_path:
        raw_candidates.append(Path(env_path))
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        raw_candidates.append(Path(project_root) / "external" / "LongBench")
    raw_candidates.append(repo_root / "external" / "LongBench")
    raw_candidates.append(
        Path("<SCRATCH>/RDKV/external/LongBench")
    )

    candidates = []
    for candidate in raw_candidates:
        candidates.append(candidate)
        candidates.append(candidate / "LongBench")
    candidates.append(
        Path("<SCRATCH>/RDKV/external/LongBench/LongBench")
    )
    for candidate in candidates:
        if (candidate / "config" / "dataset2prompt.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate LongBench config root. Set LONGBENCH_REPO or PROJECT_ROOT."
    )



def should_apply_chat_wrapper(task_name: str, tokenizer) -> bool:
    if task_name in NO_CHAT_WRAP_TASKS:
        return False
    return hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None



def build_prompt_input_ids(
    sample: Dict[str, object],
    task_name: str,
    tokenizer,
    prompt_format: str,
    max_prompt_tokens: int,
) -> torch.Tensor:
    prompt = prompt_format.format(**sample)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if raw_ids.shape[0] > max_prompt_tokens:
        raw_ids = middle_truncate_token_ids(raw_ids, max_prompt_tokens)
        prompt = tokenizer.decode(
            raw_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    if should_apply_chat_wrapper(task_name, tokenizer):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = middle_truncate_token_ids(input_ids[0], max_prompt_tokens).unsqueeze(0)
    return input_ids



def load_manifest(repo_root: Path, num_shards: int, shard_index: int) -> Tuple[Path, Dict[str, object]]:
    manifest_path = (
        repo_root
        / "results"
        / "longbench"
        / "sample_manifests"
        / f"test_num_shards_{num_shards}_shard_{shard_index}.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
    return manifest_path, load_json(manifest_path)



def load_shard_examples(
    repo_root: Path,
    manifest: Dict[str, object],
    max_examples: Optional[int] = None,
    task_filter: Optional[set] = None,
) -> List[Tuple[str, Dict[str, object]]]:
    from datasets import load_dataset

    # Look for longbench_dataset.py next to this script first, then repo_root
    dataset_script_path = Path(__file__).resolve().parent / "longbench_dataset.py"
    if not dataset_script_path.exists():
        dataset_script_path = repo_root / "longbench_dataset.py"
    if not dataset_script_path.exists():
        raise FileNotFoundError(f"Missing dataset script: {dataset_script_path}")

    work_items: List[Tuple[str, Dict[str, object]]] = []
    for task_payload in manifest["tasks"]:
        task_name = task_payload["task"]
        if task_filter is not None and task_name not in task_filter:
            continue
        example_ids = task_payload["example_ids"]
        dataset = load_dataset(
            str(dataset_script_path),
            task_name,
            split="test",
            trust_remote_code=True,
        )
        selected_ids = set(example_ids)
        dataset_by_id = {}
        for sample in dataset:
            sample_id = sample["_id"]
            if sample_id in selected_ids:
                dataset_by_id[sample_id] = sample
        missing = [example_id for example_id in example_ids if example_id not in dataset_by_id]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"{task_name}: missing example ids from dataset: {preview}")
        for example_id in example_ids:
            work_items.append((task_name, dataset_by_id[example_id]))
            if max_examples is not None and len(work_items) >= max_examples:
                return work_items
    return work_items



def load_prompt_configs(longbench_root: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    config_dir = longbench_root / "config"
    prompt_path = config_dir / "dataset2prompt.json"
    maxlen_path = config_dir / "dataset2maxlen.json"
    if not prompt_path.exists() or not maxlen_path.exists():
        raise FileNotFoundError(f"Missing LongBench config files under {config_dir}")
    return load_json(prompt_path), load_json(maxlen_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongBench evaluation with OBKV inference.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Path to RDKV root (holds longbench_dataset.py and "
                             "results/longbench/sample_manifests/). Defaults to obkv/'s parent.")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--device-map", default="single",
                        choices=["single", "auto", "balanced", "cpu_offload"])
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--task-filter", type=str, default=None,
                        help="Comma-separated task names to run (subset of 16 LongBench tasks).")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--force-max-new-tokens", action="store_true",
                        help="Ignore EOS for fixed-length correctness/perf smoke.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--token-budget", type=int, default=1024,
                        help="Global token budget B for OBKV.")
    parser.add_argument("--k-budget-ratio", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--obs-window", type=int, default=32)
    parser.add_argument("--pool-kernel-size", type=int, default=5)
    parser.add_argument("--pool-padding", type=str, default="reflect",
                        choices=["reflect", "zero"])
    parser.add_argument("--v-score-type", choices=["attn_linear"], default="attn_linear",
                        help="V-side per-token score (sum_tau a over the recent window).")
    parser.add_argument("--k-score-type", choices=["think"], default="think",
                        help="K-side per-channel score "
                             "(ThinK: mean_tau(q_c^2) * mean_t(k_c^2)).")
    parser.add_argument("--fp16-topk", type=int, default=0)
    parser.add_argument("--v-bit-options", type=str, default="0,2,4,8,16")
    parser.add_argument("--k-bit-options", type=str, default="0,2,4,8,16")
    parser.add_argument("--no-prepend-zero", action="store_true",
                        help="Do not auto-prepend 0 to v/k bit options. "
                             "Required for the no-eviction (quant-only) ablation "
                             "where every token must keep at least 2 bits.")
    parser.add_argument("--epsilon-path", type=str, default=None,
                        help="Optional calibration artifact with per-layer K/V epsilons.")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Layer-streaming per-head prefill "
                             "(run_obkv). Always on — kept "
                             "for backwards-compat with the legacy CLI.")
    parser.add_argument("--eviction-mode", choices=["joint"], default="joint",
                        help="Per-head joint knapsack over {0,2,4,8,16}.")
    parser.add_argument("--decode-blockwise-rdkv", action="store_true",
                        help="Periodically allocate and pack decode KV blocks.")
    parser.add_argument("--decode-block-size", type=int, default=64)
    parser.add_argument("--decode-block-budget-tokens", type=int, default=8,
                        help="Per-layer/per-head decode-block budget in raw "
                             "FP16-equivalent bits (not a kept-token count).")
    parser.add_argument(
        "--decode-final-flush",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress the final decode block with causal self queries.",
    )
    parser.add_argument("--decode-query-source", choices=["next_block"],
                        default="next_block")
    parser.add_argument("--decode-final-query-source", choices=["self_block"],
                        default="self_block")
    parser.add_argument("--decode-correctness-check", action="store_true")
    parser.add_argument("--log-block-timing", action="store_true")
    parser.add_argument("--log-cache-memory", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip example IDs already present in task JSONL.")

    return parser.parse_args()


def _parse_bit_options(spec: str, prepend_zero: bool = True) -> torch.Tensor:
    bits = sorted({int(b.strip()) for b in spec.split(",") if b.strip()})
    if prepend_zero and 0 not in bits:
        bits.insert(0, 0)
    return torch.tensor(bits, dtype=torch.float32)



def build_summary(
    args: argparse.Namespace,
    manifest_path: Path,
    results_by_task: Dict[str, List[Dict[str, object]]],
    longbench_root: Path,
) -> Dict[str, object]:
    return {
        "model_path": args.model_path,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "sample_manifest_path": str(manifest_path),
        "longbench_root": str(longbench_root),
        "token_budget": args.token_budget,
        "k_budget_ratio": args.k_budget_ratio,
        "obs_window": args.obs_window,
        "pool_kernel_size": args.pool_kernel_size,
        "pool_padding": args.pool_padding,
        "v_score_type": args.v_score_type,
        "fp16_topk": args.fp16_topk,
        "decode_blockwise_rdkv": args.decode_blockwise_rdkv,
        "decode_block_size": args.decode_block_size,
        "decode_block_budget_tokens": args.decode_block_budget_tokens,
        "decode_final_flush": args.decode_final_flush,
        "decode_query_source": args.decode_query_source,
        "decode_final_query_source": args.decode_final_query_source,
        "decode_correctness_check": args.decode_correctness_check,
        "log_block_timing": args.log_block_timing,
        "log_cache_memory": args.log_cache_memory,
        "tasks": {
            task_name: {"num_examples": len(rows)}
            for task_name, rows in sorted(results_by_task.items())
        },
        "num_predictions": sum(len(rows) for rows in results_by_task.values()),
    }



def write_results(
    output_dir: Path,
    results_by_task: Dict[str, List[Dict[str, object]]],
    summary: Dict[str, object],
) -> None:
    ensure_dir(output_dir)
    for task_name, rows in results_by_task.items():
        output_path = output_dir / f"{task_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_results(output_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    results: Dict[str, List[Dict[str, object]]] = {}
    for task_name in LONGBENCH_TASKS:
        path = output_dir / f"{task_name}.jsonl"
        if not path.exists():
            continue
        rows = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        if rows:
            results[task_name] = rows
    return results


def write_blockwise_summaries(
    output_dir: Path,
    results_by_task: Dict[str, List[Dict[str, object]]],
    per_sample_rows: List[Dict[str, object]],
) -> None:
    if not per_sample_rows:
        return

    def mean(key: str) -> float:
        values = [
            float(row[key]) for row in per_sample_rows
            if row.get(key) is not None
        ]
        return sum(values) / max(1, len(values))

    timing_summary = {
        "num_samples": len(per_sample_rows),
        "mean_avg_tpot_ms": mean("avg_tpot_ms"),
        "mean_completion_latency_ms": mean("completion_latency_ms"),
        "mean_final_flush_time_ms": mean("final_flush_time_ms"),
        "mean_peak_gpu_allocated_bytes": mean("peak_gpu_allocated_bytes"),
        "mean_peak_gpu_reserved_bytes": mean("peak_gpu_reserved_bytes"),
    }
    with (output_dir / "timing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(timing_summary, file, indent=2)
        file.write("\n")

    task_summary = {}
    for task_name, rows in sorted(results_by_task.items()):
        scores = [float(row["score"]) for row in rows if "score" in row]
        task_summary[task_name] = {
            "num_examples": len(rows),
            "score": sum(scores) / max(1, len(scores)),
        }
    with (output_dir / "task_summary.json").open("w", encoding="utf-8") as file:
        json.dump(task_summary, file, indent=2)
        file.write("\n")



def main() -> None:
    args = parse_args()
    if args.decode_blockwise_rdkv:
        allow_experimental = (
            os.environ.get("LONGBENCH_ALLOW_EXPERIMENTAL_BLOCKWISE", "0") == "1"
        )
        if not allow_experimental and args.decode_block_size != 64:
            raise ValueError(
                "The rebuttal protocol fixes --decode-block-size to 64."
            )
        if (
            not allow_experimental
            and args.decode_block_budget_tokens not in (8, 16)
        ):
            raise ValueError(
                "The rebuttal protocol permits decode budgets 8 or 16 only."
            )
        if not args.decode_final_flush:
            raise ValueError("Blockwise rebuttal runs require final flush.")
    if args.repo_root is not None:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
    longbench_root = resolve_longbench_root(repo_root)
    manifest_path, manifest = load_manifest(repo_root, args.num_shards, args.shard)
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError(f"`max_examples` must be positive, got {args.max_examples}.")

    set_determinism(args.seed)

    prompt_formats, task_maxlen = load_prompt_configs(longbench_root)
    filter_set = {t.strip() for t in args.task_filter.split(",")} if args.task_filter else None
    work_items = load_shard_examples(
        repo_root,
        manifest,
        max_examples=args.max_examples,
        task_filter=filter_set,
    )
    if filter_set and not work_items:
        raise ValueError(f"No examples found for task_filter={args.task_filter!r}")

    model, tokenizer, primary_device = load_model_and_tokenizer(
        args.model_path,
        mode="value_residual_probe",
        attn_implementation=args.attn_implementation,
        device_map_arg=args.device_map,
    )
    compute_device = primary_device if primary_device.type == "cuda" else torch.device("cpu")
    eos_token_ids = model.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    elif eos_token_ids is None:
        eos_token_ids = [tokenizer.eos_token_id]

    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if max_position_embeddings is None:
        raise ValueError("Model config is missing `max_position_embeddings`.")
    # Reserve enough prompt room for the longest per-task maxlen across tasks we
    # may run. ``--max-new-tokens`` acts as a fallback cap for tasks missing
    # from dataset2maxlen.json, but the per-task value (e.g. 512 for summarization,
    # 32 for triviaqa) is what the decode call actually receives below.
    _decode_budget_ub = max(
        [args.max_new_tokens] + [
            int(v) for k, v in task_maxlen.items()
            if not filter_set or k in filter_set
        ]
    )
    max_prompt_tokens = max(1, max_position_embeddings - _decode_budget_ub)

    epsilon_K = None
    epsilon_V = None
    if args.epsilon_path:
        epsilon_K, epsilon_V = load_epsilon_kv_from_calibration(args.epsilon_path)
    else:
        epsilon_K, epsilon_V = DEFAULT_EPSILON_K, DEFAULT_EPSILON_V

    prepend_zero = not args.no_prepend_zero
    v_bit_options = _parse_bit_options(args.v_bit_options, prepend_zero=prepend_zero)
    k_bit_options = _parse_bit_options(args.k_bit_options, prepend_zero=prepend_zero)

    output_dir = Path(args.output_dir).expanduser()
    ensure_dir(output_dir)
    results_by_task = (
        load_existing_results(output_dir) if args.resume else {}
    )
    auxiliary_names = (
        "predictions.jsonl",
        "per_sample_metrics.jsonl",
        "block_events.jsonl",
        "memory_trace.jsonl",
    )
    if not args.resume:
        for name in auxiliary_names:
            (output_dir / name).unlink(missing_ok=True)
        for task_name in LONGBENCH_TASKS:
            (output_dir / f"{task_name}.jsonl").unlink(missing_ok=True)
    existing_ids = {
        str(row["example_id"])
        for rows in results_by_task.values()
        for row in rows
    }
    per_sample_rows: List[Dict[str, object]] = []
    metrics_path = output_dir / "per_sample_metrics.jsonl"
    if args.resume and metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as file:
            per_sample_rows = [
                json.loads(line) for line in file if line.strip()
            ]

    config_payload = vars(args).copy()
    config_payload["output_dir"] = str(output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    for example_idx, (task_name, sample) in enumerate(work_items, start=1):
        if str(sample["_id"]) in existing_ids:
            print(
                f"[resume] skipping {task_name}/{sample['_id']}",
                flush=True,
            )
            continue
        input_ids = build_prompt_input_ids(
            sample=sample,
            task_name=task_name,
            tokenizer=tokenizer,
            prompt_format=prompt_formats[task_name],
            max_prompt_tokens=max_prompt_tokens,
        ).to(primary_device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(primary_device)
        start_ts = time.perf_counter()
        entry = run_obkv
        # LONGBENCH_FORCE_CLI_MAX_NEW=1 lets the sbatch override the official
        # per-task dataset2maxlen.json value with the CLI --max-new-tokens.
        # Default behaviour (env unset / "0") keeps per-task values.
        if os.environ.get("LONGBENCH_FORCE_CLI_MAX_NEW", "0") == "1":
            task_max_new = int(args.max_new_tokens)
        else:
            task_max_new = int(task_maxlen.get(task_name, args.max_new_tokens))
        _entry_kwargs = dict(
            model=model,
            input_ids=input_ids,
            token_budget=args.token_budget,
            k_budget_ratio=args.k_budget_ratio,
            epsilon_K=epsilon_K,
            epsilon_V=epsilon_V,
            max_new_tokens=task_max_new,
            eos_token_ids=[] if args.force_max_new_tokens else eos_token_ids,
            chunk_size=args.chunk_size,
            obs_window=args.obs_window,
            pool_kernel_size=args.pool_kernel_size,
            pool_padding=args.pool_padding,
            v_bit_options=v_bit_options,
            k_bit_options=k_bit_options,
            fp16_topk=args.fp16_topk,
            device=primary_device,
        )
        _entry_kwargs["eviction_mode"] = getattr(args, "eviction_mode", "joint")
        _entry_kwargs["v_score_type"] = args.v_score_type
        _entry_kwargs["k_score_type"] = args.k_score_type
        runtime_metrics: Dict[str, object] = {}
        _entry_kwargs.update(
            decode_blockwise_rdkv=args.decode_blockwise_rdkv,
            decode_block_size=args.decode_block_size,
            decode_block_budget_tokens=args.decode_block_budget_tokens,
            decode_final_flush=args.decode_final_flush,
            decode_correctness_check=args.decode_correctness_check,
            runtime_metrics=runtime_metrics,
        )
        generated_ids = entry(**_entry_kwargs)
        elapsed = time.perf_counter() - start_ts

        prediction = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result_row = {
            "task": task_name,
            "example_id": sample["_id"],
            "prediction": prediction,
            "references": list(sample["answers"] or []),
            "all_classes": list(sample.get("all_classes") or []),
        }
        from eval_longbench import score_prediction
        result_row["score"] = score_prediction(
            task_name,
            prediction,
            result_row["references"],
            result_row["all_classes"],
        )
        if "length" in sample:
            result_row["length"] = int(sample["length"])
        results_by_task.setdefault(task_name, []).append(result_row)
        existing_ids.add(str(sample["_id"]))
        append_jsonl(output_dir / f"{task_name}.jsonl", result_row)
        append_jsonl(output_dir / "predictions.jsonl", result_row)

        sample_metrics = {
            "task": task_name,
            "sample_id": sample["_id"],
            "score": result_row["score"],
            "elapsed_sec": elapsed,
            **{
                key: value
                for key, value in runtime_metrics.items()
                if key not in ("block_events", "memory_trace")
            },
        }
        append_jsonl(metrics_path, sample_metrics)
        per_sample_rows.append(sample_metrics)
        for event in runtime_metrics.get("block_events", []):
            append_jsonl(
                output_dir / "block_events.jsonl",
                {
                    "task": task_name,
                    "sample_id": sample["_id"],
                    **event,
                },
            )
        for memory_row in runtime_metrics.get("memory_trace", []):
            append_jsonl(
                output_dir / "memory_trace.jsonl",
                {
                    "task": task_name,
                    "sample_id": sample["_id"],
                    **memory_row,
                },
            )
        print(json.dumps({
            "example_index": example_idx,
            "task": task_name,
            "example_id": sample["_id"],
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": len(generated_ids),
            "max_new_tokens": task_max_new,
            "elapsed_sec": round(elapsed, 3),
        }, ensure_ascii=False))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = build_summary(args, manifest_path, results_by_task, longbench_root)
    write_results(output_dir, results_by_task, summary)
    write_blockwise_summaries(
        output_dir, results_by_task, per_sample_rows
    )
    print(f"[done] wrote results to {output_dir}")


if __name__ == "__main__":
    main()
