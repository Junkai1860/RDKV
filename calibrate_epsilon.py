"""
calibrate_epsilon.py

Measure normalised quantisation distortion epsilon(b) for KV cache
at different bit-widths.  This is a prerequisite for the knapsack
bit-width allocation scheme.

Supports three data sources:
  - longbench:     LongBench context fields, truncated to max_seq_len
  - ruler:         Pre-generated RULER JSONL data at exact target lengths
  - infinitebench: InfiniteBench JSONL data (128K+ sequences)

Usage (LongBench):
    python calibrate_epsilon.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --num_samples 32 \
        --max_seq_len 8192 \
        --output_path results/epsilon_calibration.json

Usage (RULER, all 6 lengths):
    python calibrate_epsilon.py \
        --data_source ruler \
        --ruler_data_dir $SCR/ruler_data/llama31_8b_instruct \
        --ruler_lengths 4096,8192,16384,32768,65536,131072 \
        --num_samples_per_length 8 \
        --output_path results/epsilon_calibration_mixed_lengths.json \
        --per_length_output_path results/epsilon_calibration_per_length.json
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the Llama-3.1 RoPE compatibility patch from obkv_fast.py
# so that transformers 4.38 can load Llama-3.1 models.
from obkv_fast import patch_llama31_rope_compat
patch_llama31_rope_compat()


# ── English-only LongBench v1 tasks (16 tasks) ──────────────────────
TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]


# ── Fake quantiser ──────────────────────────────────────────────────
def fake_quantize(x: torch.Tensor, n_bits: int, dim: int) -> torch.Tensor:
    """Uniform asymmetric fake quantisation along *dim*.

    Args:
        x:      input tensor
        n_bits: quantisation bit-width
        dim:    axis along which min/max are computed
                K per-channel → dim=2 (seq_len); V per-token → dim=3 (head_dim)
    Returns:
        x_hat:  dequantised tensor, same shape as x
    """
    qmin, qmax = 0, 2 ** n_bits - 1
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    scale = (x_max - x_min) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = qmin - x_min / scale
    zero_point = zero_point.clamp(qmin, qmax).round()
    x_q = (x / scale + zero_point).clamp(qmin, qmax).round()
    x_hat = (x_q - zero_point) * scale
    return x_hat


def calibrate_epsilon_per_channel(
    K: torch.Tensor,
    bit_widths: list,
) -> dict:
    """Compute per-channel normalised MSE for K cache quantisation.

    For each channel c in head_dim, extract K[:, :, :, c] (shape [B, H, T]),
    unsqueeze to [B, H, T, 1], fake-quantise with dim=2 (per-channel),
    compute mse/var for that channel, then average across all channels.

    Args:
        K:          [B, num_kv_heads, seq_len, head_dim] in float32
        bit_widths: list of bit-widths to evaluate, e.g. [2, 4, 8]

    Returns:
        {bit: mean_per_channel_nmse} for each bit in bit_widths
    """
    head_dim = K.shape[3]
    result = {}
    for b in bit_widths:
        channel_nmse = []
        for c in range(head_dim):
            col = K[:, :, :, c].unsqueeze(-1)       # [B, H, T, 1]
            col_hat = fake_quantize(col, b, dim=2)   # per-channel quant
            mse = ((col - col_hat) ** 2).mean()
            var = col.var()
            nmse = (mse / var).item() if var.item() > 1e-12 else 0.0
            channel_nmse.append(nmse)
        result[b] = statistics.mean(channel_nmse)
    return result


# ── KV extraction (compatible with DynamicCache & tuple-of-tuple) ──
def get_kv_from_past(past_kv, layer_idx):
    if hasattr(past_kv, "key_cache"):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    return past_kv[layer_idx][0], past_kv[layer_idx][1]


def num_layers_from_past(past_kv):
    if hasattr(past_kv, "key_cache"):
        return len(past_kv.key_cache)
    return len(past_kv)


# ── Data loading (reuses local longbench_dataset.py) ────────────────
def load_samples(dataset_script, tokenizer, num_samples, max_seq_len):
    per_task = max(1, num_samples // len(TASKS))
    samples = []
    for task in TASKS:
        ds = load_dataset(str(dataset_script), task, split="test", trust_remote_code=True)
        for i, row in enumerate(ds):
            if i >= per_task:
                break
            text = row["context"]
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   truncation=True, max_length=max_seq_len)
            if len(ids) < 32:
                continue
            samples.append((task, torch.tensor([ids])))
        if len(samples) >= num_samples:
            break
    return samples[:num_samples]


# ── RULER data loading ─────────────────────────────────────────────
RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


def load_ruler_samples(ruler_data_dir, tokenizer, seq_lengths,
                       num_samples_per_length):
    """Load RULER samples across multiple sequence lengths.

    Returns:
        list of (seq_length, task_name, input_ids_tensor) tuples
    """
    samples = []
    for seq_len in seq_lengths:
        count = 0
        for task in RULER_TASKS:
            jsonl_path = Path(ruler_data_dir) / str(seq_len) / task / "validation.jsonl"
            if not jsonl_path.exists():
                print(f"  [WARN] Missing {jsonl_path}, skipping")
                continue
            with open(jsonl_path) as f:
                for line in f:
                    if count >= num_samples_per_length:
                        break
                    row = json.loads(line.strip())
                    ids = tokenizer.encode(
                        row["input"], add_special_tokens=False,
                        truncation=True, max_length=seq_len + 512,
                    )
                    if len(ids) < 32:
                        continue
                    samples.append((seq_len, task, torch.tensor([ids])))
                    count += 1
            if count >= num_samples_per_length:
                break
        print(f"  Loaded {count} samples for seq_length={seq_len}")
    return samples


# ── InfiniteBench data loading ─────────────────────────────────────
INFINITEBENCH_TASKS = {
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


def load_infinitebench_samples(data_dir, task, tokenizer, num_samples, max_seq_len):
    """Load InfiniteBench samples for a single task.

    Concatenates context + input fields, tokenizes, and truncates
    (middle truncation) to max_seq_len.

    Returns:
        list of (label, input_ids_tensor) tuples
    """
    assert task in INFINITEBENCH_TASKS, (
        f"Unknown InfiniteBench task: {task}. "
        f"Available: {list(INFINITEBENCH_TASKS.keys())}"
    )
    fpath = Path(data_dir) / INFINITEBENCH_TASKS[task]
    assert fpath.exists(), f"Data file not found: {fpath}"

    samples = []
    with open(fpath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(samples) >= num_samples:
                break
            if line.strip() == "":
                continue
            eg = json.loads(line)
            text = eg.get("context", "") + "\n" + eg.get("input", "")
            ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
            # Middle truncation: keep first half + last half
            if ids.shape[1] > max_seq_len:
                half = max_seq_len // 2
                ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
            if ids.shape[1] < 32:
                continue
            samples.append((f"{task}/{idx}", ids))
    return samples


# ── Main ────────────────────────────────────────────────────────────
def aggregate_nmse(K_nmse, V_nmse, K_pc_nmse, bit_widths):
    """Aggregate per-sample NMSE values into a results dict."""
    results = {}
    n_layers_total = len(K_nmse[bit_widths[0]])

    results["0"] = {
        "K_mean": 1.0, "V_mean": 1.0,
        "K_std": 0.0, "V_std": 0.0,
        "K_per_layer_mean": [1.0] * n_layers_total,
        "V_per_layer_mean": [1.0] * n_layers_total,
        "K_per_channel_mean": 1.0,
    }

    for b in bit_widths:
        k_per_layer = []
        v_per_layer = []
        for l in range(n_layers_total):
            k_per_layer.append(sum(K_nmse[b][l]) / len(K_nmse[b][l]))
            v_per_layer.append(sum(V_nmse[b][l]) / len(V_nmse[b][l]))

        k_all = [v for vals in K_nmse[b].values() for v in vals]
        v_all = [v for vals in V_nmse[b].values() for v in vals]

        results[str(b)] = {
            "K_mean": statistics.mean(k_all),
            "K_std": statistics.stdev(k_all) if len(k_all) > 1 else 0.0,
            "K_per_layer_mean": k_per_layer,
            "V_mean": statistics.mean(v_all),
            "V_std": statistics.stdev(v_all) if len(v_all) > 1 else 0.0,
            "V_per_layer_mean": v_per_layer,
            "K_per_channel_mean": statistics.mean(K_pc_nmse[b]),
        }

    results["16"] = {
        "K_mean": 0.0, "V_mean": 0.0,
        "K_std": 0.0, "V_std": 0.0,
        "K_per_layer_mean": [0.0] * n_layers_total,
        "V_per_layer_mean": [0.0] * n_layers_total,
        "K_per_channel_mean": 0.0,
    }
    return results


def print_epsilon_table(results, label=""):
    """Print a summary table of epsilon values."""
    if label:
        print(f"\n=== epsilon(b) — {label} ===")
    else:
        print("\n=== epsilon(b) Calibration Results ===")
    print(f" bit |        K_eps |        V_eps | K_per_ch_eps")
    print("-" * 55)
    for b_str in ["0", "2", "4", "8", "16"]:
        km = results[b_str]["K_mean"]
        vm = results[b_str]["V_mean"]
        kpc = results[b_str]["K_per_channel_mean"]
        print(f"  {b_str:>2} | {km:12.6f} | {vm:12.6f} | {kpc:12.6f}")


def run_calibration_on_samples(model, samples, bit_widths, chunk_size=4096):
    """Run forward pass on samples and collect NMSE values.

    Args:
        model: loaded model
        samples: list of (label, input_ids_tensor) tuples
        bit_widths: list of bit widths to evaluate
        chunk_size: number of tokens per forward-pass chunk (default 4096).
                    If chunk_size >= seq_len, equivalent to a single forward pass.

    Returns:
        (K_nmse, V_nmse, K_pc_nmse) accumulators
    """
    K_nmse = {b: {} for b in bit_widths}
    V_nmse = {b: {} for b in bit_widths}
    K_pc_nmse = {b: [] for b in bit_widths}

    for s_idx, (label, input_ids) in enumerate(samples):
        seq_len = input_ids.shape[1]
        print(f"  [{s_idx+1}/{len(samples)}] {label}  seq_len={seq_len}")
        input_ids = input_ids.to(model.device)

        past_key_values = None
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        with torch.no_grad():
            for chunk_idx, i in enumerate(range(0, seq_len, chunk_size)):
                chunk_ids = input_ids[:, i:i+chunk_size]
                if n_chunks > 1:
                    print(f"    chunk {chunk_idx+1}/{n_chunks}", end="\r")
                out = model(chunk_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                del out
        if n_chunks > 1:
            print()
        past_kv = past_key_values
        n_layers = num_layers_from_past(past_kv)

        sample_K_pc = {b: [] for b in bit_widths}

        for layer_idx in range(n_layers):
            K, V = get_kv_from_past(past_kv, layer_idx)
            K = K.float()
            V = V.float()
            K_norm_sq = (K * K).sum()
            V_norm_sq = (V * V).sum()

            for b in bit_widths:
                K_hat = fake_quantize(K, b, dim=2)
                V_hat = fake_quantize(V, b, dim=3)
                k_err = ((K - K_hat) ** 2).sum() / K_norm_sq
                v_err = ((V - V_hat) ** 2).sum() / V_norm_sq
                K_nmse[b].setdefault(layer_idx, []).append(k_err.item())
                V_nmse[b].setdefault(layer_idx, []).append(v_err.item())

            pc = calibrate_epsilon_per_channel(K, bit_widths)
            for b in bit_widths:
                sample_K_pc[b].append(pc[b])

        for b in bit_widths:
            K_pc_nmse[b].append(statistics.mean(sample_K_pc[b]))

        del past_kv, past_key_values
        torch.cuda.empty_cache()

    return K_nmse, V_nmse, K_pc_nmse


def main():
    parser = argparse.ArgumentParser(description="Calibrate epsilon(b) for KV cache quantisation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Total samples for longbench mode")
    parser.add_argument("--max_seq_len", type=int, default=8192,
                        help="Max seq len for longbench mode")
    parser.add_argument("--output_path", type=str, default="results/epsilon_calibration.json")
    parser.add_argument("--device_map", type=str, default="auto",
                        choices=["auto", "balanced"],
                        help="Transformers device_map for model loading")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Prefill chunk size for long sequences")
    # Data source
    parser.add_argument("--data_source", type=str, default="longbench",
                        choices=["longbench", "ruler", "infinitebench"])
    parser.add_argument("--ruler_data_dir", type=str, default=None,
                        help="Root dir of pre-generated RULER data")
    parser.add_argument("--ruler_lengths", type=str, default="4096,8192,16384,32768,65536,131072",
                        help="Comma-separated seq lengths for RULER mode")
    parser.add_argument("--num_samples_per_length", type=int, default=8,
                        help="Samples per length for RULER mode")
    parser.add_argument("--per_length_output_path", type=str, default=None,
                        help="Output per-length epsilon breakdown (RULER mode only)")
    # InfiniteBench mode
    parser.add_argument("--task", type=str, default=None,
                        help="InfiniteBench task name (e.g. number_string)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    print(f"Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading model from {args.model_name} (FP16, device_map={args.device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    bit_widths = [2, 4, 8]

    if args.data_source == "ruler":
        # ── RULER mode: calibrate across multiple lengths ────────────
        if not args.ruler_data_dir:
            sys.exit("ERROR: --ruler_data_dir required for ruler mode")

        seq_lengths = [int(x) for x in args.ruler_lengths.split(",")]
        print(f"Loading RULER samples: lengths={seq_lengths}, "
              f"n_per_length={args.num_samples_per_length} ...")
        ruler_samples = load_ruler_samples(
            args.ruler_data_dir, tokenizer, seq_lengths,
            args.num_samples_per_length,
        )
        print(f"  Total: {len(ruler_samples)} samples")

        # ── Per-length calibration ──────────────────────────────────
        per_length_results = {}
        # Group samples by seq_length
        samples_by_length = {}
        for seq_len, task, input_ids in ruler_samples:
            samples_by_length.setdefault(seq_len, []).append(
                (f"ruler/{task}/{seq_len}", input_ids)
            )

        for seq_len in seq_lengths:
            length_samples = samples_by_length.get(seq_len, [])
            if not length_samples:
                print(f"\n[SKIP] No samples for seq_length={seq_len}")
                continue
            print(f"\n--- Calibrating seq_length={seq_len} "
                  f"({len(length_samples)} samples) ---")
            k_n, v_n, k_pc = run_calibration_on_samples(
                model, length_samples, bit_widths, chunk_size=args.chunk_size,
            )
            per_length_results[seq_len] = aggregate_nmse(k_n, v_n, k_pc, bit_widths)
            print_epsilon_table(per_length_results[seq_len], label=f"{seq_len}")

        # ── All-length aggregate ────────────────────────────────────
        print(f"\n--- Calibrating ALL lengths (aggregate) ---")
        all_samples = [(f"ruler/{task}/{sl}", ids)
                       for sl, task, ids in ruler_samples]
        k_n, v_n, k_pc = run_calibration_on_samples(
            model, all_samples, bit_widths, chunk_size=args.chunk_size,
        )
        results = aggregate_nmse(k_n, v_n, k_pc, bit_widths)
        print_epsilon_table(results, label="ALL lengths averaged")

        # ── Save per-length breakdown ───────────────────────────────
        if args.per_length_output_path:
            per_length_out = {}
            for seq_len, res in per_length_results.items():
                per_length_out[str(seq_len)] = res
            os.makedirs(os.path.dirname(args.per_length_output_path) or ".", exist_ok=True)
            with open(args.per_length_output_path, "w") as f:
                json.dump(per_length_out, f, indent=2)
            print(f"\nSaved per-length → {args.per_length_output_path}")

            # ── Per-length comparison table ─────────────────────────
            print("\n=== Per-Length Comparison ===")
            print(f" {'length':>7s} | {'K_2':>10s} {'K_4':>10s} {'K_8':>10s} | "
                  f"{'V_2':>10s} {'V_4':>10s} {'V_8':>10s} | "
                  f"{'Kpc_2':>10s} {'Kpc_4':>10s} {'Kpc_8':>10s}")
            print("-" * 115)
            for seq_len in seq_lengths:
                res = per_length_results.get(seq_len)
                if not res:
                    continue
                row = f" {seq_len:>7d} |"
                for b in [2, 4, 8]:
                    row += f" {res[str(b)]['K_mean']:10.6f}"
                row += " |"
                for b in [2, 4, 8]:
                    row += f" {res[str(b)]['V_mean']:10.6f}"
                row += " |"
                for b in [2, 4, 8]:
                    row += f" {res[str(b)]['K_per_channel_mean']:10.6f}"
                print(row)
            # Average row
            row = f" {'AVG':>7s} |"
            for b in [2, 4, 8]:
                row += f" {results[str(b)]['K_mean']:10.6f}"
            row += " |"
            for b in [2, 4, 8]:
                row += f" {results[str(b)]['V_mean']:10.6f}"
            row += " |"
            for b in [2, 4, 8]:
                row += f" {results[str(b)]['K_per_channel_mean']:10.6f}"
            print(row)

    elif args.data_source == "infinitebench":
        # ── InfiniteBench mode ────────────────────────────────────
        if not args.task:
            sys.exit("ERROR: --task required for infinitebench mode")
        data_dir = repo_root / "data" / "InfiniteBench"
        print(f"Loading {args.num_samples} InfiniteBench/{args.task} samples "
              f"(max_seq_len={args.max_seq_len}) ...")
        samples = load_infinitebench_samples(
            data_dir, args.task, tokenizer, args.num_samples, args.max_seq_len,
        )
        print(f"  Loaded {len(samples)} samples")

        k_n, v_n, k_pc = run_calibration_on_samples(
            model, samples, bit_widths, chunk_size=args.chunk_size,
        )
        results = aggregate_nmse(k_n, v_n, k_pc, bit_widths)
        print_epsilon_table(results, label=f"infinitebench/{args.task}")

    else:
        # ── LongBench mode (original) ──────────────────────────────
        dataset_script = repo_root / "longbench_dataset.py"
        if not dataset_script.exists():
            sys.exit(f"ERROR: missing {dataset_script}")

        print(f"Loading {args.num_samples} LongBench samples "
              f"(max_seq_len={args.max_seq_len}) ...")
        raw_samples = load_samples(
            dataset_script, tokenizer, args.num_samples, args.max_seq_len,
        )
        print(f"  Loaded {len(raw_samples)} samples")

        samples = [(task, ids) for task, ids in raw_samples]
        k_n, v_n, k_pc = run_calibration_on_samples(
            model, samples, bit_widths, chunk_size=args.chunk_size,
        )
        results = aggregate_nmse(k_n, v_n, k_pc, bit_widths)
        print_epsilon_table(results)

    # ── Save main output ────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output_path}")

    # ── Per-channel K comparison with defaults ──────────────────────
    from knapsack_solver import DEFAULT_EPSILON_K
    print("\n=== Per-Channel K epsilon vs defaults ===")
    print(f" bit | default (old)  | calibrated (new) | rel_diff")
    print("-" * 60)
    for b_str in ["2", "4", "8", "16"]:
        b = int(b_str)
        old_val = DEFAULT_EPSILON_K.get(b, 0.0)
        new_val = results[b_str]["K_per_channel_mean"]
        if old_val > 0:
            rel_diff = abs(new_val - old_val) / old_val
        else:
            rel_diff = 0.0
        print(f"  {b_str:>2} | {old_val:14.6f} | {new_val:16.6f} | {rel_diff:8.2%}")


if __name__ == "__main__":
    main()
