#!/usr/bin/env python3
"""
Score RULER benchmark predictions and produce summary tables.

Reads per-task JSONL files from results directories, computes:
  - Per-task per-length accuracy
  - Category averages (Retrieval, Multi-hop, Aggregation, QA)
  - 11-task avg (matches ReST-KV Table 3, excludes QA)
  - 13-task avg (full)
  - Per-length averages

Usage:
  python eval_ruler.py \
      --result-root results/ruler \
      --methods fullkv,snapkv,ada_snapkv,obkv \
      --seq-lengths 4096,8192,16384,32768,65536,131072
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ruler_dataset import (
    RULER_11_TASKS,
    RULER_CATEGORIES,
    RULER_SEQ_LENGTHS,
    RULER_SHARD_SPLIT,
    RULER_TASKS,
    STRING_MATCH_PART_TASKS,
    evaluate_sample,
)


def parse_args():
    p = argparse.ArgumentParser(description="Score RULER benchmark results.")
    p.add_argument("--result-root", required=True,
                    help="Root directory containing method subdirectories")
    p.add_argument("--methods", required=True,
                    help="Comma-separated list of methods to score")
    p.add_argument("--seq-lengths", default=None,
                    help="Comma-separated seq lengths (default: all 6 standard lengths)")
    p.add_argument("--output-dir", default=None,
                    help="Output dir for summary files (default: result-root)")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--check-completeness", action="store_true",
                    help="Report missing task/length/shard combinations")
    return p.parse_args()


def load_predictions(result_root: Path, method: str, seq_length: int,
                     num_shards: int) -> Dict[str, List[Dict]]:
    """Load all prediction JSONL files for a method + seq_length across shards."""
    preds_by_task: Dict[str, List[Dict]] = defaultdict(list)

    for shard_id in range(num_shards):
        shard_dir = result_root / method / str(seq_length) / f"shard_{shard_id}"
        if not shard_dir.exists():
            continue
        for jsonl_path in shard_dir.glob("*.jsonl"):
            task_name = jsonl_path.stem
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    preds_by_task[task_name].append(row)

    return dict(preds_by_task)


def score_predictions(preds_by_task: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Compute per-task accuracy from loaded predictions."""
    task_acc = {}
    for task_name, preds in preds_by_task.items():
        if not preds:
            continue
        correct = sum(
            evaluate_sample(task_name, p.get("prediction", ""), p.get("outputs", []))
            for p in preds
        )
        task_acc[task_name] = correct / len(preds)
    return task_acc


def compute_averages(task_acc: Dict[str, float]) -> Dict[str, float]:
    """Compute various average metrics from per-task accuracies."""
    avgs = {}

    # 13-task avg
    all_scores = [task_acc[t] for t in RULER_TASKS if t in task_acc]
    if all_scores:
        avgs["avg_13"] = sum(all_scores) / len(all_scores)

    # 11-task avg (excludes QA, matches ReST-KV)
    scores_11 = [task_acc[t] for t in RULER_11_TASKS if t in task_acc]
    if scores_11:
        avgs["avg_11"] = sum(scores_11) / len(scores_11)

    # Category averages
    for cat_name, cat_tasks in RULER_CATEGORIES.items():
        cat_scores = [task_acc[t] for t in cat_tasks if t in task_acc]
        if cat_scores:
            avgs[f"cat_{cat_name}"] = sum(cat_scores) / len(cat_scores)

    return avgs


def check_completeness(result_root: Path, methods: List[str],
                       seq_lengths: List[int], num_shards: int):
    """Report missing combinations."""
    missing = []
    for method in methods:
        for seq_len in seq_lengths:
            for shard_id in range(num_shards):
                shard_dir = result_root / method / str(seq_len) / f"shard_{shard_id}"
                tasks_in_shard = RULER_SHARD_SPLIT.get(shard_id, [])
                for task_name in tasks_in_shard:
                    jsonl_path = shard_dir / f"{task_name}.jsonl"
                    if not jsonl_path.exists():
                        missing.append((method, seq_len, shard_id, task_name))

    if missing:
        print(f"\n[MISSING] {len(missing)} combinations missing:")
        for m, sl, si, tn in missing:
            print(f"  {m} / {sl} / shard_{si} / {tn}")
    else:
        print("\n[OK] All combinations present.")
    return missing


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    methods = [m.strip() for m in args.methods.split(",")]
    seq_lengths = (
        [int(x) for x in args.seq_lengths.split(",")]
        if args.seq_lengths
        else RULER_SEQ_LENGTHS
    )
    output_dir = Path(args.output_dir) if args.output_dir else result_root

    if args.check_completeness:
        check_completeness(result_root, methods, seq_lengths, args.num_shards)

    # ── Collect all scores ───────────────────────────────────────────
    # scores[method][seq_length][task_name] = accuracy
    all_scores: Dict[str, Dict[int, Dict[str, float]]] = {}
    # sample_counts[method][seq_length][task_name] = n
    all_counts: Dict[str, Dict[int, Dict[str, int]]] = {}

    for method in methods:
        all_scores[method] = {}
        all_counts[method] = {}
        for seq_len in seq_lengths:
            preds = load_predictions(result_root, method, seq_len, args.num_shards)
            task_acc = score_predictions(preds)
            all_scores[method][seq_len] = task_acc
            all_counts[method][seq_len] = {t: len(p) for t, p in preds.items()}

    # ── Print per-method tables ──────────────────────────────────────
    len_labels = [f"{sl//1024}K" for sl in seq_lengths]

    for method in methods:
        print(f"\n{'='*80}")
        print(f"  Method: {method}")
        print(f"{'='*80}")

        # Header
        header = f"{'Task':25s}" + "".join(f"{ll:>8s}" for ll in len_labels) + f"{'AVG':>8s}"
        print(header)
        print("-" * len(header))

        task_avgs = {}
        for task_name in RULER_TASKS:
            row = f"{task_name:25s}"
            scores_for_task = []
            for seq_len in seq_lengths:
                acc = all_scores[method].get(seq_len, {}).get(task_name)
                if acc is not None:
                    row += f"{acc*100:8.1f}"
                    scores_for_task.append(acc)
                else:
                    row += f"{'—':>8s}"
            if scores_for_task:
                avg = sum(scores_for_task) / len(scores_for_task)
                row += f"{avg*100:8.1f}"
                task_avgs[task_name] = avg
            else:
                row += f"{'—':>8s}"
            print(row)

        # Category averages
        print("-" * len(header))
        for cat_name, cat_tasks in RULER_CATEGORIES.items():
            row = f"{cat_name:25s}"
            for seq_len in seq_lengths:
                cat_scores = [
                    all_scores[method].get(seq_len, {}).get(t)
                    for t in cat_tasks
                ]
                cat_scores = [s for s in cat_scores if s is not None]
                if cat_scores:
                    row += f"{sum(cat_scores)/len(cat_scores)*100:8.1f}"
                else:
                    row += f"{'—':>8s}"
            cat_task_avgs = [task_avgs[t] for t in cat_tasks if t in task_avgs]
            if cat_task_avgs:
                row += f"{sum(cat_task_avgs)/len(cat_task_avgs)*100:8.1f}"
            else:
                row += f"{'—':>8s}"
            print(row)

        # Overall averages
        print("-" * len(header))
        # 13-task avg
        row_13 = f"{'Avg (13 tasks)':25s}"
        for seq_len in seq_lengths:
            scores_all = [
                all_scores[method].get(seq_len, {}).get(t)
                for t in RULER_TASKS
            ]
            scores_all = [s for s in scores_all if s is not None]
            if scores_all:
                row_13 += f"{sum(scores_all)/len(scores_all)*100:8.1f}"
            else:
                row_13 += f"{'—':>8s}"
        all_task_avgs_13 = [task_avgs[t] for t in RULER_TASKS if t in task_avgs]
        if all_task_avgs_13:
            row_13 += f"{sum(all_task_avgs_13)/len(all_task_avgs_13)*100:8.1f}"
        else:
            row_13 += f"{'—':>8s}"
        print(row_13)

        # 11-task avg (no QA)
        row_11 = f"{'Avg (11 tasks, no QA)':25s}"
        for seq_len in seq_lengths:
            scores_11 = [
                all_scores[method].get(seq_len, {}).get(t)
                for t in RULER_11_TASKS
            ]
            scores_11 = [s for s in scores_11 if s is not None]
            if scores_11:
                row_11 += f"{sum(scores_11)/len(scores_11)*100:8.1f}"
            else:
                row_11 += f"{'—':>8s}"
        all_task_avgs_11 = [task_avgs[t] for t in RULER_11_TASKS if t in task_avgs]
        if all_task_avgs_11:
            row_11 += f"{sum(all_task_avgs_11)/len(all_task_avgs_11)*100:8.1f}"
        else:
            row_11 += f"{'—':>8s}"
        print(row_11)

    # ── Cross-method comparison table ────────────────────────────────
    print(f"\n{'='*80}")
    print("  Cross-Method Comparison (13-task avg)")
    print(f"{'='*80}")
    header = f"{'Method':20s}" + "".join(f"{ll:>8s}" for ll in len_labels) + f"{'AVG':>8s}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:20s}"
        per_len_avgs = []
        for seq_len in seq_lengths:
            scores_all = [
                all_scores[method].get(seq_len, {}).get(t)
                for t in RULER_TASKS
            ]
            scores_all = [s for s in scores_all if s is not None]
            if scores_all:
                avg = sum(scores_all) / len(scores_all)
                row += f"{avg*100:8.1f}"
                per_len_avgs.append(avg)
            else:
                row += f"{'—':>8s}"
        if per_len_avgs:
            row += f"{sum(per_len_avgs)/len(per_len_avgs)*100:8.1f}"
        else:
            row += f"{'—':>8s}"
        print(row)

    # ── Save JSON + CSV ──────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON: full results
    json_out = {}
    for method in methods:
        json_out[method] = {}
        for seq_len in seq_lengths:
            json_out[method][str(seq_len)] = {
                "task_accuracies": {
                    k: round(v, 4)
                    for k, v in all_scores[method].get(seq_len, {}).items()
                },
                "sample_counts": all_counts[method].get(seq_len, {}),
                "averages": compute_averages(all_scores[method].get(seq_len, {})),
            }

    json_path = output_dir / "ruler_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")

    # CSV: flat table
    csv_path = output_dir / "ruler_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "task", "seq_length", "accuracy", "n_samples"])
        for method in methods:
            for seq_len in seq_lengths:
                for task_name in RULER_TASKS:
                    acc = all_scores[method].get(seq_len, {}).get(task_name)
                    n = all_counts[method].get(seq_len, {}).get(task_name, 0)
                    writer.writerow([
                        method, task_name, seq_len,
                        f"{acc:.4f}" if acc is not None else "",
                        n,
                    ])
    print(f"[Saved] {csv_path}")

    # Markdown summary
    md_path = output_dir / "ruler_summary.md"
    with open(md_path, "w") as f:
        f.write("# RULER Benchmark Results\n\n")
        f.write("## Cross-Method Comparison (13-task avg)\n\n")
        f.write("| Method |" + "|".join(f" {ll} " for ll in len_labels) + "| AVG |\n")
        f.write("|--------|" + "|".join("------" for _ in len_labels) + "|-----|\n")
        for method in methods:
            row = f"| {method} |"
            per_len = []
            for seq_len in seq_lengths:
                scores_all = [
                    all_scores[method].get(seq_len, {}).get(t)
                    for t in RULER_TASKS
                ]
                scores_all = [s for s in scores_all if s is not None]
                if scores_all:
                    avg = sum(scores_all) / len(scores_all)
                    row += f" {avg*100:.1f} |"
                    per_len.append(avg)
                else:
                    row += " — |"
            if per_len:
                row += f" {sum(per_len)/len(per_len)*100:.1f} |"
            else:
                row += " — |"
            f.write(row + "\n")
        f.write("\n")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
