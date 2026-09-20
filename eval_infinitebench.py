#!/usr/bin/env python3
"""
Score InfiniteBench predictions and produce summary tables.

Reads per-task JSONL files from prediction directories, computes per-task
scores using the official InfiniteBench scoring functions (from ReST-KV),
and outputs summary tables.

Usage:
  python eval_infinitebench.py \
      --pred-dir results/infinitebench/obkv \
      --methods fullkv,ada_snapkv,obkv

  # Single method:
  python eval_infinitebench.py --pred-dir results/infinitebench/fullkv
"""

import argparse
import csv
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import evaluate

ROUGE_SCORER = evaluate.load("rouge")


# ── InfiniteBench task list (10 tasks, no math_calc / code_run) ──────────

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


# ── Scoring functions (from ReST-KV eval_infinite.py) ───────────────────

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    f1 = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1 = scores[0]
        f1 = max(f1, this_f1)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list) -> float:
    f1 = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1 = scores[0]
        f1 = max(f1, this_f1)
    return f1


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def get_score_one_kv_retrieval(pred, label) -> bool:
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}", "</s>", "The", "To"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if isinstance(label, list):
        label = label[0]
    return label in words


def get_score_one_passkey(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_debug(pred, label) -> bool:
    pred = pred.strip()
    label_c = label[1]
    fn_name = label[0]
    if pred[:2] in [f"{label_c}.", f"{label_c}:", f"{label_c} ", f"{label_c}\n"]:
        return True
    if pred[:0] == f"{label_c}":
        return True

    ans_prefixes = [
        "answer is:",
        "is:",
        "answer:",
    ]
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1:]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    if isinstance(label, int):
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(float(first_num)) == label
    elif isinstance(label, float):
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred, label) -> bool:
    label = label[0]
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    words = [x.upper() for x in words]
    return label in words


def get_score_one_longbook_choice_eng(pred, label) -> bool:
    pred = pred.strip()
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1:]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one_longbook_qa_eng(pred, label) -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(pred: str, label: str) -> float:
    score = ROUGE_SCORER.compute(
        predictions=[pred], references=[label], use_aggregator=False
    )
    return score["rougeLsum"][0]


def get_score_one_longbook_qa_chn(pred, label) -> float:
    return qa_f1_score_zh(pred, label)


NAME_TO_SCORE_GETTER = {
    "kv_retrieval": get_score_one_kv_retrieval,
    "passkey": get_score_one_passkey,
    "number_string": get_score_one_number_string,
    "code_debug": get_score_one_code_debug,
    "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
    "longbook_qa_eng": get_score_one_longbook_qa_eng,
    "longbook_sum_eng": get_score_one_longbook_sum_eng,
    "longbook_choice_eng": get_score_one_longbook_choice_eng,
    "longbook_qa_chn": get_score_one_longbook_qa_chn,
    "math_find": get_score_one_math_find,
}


def get_score_one(pred: str, label, task_name: str) -> float:
    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label)
    return float(score)


# ── Loading & scoring ────────────────────────────────────────────────────

def load_predictions(pred_dir: Path) -> Dict[str, List[dict]]:
    """Load all per-task JSONL prediction files from a directory."""
    preds_by_task = {}
    for task in INFINITEBENCH_TASKS:
        jsonl_path = pred_dir / f"{task}.jsonl"
        if not jsonl_path.exists():
            continue
        preds = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
        if preds:
            preds_by_task[task] = preds
    return preds_by_task


def score_task(task_name: str, predictions: List[dict]) -> float:
    """Compute average score for a task."""
    scores = []
    for p in predictions:
        pred = p.get("prediction", "")
        label = p.get("ground_truth", "")
        s = get_score_one(pred, label, task_name)
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


# ── CLI & Main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Score InfiniteBench predictions."
    )
    p.add_argument("--pred-dir", required=True,
                    help="Directory containing prediction JSONL files, "
                         "or parent dir with method subdirectories")
    p.add_argument("--methods", default=None,
                    help="Comma-separated method names (subdirs of pred-dir). "
                         "If omitted, score pred-dir directly")
    p.add_argument("--output-dir", default=None,
                    help="Where to write summary files (defaults to pred-dir)")
    return p.parse_args()


def score_one_method(pred_dir: Path, method_name: str) -> Dict[str, float]:
    """Score a single method's predictions. Returns {task: score}."""
    preds_by_task = load_predictions(pred_dir)
    if not preds_by_task:
        print(f"  [WARN] No predictions found in {pred_dir}")
        return {}

    task_scores = {}
    for task in INFINITEBENCH_TASKS:
        if task not in preds_by_task:
            continue
        score = score_task(task, preds_by_task[task])
        n = len(preds_by_task[task])
        task_scores[task] = score
        print(f"  {task:25s}  {score*100:6.2f}%  (n={n})")

    if task_scores:
        avg = sum(task_scores.values()) / len(task_scores)
        print(f"  {'Avg (' + str(len(task_scores)) + ' tasks)':25s}  {avg*100:6.2f}%")
    return task_scores


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir

    if args.methods:
        method_list = [m.strip() for m in args.methods.split(",")]
    else:
        method_list = [None]  # score pred_dir directly

    all_results = {}

    for method in method_list:
        if method:
            mdir = pred_dir / method
            print(f"\n=== {method} ===")
        else:
            mdir = pred_dir
            method = pred_dir.name
            print(f"\n=== {method} ===")

        task_scores = score_one_method(mdir, method)
        if task_scores:
            all_results[method] = task_scores

    if not all_results:
        print("No results to summarize.")
        return

    # ── Write summary files ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    summary = {}
    for method, scores in all_results.items():
        avg = sum(scores.values()) / len(scores) if scores else 0
        summary[method] = {
            "per_task": {t: round(s * 100, 2) for t, s in scores.items()},
            "avg": round(avg * 100, 2),
            "num_tasks": len(scores),
        }
    json_path = output_dir / "infinitebench_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    # CSV
    csv_path = output_dir / "infinitebench_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        tasks_present = sorted(
            set(t for scores in all_results.values() for t in scores)
        )
        writer.writerow(["method"] + tasks_present + ["avg"])
        for method, scores in all_results.items():
            avg = sum(scores.values()) / len(scores) if scores else 0
            row = [method]
            for t in tasks_present:
                row.append(f"{scores.get(t, 0)*100:.2f}" if t in scores else "-")
            row.append(f"{avg*100:.2f}")
            writer.writerow(row)
    print(f"CSV:  {csv_path}")

    # Markdown
    md_path = output_dir / "infinitebench_summary.md"
    with open(md_path, "w") as f:
        tasks_present = sorted(
            set(t for scores in all_results.values() for t in scores)
        )
        # Header
        f.write("# InfiniteBench Results\n\n")
        header = "| Method |"
        sep = "|--------|"
        for t in tasks_present:
            short = t.replace("longbook_", "lb_").replace("longdialogue_", "ld_")
            header += f" {short} |"
            sep += "------:|"
        header += " Avg |"
        sep += "----:|"
        f.write(header + "\n")
        f.write(sep + "\n")

        for method, scores in all_results.items():
            avg = sum(scores.values()) / len(scores) if scores else 0
            row = f"| {method} |"
            for t in tasks_present:
                if t in scores:
                    row += f" {scores[t]*100:.1f} |"
                else:
                    row += " - |"
            row += f" {avg*100:.1f} |"
            f.write(row + "\n")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
