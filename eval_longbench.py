#!/usr/bin/env python3
"""Score LongBench predictions written by run_longbench.py.

run_longbench.py emits per-task JSONL files with fields
``{"task", "example_id", "prediction", "references", "all_classes", "length"}``
under ``--output-dir``. This script walks that directory, looks up each task's
metric in the official LongBench v1 mapping, and prints per-task scores plus
an arithmetic mean.

The metric implementations live in ``external/LongBench/metrics.py`` (a
verbatim copy of the upstream THUDM/LongBench v1 scorer). The dataset →
metric mapping mirrors ``external/LongBench/eval.py`` and matches every
LongBench score reported in the paper.

Usage::

    python eval_longbench.py --pred-dir results/longbench/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the upstream LongBench v1 scorer importable.
LONGBENCH_DIR = Path(__file__).resolve().parent / "external" / "LongBench"
if str(LONGBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(LONGBENCH_DIR))

from metrics import (  # noqa: E402
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)


# Official LongBench v1 dataset → metric mapping for the English tasks shipped
# with this supplementary; mirrors external/LongBench/eval.py.
DATASET2METRIC = {
    "narrativeqa":          qa_f1_score,
    "qasper":               qa_f1_score,
    "multifieldqa_en":      qa_f1_score,
    "hotpotqa":             qa_f1_score,
    "2wikimqa":             qa_f1_score,
    "musique":              qa_f1_score,
    "gov_report":           rouge_score,
    "qmsum":                rouge_score,
    "multi_news":           rouge_score,
    "trec":                 classification_score,
    "triviaqa":             qa_f1_score,
    "samsum":               rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count":        count_score,
    "lcc":                  code_sim_score,
    "repobench-p":          code_sim_score,
}


def score_prediction(task: str, prediction: str, references, all_classes=None):
    """Score one prediction with the official LongBench v1 convention."""
    metric = DATASET2METRIC[task]
    if task in ("trec", "triviaqa", "samsum"):
        prediction = prediction.lstrip("\n").split("\n")[0]
    return 100.0 * max(
        (
            metric(prediction, reference, all_classes=all_classes)
            for reference in references
        ),
        default=0.0,
    )


def score_task(task: str, jsonl_path: Path):
    """Return (score×100, num_predictions) for one task's JSONL."""
    metric = DATASET2METRIC[task]
    preds, refs, all_classes = [], [], None
    with open(jsonl_path) as fh:
        for line in fh:
            row = json.loads(line)
            preds.append(row["prediction"])
            refs.append(row["references"])
            all_classes = row.get("all_classes")
    total = 0.0
    for pred, gts in zip(preds, refs):
        total += score_prediction(task, pred, gts, all_classes)
    return round(total / max(len(preds), 1), 2), len(preds)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score LongBench predictions.")
    ap.add_argument(
        "--pred-dir", required=True,
        help="Directory containing per-task subfolders, each with a "
             "<task>.jsonl produced by run_longbench.py.",
    )
    args = ap.parse_args()

    root = Path(args.pred_dir)
    if not root.is_dir():
        sys.stderr.write(f"--pred-dir not found: {root}\n")
        return 2

    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        task = task_dir.name
        if task not in DATASET2METRIC:
            continue
        jsonl = task_dir / f"{task}.jsonl"
        if not jsonl.exists():
            continue
        score, n = score_task(task, jsonl)
        rows.append((task, score, n))

    if not rows:
        sys.stderr.write("no scoreable per-task JSONL files under pred-dir\n")
        return 1

    width = max(len(r[0]) for r in rows)
    print(f"{'task':<{width}}  {'score':>8}  {'n':>5}")
    print("-" * (width + 18))
    for task, score, n in rows:
        print(f"{task:<{width}}  {score:>8.2f}  {n:>5}")
    print("-" * (width + 18))
    mean = sum(s for _, s, _ in rows) / len(rows)
    print(f"mean ({len(rows)} tasks): {mean:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
