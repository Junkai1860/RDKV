#!/usr/bin/env python3
"""RULER benchmark dataset constants, data loader, and metric functions."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# ── 13 RULER tasks ──────────────────────────────────────────────────────────
RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]

# ── Fixed 4-shard split ────────────────────────────────────────────────────
RULER_SHARD_SPLIT = {
    0: ["niah_single_1", "niah_single_2", "niah_single_3"],
    1: ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"],
    2: ["niah_multiquery", "niah_multivalue", "qa_1", "qa_2"],
    3: ["vt", "cwe", "fwe"],
}

# ── Category groupings (for reporting) ─────────────────────────────────────
RULER_CATEGORIES = {
    "Retrieval": [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multiquery", "niah_multivalue",
    ],
    "Multi-hop": ["vt"],
    "Aggregation": ["cwe", "fwe"],
    "QA": ["qa_1", "qa_2"],
}

# ── Metric assignment ──────────────────────────────────────────────────────
# string_match_all: ALL references must appear in prediction (most tasks)
# string_match_part: ANY reference appears in prediction (QA tasks only)
STRING_MATCH_PART_TASKS = {"qa_1", "qa_2"}

# ── Sequence lengths ───────────────────────────────────────────────────────
RULER_SEQ_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

# ── Max generation tokens per task (from RULER synthetic/constants.py) ─────
RULER_TOKENS_TO_GENERATE = {
    "niah_single_1": 128, "niah_single_2": 128, "niah_single_3": 128,
    "niah_multikey_1": 128, "niah_multikey_2": 128, "niah_multikey_3": 128,
    "niah_multiquery": 128, "niah_multivalue": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32, "qa_2": 32,
}
RULER_MAX_TOKENS_TO_GENERATE = max(RULER_TOKENS_TO_GENERATE.values())  # 128

# 11-task subset (matches ReST-KV Table 3, excludes QA)
RULER_11_TASKS = [t for t in RULER_TASKS if t not in STRING_MATCH_PART_TASKS]


# ── Metric functions ───────────────────────────────────────────────────────

def string_match_all(prediction: str, references: List[str]) -> float:
    """Fraction of references that appear in the prediction (case-insensitive).

    Matches official RULER metric: partial credit based on how many references
    are found. E.g. 3/4 refs found → 0.75.
    """
    if not references:
        return 1.0
    prediction_lower = prediction.lower()
    found = sum(1.0 for ref in references if ref.lower() in prediction_lower)
    return found / len(references)


def string_match_part(prediction: str, references: List[str]) -> float:
    """Returns 1.0 if ANY reference appears in the prediction (case-insensitive)."""
    prediction_lower = prediction.lower()
    for ref in references:
        if ref.lower() in prediction_lower:
            return 1.0
    return 0.0


def evaluate_sample(task_name: str, prediction: str, references: List[str]) -> float:
    """Evaluate a single prediction against references using the task-appropriate metric."""
    if task_name in STRING_MATCH_PART_TASKS:
        return string_match_part(prediction, references)
    return string_match_all(prediction, references)


def evaluate_task(task_name: str, predictions: List[Dict]) -> float:
    """Compute accuracy for a task given list of dicts with 'prediction' and 'outputs' keys.

    Returns accuracy as a float in [0, 1].
    """
    if not predictions:
        return 0.0
    correct = sum(
        evaluate_sample(task_name, p["prediction"], p["outputs"])
        for p in predictions
    )
    return correct / len(predictions)


# ── Data loading ───────────────────────────────────────────────────────────

def load_ruler_task(data_dir: str, seq_length: int, task_name: str) -> List[Dict]:
    """Load RULER JSONL data for a single task at a given sequence length.

    Checks two path formats in order:
      1. RULER native: {data_dir}/{seq_length}/{task_name}/validation.jsonl
      2. Flat:         {data_dir}/{seq_length}/{task_name}.jsonl

    Each line is JSON with at least 'input' and 'outputs' fields.

    Returns list of dicts, each with at least:
      - 'input': str (the full prompt text)
      - 'outputs': List[str] (reference answers)
      - 'index': int (0-based sample index)
      - 'task': str (task name)
    """
    # Try RULER native path first, then flat path
    candidates = [
        Path(data_dir) / str(seq_length) / task_name / "validation.jsonl",
        Path(data_dir) / str(seq_length) / f"{task_name}.jsonl",
    ]
    jsonl_path = None
    for c in candidates:
        if c.exists():
            jsonl_path = c
            break
    if jsonl_path is None:
        raise FileNotFoundError(
            f"RULER data not found for task={task_name} seq_length={seq_length}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    samples = []
    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["index"] = idx
            row["task"] = task_name
            samples.append(row)
    return samples


def load_ruler_shard(
    data_dir: str,
    seq_length: int,
    shard_id: int,
    max_examples: int = None,
) -> List[Tuple[str, Dict]]:
    """Load all RULER examples for a given shard and sequence length.

    Returns list of (task_name, sample_dict) tuples.
    """
    tasks = RULER_SHARD_SPLIT[shard_id]
    work_items = []
    for task_name in tasks:
        samples = load_ruler_task(data_dir, seq_length, task_name)
        if max_examples is not None:
            samples = samples[:max_examples]
        for sample in samples:
            work_items.append((task_name, sample))
    return work_items


def load_ruler_tasks(
    data_dir: str,
    seq_length: int,
    task_filter: List[str] = None,
    max_examples: int = None,
) -> List[Tuple[str, Dict]]:
    """Load RULER examples for specified tasks (or all tasks if task_filter is None).

    Returns list of (task_name, sample_dict) tuples.
    """
    tasks = task_filter if task_filter else RULER_TASKS
    work_items = []
    for task_name in tasks:
        samples = load_ruler_task(data_dir, seq_length, task_name)
        if max_examples is not None:
            samples = samples[:max_examples]
        for sample in samples:
            work_items.append((task_name, sample))
    return work_items
