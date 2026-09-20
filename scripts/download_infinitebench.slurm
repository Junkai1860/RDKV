#!/bin/bash
#SBATCH --job-name=dl-infinitebench
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=<REPO_ROOT>/obkv_head/slurm_logs/infinitebench/dl-infinitebench-%j.out
#SBATCH --error=<REPO_ROOT>/obkv_head/slurm_logs/infinitebench/dl-infinitebench-%j.err

set -euo pipefail

OUT_DIR=<REPO_ROOT>/data/InfiniteBench
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

BASE_URL="https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main"

FILES=(
  code_debug.jsonl
  code_run.jsonl
  kv_retrieval.jsonl
  longbook_choice_eng.jsonl
  longbook_qa_chn.jsonl
  longbook_qa_eng.jsonl
  longbook_sum_eng.jsonl
  longdialogue_qa_eng.jsonl
  math_calc.jsonl
  math_find.jsonl
  number_string.jsonl
  passkey.jsonl
)

echo "[$(date)] downloading ${#FILES[@]} InfiniteBench jsonl files -> ${OUT_DIR}"
for f in "${FILES[@]}"; do
  echo "-> ${f}"
  curl -L --fail --retry 5 --retry-delay 10 -C - \
    "${BASE_URL}/${f}" -o "${f}"
done

echo "[$(date)] done; total size:"
du -sh "${OUT_DIR}"
ls -la "${OUT_DIR}"
