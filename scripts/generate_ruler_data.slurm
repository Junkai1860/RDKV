#!/bin/bash
#SBATCH --job-name=gen-ruler
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-65
#SBATCH --output=<REPO_ROOT>/obkv_head/slurm_logs/ruler-gen/gen-ruler-%A_%a.out
#SBATCH --error=<REPO_ROOT>/obkv_head/slurm_logs/ruler-gen/gen-ruler-%A_%a.err

set -euo pipefail

REPO_ROOT="<REPO_ROOT>"
OBKV_VENV="${REPO_ROOT}/obkv_head/obkv"
RULER_REPO="${REPO_ROOT}/external/RULER"
OUT_ROOT="${REPO_ROOT}/ruler_data/llama31_8b_instruct"
TOKENIZER_PATH="${REPO_ROOT}/hf_cache/Llama-3.1-8B-Instruct"

TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multiquery niah_multivalue
  vt cwe fwe
)
SEQ_LENGTHS=(4096 8192 16384 32768 65536 131072)

NUM_TASKS=${#TASKS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TASK_INDEX=$(( TASK_ID % NUM_TASKS ))
SEQ_INDEX=$(( TASK_ID / NUM_TASKS ))
TASK_NAME="${TASKS[${TASK_INDEX}]}"
SEQ_LEN="${SEQ_LENGTHS[${SEQ_INDEX}]}"

SAVE_DIR="${OUT_ROOT}/${SEQ_LEN}"
mkdir -p "${SAVE_DIR}"

source "${OBKV_VENV}/bin/activate"

echo "[$(date)] task=${TASK_NAME} seq=${SEQ_LEN} array_id=${TASK_ID}"
echo "[$(date)] save_dir=${SAVE_DIR}"

cd "${RULER_REPO}/scripts/data"

python prepare.py \
  --save_dir "${SAVE_DIR}" \
  --benchmark synthetic \
  --task "${TASK_NAME}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --tokenizer_type hf \
  --max_seq_length "${SEQ_LEN}" \
  --num_samples 500 \
  --random_seed 42 \
  --model_template_type meta-llama3

echo "[$(date)] done task=${TASK_NAME} seq=${SEQ_LEN}"
ls -la "${SAVE_DIR}/${TASK_NAME}/" || true
