#!/bin/bash
#SBATCH --job-name=lb_B1024_pertask_tz_kthink_attnlin
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS_LONG>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
#SBATCH --output=<SCRATCH>/slurm_logs/rdkv/longbench/lb-B1024-pertask-kthink-attnlin-%A_%a.out
#SBATCH --error=<SCRATCH>/slurm_logs/rdkv/longbench/lb-B1024-pertask-kthink-attnlin-%A_%a.err

set -euo pipefail
mkdir -p <SCRATCH>/slurm_logs/rdkv/longbench

REPO_ROOT="<REPO_ROOT>"
OBKV_ROOT="${REPO_ROOT}/rdkv"
SCR="<SCRATCH>"

module load gcc/12.2.0
source "${SCR}/venvs/<VENV>/bin/activate"
RUNVENV_PY="$(command -v python3)"
# Pick model + matching per-layer epsilon calibration. Pre-shipped pairs:
#   meta-llama/Llama-3.1-8B-Instruct  -> epsilon_calibration_llama31_8b.json (32 layers)
#   mistralai/Mistral-7B-Instruct-v0.2 -> epsilon_calibration_mistral_7b.json (32 layers)
#   meta-llama/Llama-2-13b-chat-hf    -> epsilon_calibration_llama2_13b.json (40 layers)
#   Qwen/Qwen3-4B-Instruct            -> epsilon_calibration_qwen3_4b.json   (36 layers)
#   Qwen/Qwen2.5-72B-Instruct         -> epsilon_calibration_qwen25_72b.json (80 layers)
# For an unsupported model, run calibrate_epsilon.py on it first.
MODEL_PATH="${SCR}/hf_models/Llama-3.1-8B-Instruct"
EPSILON_PATH="${OBKV_ROOT}/results/epsilon_calibration_llama31_8b.json"

TOKEN_BUDGET=1024
POOL_KERNEL_SIZE=5
K_RATIO=0.5
K_SCORE_TYPE=think
NUM_SHARDS=4

# Flat array of 16 tasks. Order MUST match longbench_manifest/test_num_shards_4_shard_*.json:
#   shard 0 -> indices 0..3, shard 1 -> 4..7, shard 2 -> 8..11, shard 3 -> 12..15.
TASKS=( \
  narrativeqa 2wikimqa multi_news passage_count \
  qasper      musique  trec       passage_retrieval_en \
  multifieldqa_en gov_report triviaqa lcc \
  hotpotqa    qmsum    samsum     repobench-p \
)

ARRAY_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SHARD_ID=$(( ARRAY_ID / 4 ))
TASK_NAME="${TASKS[${ARRAY_ID}]}"

RESULTS_ROOT="${SCR}/rdkv_results/rdkv/longbench"
DATA_ARCHIVE="${SCR}/hf_datasets/LongBench/data.zip"

export LONGBENCH_REPO="${RESULTS_ROOT}"
export LONGBENCH_DATA_ARCHIVE="${DATA_ARCHIVE}"
export HF_HOME="${SCR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${SCR}/tmp"
export TORCH_EXTENSIONS_DIR="${SCR}/torch_extensions"
export PYTHONPATH="${OBKV_ROOT}:${PYTHONPATH:-}"
export CPATH="${SCR}/venvs/<VENV_WITH_HEADERS>/include/python3.11:${CPATH:-}"
# TriZone P0/P1 eager-only — Phase 3d CG recapture not yet ported.
export OBKV_DISABLE_CUDA_GRAPH=1
# Per-head K bit budget (k_avg_h independent per head).
export OBKV_K_AVG_MODE=perhead

cd "${OBKV_ROOT}"

OUTPUT_DIR="${RESULTS_ROOT}/obkv_khead_TriZone/budget_${TOKEN_BUDGET}_per_task_streaming_joint_eps_pool5_kavg_perhead_thinkK_attnlinear/${TASK_NAME}"
mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/run.log"
TRITON_DIR="${SCR}/triton_cache_clean/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TRITON_DIR}"
export TRITON_CACHE_DIR="${TRITON_DIR}"

echo "[$(date)] array=${ARRAY_ID} shard=${SHARD_ID}/${NUM_SHARDS} task=${TASK_NAME} budget=${TOKEN_BUDGET} pool=${POOL_KERNEL_SIZE} k=${K_RATIO} kavg_mode=${OBKV_K_AVG_MODE} v_score=attn_linear(default) k_score=${K_SCORE_TYPE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

"${RUNVENV_PY}" run_longbench.py \
  --model-path "${MODEL_PATH}" \
  --repo-root "${REPO_ROOT}" \
  --attn-implementation flash_attention_2 \
  --device-map single \
  --num-shards "${NUM_SHARDS}" \
  --shard "${SHARD_ID}" \
  --task-filter "${TASK_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --token-budget "${TOKEN_BUDGET}" \
  --pool-kernel-size "${POOL_KERNEL_SIZE}" \
  --k-budget-ratio "${K_RATIO}" \
  --k-score-type "${K_SCORE_TYPE}" \
  --streaming \
  --eviction-mode joint \
  --epsilon-path "${EPSILON_PATH}" \
  2>&1 | tee "${LOGFILE}"

echo "[$(date)] array=${ARRAY_ID} task=${TASK_NAME} done"
