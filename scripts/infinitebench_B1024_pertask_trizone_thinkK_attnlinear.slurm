#!/bin/bash
#SBATCH --job-name=ib_B1024_pertask_tz_kthink_attnlin
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS_LONG>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=<SCRATCH>/slurm_logs/rdkv/infinitebench/ib-B1024-pertask-kthink-attnlin-%A_%a.out
#SBATCH --error=<SCRATCH>/slurm_logs/rdkv/infinitebench/ib-B1024-pertask-kthink-attnlin-%A_%a.err

set -euo pipefail
mkdir -p <SCRATCH>/slurm_logs/rdkv/infinitebench

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
SCORE_POOLING=5
K_RATIO=0.5

TASKS=(
  passkey number_string kv_retrieval longdialogue_qa_eng longbook_sum_eng
  longbook_choice_eng longbook_qa_eng longbook_qa_chn math_find code_debug
)

ARRAY_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TASK_NAME="${TASKS[${ARRAY_ID}]}"

RESULTS_ROOT="${SCR}/rdkv_results/rdkv/infinitebench"
DATA_DIR="${REPO_ROOT}/data/InfiniteBench"

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
export OBKV_DISABLE_CUDA_GRAPH=1
export OBKV_K_AVG_MODE=perhead

cd "${OBKV_ROOT}"

OUTPUT_DIR="${RESULTS_ROOT}/obkv_khead_TriZone/budget_${TOKEN_BUDGET}_streaming_joint_eps_pool5_kavg_perhead_thinkK_attnlinear/${TASK_NAME}"
mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/run.log"
TRITON_DIR="/dev/shm/${USER}/triton_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TRITON_DIR}"
export TRITON_CACHE_DIR="${TRITON_DIR}"

echo "[$(date)] array=${ARRAY_ID} task=${TASK_NAME} budget=${TOKEN_BUDGET} pool=${SCORE_POOLING} k=${K_RATIO} v_score=attn_linear k_score=think"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

"${RUNVENV_PY}" run_infinitebench.py \
  --method obkv \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --task "${TASK_NAME}" \
  --token-budget "${TOKEN_BUDGET}" \
  --score-pooling "${SCORE_POOLING}" \
  --k-budget-ratio "${K_RATIO}" \
  --v-score-type attn_linear \
  --k-score-type think \
  --streaming \
  --eviction-mode joint \
  --epsilon-path "${EPSILON_PATH}" \
  --attn-implementation flash_attention_2 \
  --device-map single \
  --seed 42 \
  2>&1 | tee "${LOGFILE}"

echo "[$(date)] array=${ARRAY_ID} task=${TASK_NAME} done"
