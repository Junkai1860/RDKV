#!/bin/bash
#SBATCH --job-name=rdkv_niah_stream_kthink_attnlin
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS_LONG>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=<SCRATCH>/slurm_logs/rdkv/niah/niah-stream-kthink-attnlin-%A_%a.out
#SBATCH --error=<SCRATCH>/slurm_logs/rdkv/niah/niah-stream-kthink-attnlin-%A_%a.err

set -euo pipefail
mkdir -p <SCRATCH>/slurm_logs/rdkv/niah

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
HAYSTACK_DIR="<HAYSTACK_DIR>"

NUM_SHARDS=4
POOL_KERNEL_SIZE=5
K_RATIO=0.5
OBS_WINDOW=32
CHUNK_SIZE=128

# array 0..3 -> B=64,  shard 0..3
# array 4..7 -> B=128, shard 0..3
ARRAY_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ "${ARRAY_ID}" -lt 4 ]]; then
  TOKEN_BUDGET=64
  SHARD_ID="${ARRAY_ID}"
else
  TOKEN_BUDGET=128
  SHARD_ID=$(( ARRAY_ID - 4 ))
fi

METHOD_VERSION="rdkv_niah_stream_attnlin_thinkK_B${TOKEN_BUDGET}_kr0.5_pool5"
RESULTS_ROOT="${SCR}/rdkv_results/rdkv/niah/streaming_attnlin_thinkK"
OUTPUT_BASE="${RESULTS_ROOT}/results"
LOG_DIR="${RESULTS_ROOT}/logs/${METHOD_VERSION}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"
LOGFILE="${LOG_DIR}/shard${SHARD_ID}.log"

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
# Triton needs Python.h; <VENV_WITH_HEADERS> provides headers (<VENV> lacks them).
export CPATH="${SCR}/venvs/<VENV_WITH_HEADERS>/include/python3.11:${CPATH:-}"
# TriZone P0/P1 eager-only — Phase 3d CG recapture not yet ported.
export OBKV_DISABLE_CUDA_GRAPH=1
# Per-head K bit budget (k_avg_h independent per head).
export OBKV_K_AVG_MODE=perhead

TRITON_DIR="${SCR}/triton_cache_clean/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TRITON_DIR}" "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"
export TRITON_CACHE_DIR="${TRITON_DIR}"

cd "${OBKV_ROOT}"

echo "[$(date)] array=${ARRAY_ID} budget=${TOKEN_BUDGET} shard=${SHARD_ID}/${NUM_SHARDS} v_score=attn_linear k_score=think(default) streaming=on"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

"${RUNVENV_PY}" run_niah.py \
  --method obkv \
  --haystack-dir "${HAYSTACK_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_BASE}" \
  --method-version "${METHOD_VERSION}" \
  --s-len 2000 --e-len 32001 --step 400 \
  --token-budget "${TOKEN_BUDGET}" \
  --k-budget-ratio "${K_RATIO}" \
  --epsilon-path "${EPSILON_PATH}" \
  --obs-window "${OBS_WINDOW}" \
  --score-pooling "${POOL_KERNEL_SIZE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --streaming \
  --eviction-mode joint \
  --v-score-type attn_linear \
  --shard "${SHARD_ID}" --num-shards "${NUM_SHARDS}" \
  --seed 42 \
  2>&1 | tee "${LOGFILE}"

echo "[$(date)] array=${ARRAY_ID} budget=${TOKEN_BUDGET} shard=${SHARD_ID} done"
