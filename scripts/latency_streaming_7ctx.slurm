#!/bin/bash
#SBATCH --job-name=rdkv_latency_7ctx
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=<SCRATCH>/slurm_logs/rdkv/latency/lat-%j.out
#SBATCH --error=<SCRATCH>/slurm_logs/rdkv/latency/lat-%j.err

set -euo pipefail
mkdir -p <SCRATCH>/slurm_logs/rdkv/latency

# OBKV_ROOT defaults to the directory you ran `sbatch` from. Submit with:
#   cd <unzip_path>/rdkv && sbatch scripts/latency_streaming_7ctx.slurm
# Override by exporting OBKV_ROOT before submission if your layout differs.
OBKV_ROOT="${OBKV_ROOT:-${SLURM_SUBMIT_DIR}}"
SCR="<SCRATCH>"

module load gcc/12.2.0
source "${SCR}/venvs/<VENV>/bin/activate"
RUNVENV_PY="$(command -v python3)"

# Backbone — latency benchmark feeds synthetic random ids of fixed length, so
# only the architecture matters (no epsilon needed because run_latency.py
# defaults to DEFAULT_EPSILON_K/V baked into knapsack_solver.py).
MODEL_PATH="${SCR}/hf_models/Llama-3.1-8B-Instruct"

OUTPUT_DIR="${SCR}/rdkv_results/rdkv/latency"
mkdir -p "${OUTPUT_DIR}"

TRITON_DIR="${SCR}/triton_cache_clean/${SLURM_JOB_ID}"
mkdir -p "${TRITON_DIR}"

export HF_HOME="${SCR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TMPDIR="${SCR}/tmp"
export TORCH_EXTENSIONS_DIR="${SCR}/torch_extensions"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CPATH="${SCR}/venvs/<VENV_WITH_HEADERS>/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="${TRITON_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${OBKV_ROOT}:${PYTHONPATH:-}"
export OBKV_K_AVG_MODE=perhead
export OBKV_PREFILL_WARMUP=1
# pre_extract_weights frees the original q/k/v_proj + gate/up_proj weight
# storage after copying into cat'd qkv_w / gate_up_w tensors. Drops decode
# peak from ~25.8 GB to ~16.8 GB without affecting TPOT.
export OBKV_FREE_ORIGINAL_WEIGHTS=1

cd "${OBKV_ROOT}"

echo "[$(date)] RDKV latency benchmark — 7 contexts (4k..256k), B=1024, k_ratio=0.5, attn=SDPA, eviction=joint"
echo "obkv_fast.py lines: $(wc -l < ${OBKV_ROOT}/obkv_fast.py)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

"${RUNVENV_PY}" run_latency.py \
  --model "${MODEL_PATH}" \
  --streaming \
  --skip-full-kv \
  --context-lengths 4096 8192 16384 32768 65536 131072 262144 \
  --num-tokens 1024 \
  --token-budget 1024 \
  --k-budget-ratio 0.5 \
  --k-options 0 2 4 8 16 \
  --v-options 0 2 4 8 16 \
  --warmup 2 \
  --repeats 3 \
  --attn-implementation sdpa \
  --pool-kernel-size 5 \
  --pool-padding reflect \
  --v-score-type attn_linear \
  --eviction-mode joint \
  --output "${OUTPUT_DIR}/streaming_7ctx_default.json"

echo "[$(date)] done -> ${OUTPUT_DIR}/streaming_7ctx_default.json"
