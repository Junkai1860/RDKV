# RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache

Official implementation of **RDKV**. Paper: [arXiv:2605.08317](https://arxiv.org/abs/2605.08317)

RDKV treats KV-cache compression for long-context LLM inference as a **bit-allocation problem** instead of a keep-or-evict decision. Each KV head receives a per-unit precision (from eviction to FP16) chosen by a rate-distortion objective, and decoding runs directly on the packed mixed-precision cache.

**Key components**

- Mixed-precision quantization and eviction of **V-cache tokens**
- Mixed-precision quantization and channel pruning of **K-cache channels**
- Per-head discrete bit allocation via Lagrangian relaxation and bisection
- Empirical quantization-distortion calibration
- Packed KV storage with fused Triton decoding kernels

> This is research code released for reproducibility. It is not a production inference library, and APIs may change.

---

## Method Overview

For every layer and KV head, RDKV compresses the cache once during prefill:

1. **Importance estimation.** A window of recent queries estimates V-side token importance (from attention weights) and K-side channel importance (from query/key statistics).
2. **Bit allocation.** Each cache unit is assigned a bit-width using its importance and a calibrated distortion table ε(b). The discrete problem is solved with Lagrangian relaxation and 1D bisection.
3. **Packing.** V is stored as 2/4/8-bit and FP16 segments plus evicted tokens. K is channel-pruned and stored as 2/4/8-bit segments.
4. **Compressed decoding.** Triton kernels read the packed cache directly, without reconstructing an FP16 KV cache.

The allocation is static during decoding. Experimental block-wise decode-time recompression is also included.

---

## Installation

Tested with Python 3.11, CUDA 12.x, PyTorch 2.4–2.5, Transformers 4.55–4.59, Triton 3.x and FlashAttention-2 on A100 / H100 / GH200.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install "torch>=2.4,<2.6" "transformers>=4.55,<4.60" "accelerate>=0.30" \
    "datasets>=2.18" "evaluate>=0.4" "triton>=3.0" \
    numpy scipy einops sentencepiece protobuf jieba rouge fuzzywuzzy

pip install "flash-attn>=2.6.3" --no-build-isolation
```

Triton kernels are JIT-compiled on first use. FlashAttention needs a matching CUDA toolkit at install time.

---

## Quick Start

```bash
python run_longbench.py \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --attn-implementation flash_attention_2 \
    --device-map single \
    --task-filter narrativeqa \
    --token-budget 1024 \
    --k-budget-ratio 0.5 \
    --epsilon-path results/epsilon_calibration_llama31_8b.json \
    --output-dir ./outputs/longbench/narrativeqa

python eval_longbench.py --pred-dir ./outputs/longbench
```

---

## Distortion Calibration

The allocator uses an empirical distortion table ε_K(b), ε_V(b) per bit-width, with ε(0) = 1 (eviction) and ε(16) = 0 (FP16). Precomputed tables are in `results/`:

| Model | File |
|---|---|
| Llama-3.1-8B | `epsilon_calibration_llama31_8b.json` |
| Mistral-7B | `epsilon_calibration_mistral_7b.json` |
| Llama-2-13B | `epsilon_calibration_llama2_13b.json` |
| Qwen3-4B | `epsilon_calibration_qwen3_4b.json` |
| Qwen2.5-72B | `epsilon_calibration_qwen25_72b.json` |

To calibrate a new model (data sources: `longbench`, `ruler`, `infinitebench`):

```bash
python calibrate_epsilon.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_source ruler \
    --ruler_data_dir /path/to/ruler_data \
    --ruler_lengths 4096,8192,16384,32768,65536,131072 \
    --num_samples_per_length 8 \
    --output_path results/epsilon_calibration_new_model.json
```

---

## Benchmarks

All runners share the same compression options (see below).

| Benchmark | Run | Evaluate | Notes |
|---|---|---|---|
| LongBench | `run_longbench.py` | `eval_longbench.py` | Official scoring in `external/LongBench/` |
| RULER | `run_ruler.py --ruler-data-dir ...` | `eval_ruler.py` | Generate data with NVIDIA RULER first (`scripts/generate_ruler_data.slurm`) |
| InfiniteBench | `run_infinitebench.py --data-dir ...` | built in | Download via `scripts/download_infinitebench.slurm` |
| Needle-in-a-Haystack | `run_niah.py --haystack-dir ...` | built in | Paul Graham essays as haystack |
| Latency / memory | `run_latency.py --context-length 131072` | built in | Synthetic inputs; reports TTFT, TPOT, decode time, peak memory |

Reference SLURM scripts are in `scripts/`. Replace placeholders such as `<ACCOUNT>`, `<PARTITION>`, `<REPO_ROOT>`, `<SCRATCH>` and `<VENV>` before use.

---

## Main Options

| Option | Meaning | Default |
|---|---|---|
| `--token-budget` | FP16-equivalent KV-cache budget | varies |
| `--k-budget-ratio` | Fraction of the bit budget given to K | 0.5 |
| `--obs-window` | Recent queries used for importance estimation | 32 |
| `--pool-kernel-size` | Smoothing kernel for token importance | 5 |
| `--epsilon-path` | Distortion calibration file | per model |
| `--v-bit-options` / `--k-bit-options` | Candidate bit-widths | |

`--token-budget` is a memory budget, not a token count. Because RDKV keeps many low-bit tokens and evicts others, the number of retained tokens is decided by the allocator.

---

## Repository Structure

```text
obkv_fast.py           Prefill, scoring, allocation, packing, inference
knapsack_solver.py     Discrete bit-allocation solver
calibrate_epsilon.py   Distortion calibration
obkv_accel/            Packing, bit-packing and Triton decode kernels
run_*.py / eval_*.py   Benchmark runners and evaluators
results/               Precomputed calibration tables
scripts/               Reference SLURM scripts
tests/                 Unit tests (pytest tests/)
external/LongBench/    Official LongBench evaluation code
```

Some files and symbols still use the legacy prefix `OBKV` from an earlier prototype. They belong to the current RDKV implementation.

---

## Citation

```bibtex
@misc{zhang2026rdkvratedistortionbitallocation,
      title={RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache}, 
      author={Junkai Zhang and Hang Guo and Luca Benini and Yawei Li},
      year={2026},
      eprint={2605.08317},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.08317}, 
}
```

## License

RDKV is released under the MIT License. `external/LongBench/` keeps its original MIT license and attribution. Datasets and model checkpoints are subject to their own licenses.
