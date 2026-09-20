# RDKV

Reference implementation for **RDKV**, a rate-distortion based framework for KV-cache compression in long-context LLM inference.

RDKV treats KV-cache compression as a **bit-allocation problem** rather than a fixed keep-or-evict decision. The implementation combines:

* mixed-precision quantization and eviction of **V-cache tokens**;
* mixed-precision quantization and channel pruning of **K-cache channels**;
* per-head discrete bit allocation using a Lagrangian solver;
* empirical quantization-distortion calibration;
* packed KV-cache storage with fused Triton decoding kernels.

The current repository is research code used for our long-context experiments. It is intended for reproducibility and further research rather than as a production inference library.

---

## Overview

For every transformer layer and KV head, RDKV performs a static compression step during prefill.

The implementation follows the pipeline below:

1. **Importance estimation**

   A recent-query observation window is used to estimate:

   * V-side token importance from attention weights;
   * K-side channel importance from query/key activation statistics.

2. **Bit allocation**

   Each cache unit is assigned a compression action according to its importance and an empirical quantization-distortion table.

   The discrete allocation is solved using Lagrangian relaxation with one-dimensional bisection.

3. **Packed cache construction**

   Selected KV states are quantized and packed into mixed-precision segments.

   The current packed representation uses:

   * V-cache: 2-bit, 4-bit, 8-bit, and FP16 segments, together with token eviction;
   * K-cache: channel pruning and 2-bit / 4-bit / 8-bit packed segments.

4. **Compressed decoding**

   Triton kernels consume the packed representation directly during autoregressive decoding, avoiding reconstruction of the entire KV cache in FP16.

The allocation is static during normal decoding. Experimental block-wise decode recompression code is also included.

---

## Repository Structure

```text
RDKV/
├── obkv_fast.py
│   └── main RDKV prefill, scoring, allocation, packing, and inference path
│
├── knapsack_solver.py
│   └── discrete bit-allocation solver
│
├── calibrate_epsilon.py
│   └── quantization-distortion calibration
│
├── obkv_accel/
│   ├── fast_decode.py
│   ├── packing.py
│   ├── bitpack.py
│   ├── triton_k_kernel.py
│   ├── triton_v_kernel.py
│   ├── triton_dequant_utils.py
│   ├── triton_rope.py
│   ├── triton_rmsnorm.py
│   ├── trizone_decompress.py
│   └── decode_hook.py
│
├── run_longbench.py
├── eval_longbench.py
├── longbench_dataset.py
│
├── run_ruler.py
├── eval_ruler.py
├── ruler_dataset.py
│
├── run_infinitebench.py
├── eval_infinitebench.py
│
├── run_niah.py
├── run_latency.py
│
├── results/
│   └── precomputed quantization-distortion calibration files
│
├── scripts/
│   └── reference SLURM scripts
│
├── tests/
│   └── implementation tests
│
└── external/LongBench/
    └── official LongBench evaluation code
```

Some internal function names, environment variables, and file names still use the prefix `OBKV`, inherited from an earlier prototype. They are part of the current RDKV implementation and will be cleaned up in future releases.

---

## Environment

The code has primarily been tested with:

* Python 3.11
* CUDA 12.x
* NVIDIA A100 / H100 / GH200 GPUs
* PyTorch 2.4–2.5
* Hugging Face Transformers 4.55–4.59
* Triton 3.x
* FlashAttention-2

A minimal environment can be created with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install \
    "torch>=2.4,<2.6" \
    "transformers>=4.55,<4.60" \
    "accelerate>=0.30" \
    "datasets>=2.18" \
    "evaluate>=0.4" \
    "triton>=3.0" \
    numpy scipy einops sentencepiece protobuf jieba rouge fuzzywuzzy

pip install "flash-attn>=2.6.3" --no-build-isolation
```

FlashAttention requires a compatible CUDA toolkit and CUDA headers during installation.

The Triton kernels are JIT-compiled on first use and do not require a separate build step.

---

## Precomputed Quantization Calibration

The discrete allocator uses empirical quantization distortion

```text
epsilon_K(b)
epsilon_V(b)
```

for each supported bit-width.

Precomputed calibration files are provided under `results/`:

```text
results/
├── epsilon_calibration_llama31_8b.json
├── epsilon_calibration_mistral_7b.json
├── epsilon_calibration_llama2_13b.json
├── epsilon_calibration_qwen3_4b.json
└── epsilon_calibration_qwen25_72b.json
```

These files contain normalized reconstruction-error statistics obtained from fake quantization.

At runtime, RDKV uses model-level aggregated distortion values for:

* per-channel K quantization;
* per-token V quantization.

The following conventions are used by the allocator:

```text
epsilon(0)  = 1      # eviction / removal
epsilon(16) = 0      # full-precision V representation
```

For K, the current packed implementation uses 2/4/8-bit storage after channel selection.

---

## Calibrating a New Model

`calibrate_epsilon.py` can be used to estimate the quantization-distortion table for a new backbone.

For example:

```bash
python calibrate_epsilon.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_source ruler \
    --ruler_data_dir /path/to/ruler_data \
    --ruler_lengths 4096,8192,16384,32768,65536,131072 \
    --num_samples_per_length 8 \
    --output_path results/epsilon_calibration_new_model.json
```

The script currently supports:

```text
longbench
ruler
infinitebench
```

as calibration data sources.

---

## LongBench

LongBench tasks can be run with `run_longbench.py`.

Example:

```bash
python run_longbench.py \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --attn-implementation flash_attention_2 \
    --device-map single \
    --task-filter narrativeqa \
    --token-budget 1024 \
    --k-budget-ratio 0.5 \
    --pool-kernel-size 5 \
    --epsilon-path results/epsilon_calibration_llama31_8b.json \
    --output-dir ./outputs/longbench/narrativeqa
```

Evaluate the generated predictions with:

```bash
python eval_longbench.py \
    --pred-dir ./outputs/longbench
```

The official LongBench scoring implementation included in `external/LongBench/` is used for evaluation.

---

## RULER

RULER data should first be generated using the upstream NVIDIA RULER repository.

A reference SLURM script is provided:

```text
scripts/generate_ruler_data.slurm
```

After generating the dataset:

```bash
python run_ruler.py \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --ruler-data-dir /path/to/ruler_data \
    --token-budget 1024 \
    --k-budget-ratio 0.5 \
    --epsilon-path results/epsilon_calibration_llama31_8b.json \
    --output-dir ./outputs/ruler
```

Evaluation is performed with:

```bash
python eval_ruler.py \
    --pred-dir ./outputs/ruler
```

---

## InfiniteBench

InfiniteBench can be evaluated with:

```bash
python run_infinitebench.py \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --data-dir /path/to/InfiniteBench \
    --token-budget 1024 \
    --k-budget-ratio 0.5 \
    --epsilon-path results/epsilon_calibration_llama31_8b.json \
    --output-dir ./outputs/infinitebench
```

A reference download script is included in:

```text
scripts/download_infinitebench.slurm
```

---

## Needle-in-a-Haystack

The repository also includes a standard NIAH evaluation:

```bash
python run_niah.py \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --haystack-dir /path/to/PaulGrahamEssays \
    --token-budget 128 \
    --epsilon-path results/epsilon_calibration_llama31_8b.json \
    --output-dir ./outputs/niah
```

Paul Graham essays from the commonly used Needle-in-a-Haystack benchmark can be used as the haystack source.

---

## Latency Benchmark

`run_latency.py` evaluates:

* time to first token;
* total decode time;
* time per output token;
* peak GPU memory.

The benchmark uses synthetic input token IDs and therefore does not require an external dataset.

Example:

```bash
python run_latency.py \
    --model /path/to/Llama-3.1-8B-Instruct \
    --context-length 131072 \
    --num-tokens 1024 \
    --token-budget 1024
```

A reference multi-context SLURM script is provided in:

```text
scripts/latency_streaming_7ctx.slurm
```

---

## Important Configuration Options

The main options used in the experiments are:

| Option               | Meaning                                                 |
| -------------------- | ------------------------------------------------------- |
| `--token-budget`     | FP16-equivalent KV-cache budget                         |
| `--k-budget-ratio`   | Fraction of the total bit budget allocated to K         |
| `--pool-kernel-size` | Local smoothing kernel for token importance             |
| `--epsilon-path`     | Quantization-distortion calibration file                |
| `--v-bit-options`    | Candidate V-cache bit-widths                            |
| `--k-bit-options`    | Candidate K-cache bit-widths                            |
| `--obs-window`       | Number of recent queries used for importance estimation |

The default experimental setting uses:

```text
observation window = 32
pooling kernel     = 5
K/V budget ratio  = 0.5 / 0.5
```

`--token-budget` is an **FP16-equivalent cache budget**, not the final number of retained tokens.

Because RDKV may retain many low-bit tokens and evict other tokens entirely, the number of surviving tokens is determined automatically by the bit allocator.

---

## SLURM Scripts

Reference cluster scripts are provided under:

```text
scripts/
```

Before using them, replace placeholders such as:

```text
<ACCOUNT>
<PARTITION>
<REPO_ROOT>
<SCRATCH>
<VENV>
```

with paths and resource settings for your own cluster.

These scripts are examples rather than portable cluster configurations.

---

## Implementation Notes

This repository reflects the research implementation used during development of RDKV.

A few practical details are worth noting:

* compression is performed layer-by-layer during prefill, after each layer has processed the complete prompt;
* allocation is performed independently for each KV head;
* the normal inference path uses a static compressed prompt cache during decoding;
* K and V use different natural compression granularities: channels for K and tokens for V;
* V has a dedicated FP16 zone for high-importance tokens;
* K channels are stored using the current packed 2/4/8-bit representation after pruning;
* scale and zero-point metadata are stored alongside packed values;
* several internal symbols still use legacy `OBKV_*` names.

The repository is under active cleanup, so APIs and internal implementation details may change.

---

## Tests

Basic implementation tests are available under:

```text
tests/
```

For example:

```bash
pytest tests/
```

---

## License and Third-Party Code

RDKV code is released under the MIT License.

The code under:

```text
external/LongBench/
```

comes from the official LongBench repository and retains its original MIT license and attribution.

Other external datasets and model checkpoints remain subject to their respective licenses.

---

## Citation

A citation entry will be added together with the public paper release.

For now, if you use this implementation in your research, please refer to the project as:

**RDKV: Rate-Distortion Bit Allocation for KV-Cache Compression**
