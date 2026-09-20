# RDKV — Reference Implementation (NeurIPS Supplementary)

Code and reproduction scripts for the OBKV / RDKV KV-cache compression method evaluated in the paper.

## Repository layout

```
rdkv/
├── obkv_fast.py                 # core algorithm: streaming compression + tri-zone scoring + joint eviction
├── knapsack_solver.py           # per-head budget allocation
├── obkv_accel/                  # Triton kernels for compressed-cache decoding
│   ├── fast_decode.py           #   greedy_decode_fast entry point
│   ├── triton_{k,v}_kernel.py   #   masked attention over packed K/V zones
│   ├── triton_rope.py
│   ├── triton_rmsnorm.py
│   ├── triton_dequant_utils.py
│   ├── trizone_decompress.py
│   ├── packing.py               #   DualZoneCache + bit-packed layout
│   ├── bitpack.py
│   └── decode_hook.py
├── run_longbench.py / longbench_dataset.py / eval_longbench.py
├── run_ruler.py     / ruler_dataset.py     / eval_ruler.py
├── run_infinitebench.py                    / eval_infinitebench.py
├── run_niah.py
├── run_latency.py               # 7-context latency benchmark (TTFT / decode / TPOT / peak mem)
├── calibrate_epsilon.py         # regenerate per-layer ε(b) for a new backbone
├── external/LongBench/          # official LongBench v1 scorer (THUDM, MIT) — eval.py + metrics.py used by eval_longbench.py
├── longbench_manifest/          # 4-shard test manifest used by run_longbench.py
├── results/                     # per-backbone quantization-distortion calibrations
│   ├── epsilon_calibration_llama31_8b.json   (= epsilon_calibration_mixed_lengths.json, aliased)
│   ├── epsilon_calibration_mistral_7b.json
│   ├── epsilon_calibration_llama2_13b.json
│   ├── epsilon_calibration_qwen3_4b.json
│   └── epsilon_calibration_qwen25_72b.json
└── scripts/                     # 7 reference SLURM scripts (default config)
    ├── longbench_B1024_pertask_trizone_thinkK_attnlinear.slurm
    ├── ruler_64k_B1024_pertask_trizone_thinkK_attnlinear.slurm
    ├── infinitebench_B1024_pertask_trizone_thinkK_attnlinear.slurm
    ├── niah_B64_B128_streaming_thinkK_attnlin.slurm
    ├── latency_streaming_7ctx.slurm
    ├── download_infinitebench.slurm
    └── generate_ruler_data.slurm
```

## Default configuration

The shipped scripts use the default config reported in the paper:

| Component | Setting | Meaning |
|---|---|---|
| `--v-score-type` | `attn_linear` | SnapKV-style linear sum of recent-window attention as the V-token importance score |
| `--k-score-type` | `think` | "Think-step" K-token importance: K-projected query inner product over the recent window |
| `--streaming` | on | Compress the prompt in fixed-size streaming chunks during prefill (instead of full prompt → score → evict) |
| `--eviction-mode` | `joint` | Allocate a joint per-head token budget across all layers via knapsack; evict K and V positions jointly |
| `--token-budget` | `1024` | Total compressed tokens per head |
| `--k-budget-ratio` | `0.5` | Half of the budget is spent on K-side surviving positions; the other half feeds the V zones |
| `--pool-kernel-size` | `5` | Reflective average pool over importance scores before top-k selection |

## Supported backbones and epsilon calibration

The bit-allocation knapsack consumes a per-bit, per-layer **quantization-distortion calibration ε(b, ℓ)**. We ship calibrations for every backbone reported in the paper:

| Backbone | Layers | Epsilon file (under `results/`) |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 32 | `epsilon_calibration_llama31_8b.json` (also aliased as `epsilon_calibration_mixed_lengths.json`) |
| `mistralai/Mistral-7B-Instruct-v0.2` | 32 | `epsilon_calibration_mistral_7b.json` |
| `meta-llama/Llama-2-13b-chat-hf` | 40 | `epsilon_calibration_llama2_13b.json` |
| `Qwen/Qwen3-4B-Instruct` | 36 | `epsilon_calibration_qwen3_4b.json` |
| `Qwen/Qwen2.5-72B-Instruct` | 80 | `epsilon_calibration_qwen25_72b.json` |

Each `scripts/*.slurm` exposes `MODEL_PATH` and `EPSILON_PATH` at the top of the file — swap both together when changing backbones (epsilon is layer-count-specific). Without `--epsilon-path`, `knapsack_solver.py` falls back to a built-in `DEFAULT_EPSILON_K/V` that is conservative but suboptimal.

### Calibrating ε for a new backbone

[calibrate_epsilon.py](calibrate_epsilon.py) regenerates the file in ~10 minutes on a single A100. The metric is **layer-wise normalized mean-squared error of fake-quantization** with uniform asymmetric per-axis quantization:

$$
\text{NMSE}(b, \ell) \;=\; \frac{\mathbb{E}[(X_{\ell} - \hat{X}_{\ell}^{(b)})^2]}{\text{Var}(X_{\ell})}
$$

where the inner fake-quantization

$$
\hat{X} \;=\; s \cdot \big(\,\text{round}(X/s + z) - z\,\big),\quad s = \frac{\max X - \min X}{2^b - 1},\quad z = \text{round}\!\left(-\min X / s\right)
$$

is applied **per-channel along the seq-len axis** for `K` (`dim=2`) and **per-token along the head-dim axis** for `V` (`dim=3`). Bits 0 and 16 are sentinels (1.0 and 0.0, respectively) — they bracket the knapsack as "fully evicted" and "no quantization." Example:

```bash
python calibrate_epsilon.py \
    --model_name <backbone> \
    --data_source ruler \
    --ruler_data_dir /path/to/ruler_data \
    --ruler_lengths 4096,8192,16384,32768,65536,131072 \
    --num_samples_per_length 8 \
    --output_path results/epsilon_calibration_<backbone_tag>.json
```

The script also accepts `--data_source longbench` (LongBench `context` field, fast) or `--data_source infinitebench` (long-context tail).

## Environment

Tested on Python **3.11** + CUDA 12.x + 1×A100/H100 (80 GB).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install \
  "torch>=2.4,<2.6" \
  "transformers>=4.55,<4.60" \
  "accelerate>=0.30" \
  "datasets>=2.18" \
  "evaluate>=0.4" \
  "triton>=3.0" \
  "flash-attn>=2.6.3" --no-build-isolation \
  numpy scipy einops sentencepiece protobuf jieba rouge fuzzywuzzy
```

`evaluate` is required by [eval_infinitebench.py](eval_infinitebench.py) for ROUGE scoring. The other LongBench / RULER / NIAH scorers are pure-Python and pull `jieba`, `rouge`, `fuzzywuzzy` from the list above.

`flash-attn` requires CUDA-toolkit headers at build time; on a cluster, load the matching `cuda/` module first and set `CUDA_HOME`.

`obkv_accel/*.py` are JIT-compiled by Triton on first run; no separate build step.

## Datasets

### LongBench
Loaded on-the-fly from Hugging Face via the shipped dataset script ([longbench_dataset.py](longbench_dataset.py)) — it wraps `THUDM/LongBench`. The first run downloads each task into `~/.cache/huggingface/datasets/`; no manual download.

The 4-shard test manifest in [longbench_manifest/](longbench_manifest/) selects the official test split. The runner expects it under `<repo>/results/longbench/sample_manifests/`:

```bash
mkdir -p results/longbench/sample_manifests
ln -s ../../../longbench_manifest/test_num_shards_4_shard_*.json \
      results/longbench/sample_manifests/
```

### RULER
RULER data is generated locally from the upstream NVIDIA repo. Run [scripts/generate_ruler_data.slurm](scripts/generate_ruler_data.slurm) after cloning `https://github.com/NVIDIA/RULER` into `external/RULER` and adjusting the paths inside the script. Output goes to `ruler_data/llama31_8b_instruct/`.

### InfiniteBench
Run [scripts/download_infinitebench.slurm](scripts/download_infinitebench.slurm) — it pulls the `xinrongzhang2022/InfiniteBench` HF dataset.

### NIAH (Needle-in-a-Haystack)
Uses Paul Graham essays as the haystack. Either download `https://github.com/gkamradt/LLMTest_NeedleInAHaystack` and point `HAYSTACK_DIR` in the NIAH script at its `PaulGrahamEssays/` folder, or reuse the `niah_*` task data from RULER.

### Latency benchmark
[run_latency.py](run_latency.py) measures TTFT, decode total, TPOT, and peak memory across 7 contexts (4K → 256K). It feeds synthetic random token IDs of the requested length, so no external dataset download is needed. Run via [scripts/latency_streaming_7ctx.slurm](scripts/latency_streaming_7ctx.slurm) — output is one JSON with per-context timings (median over 3 repeats after 2 warmups).

## Running the scripts

Each `scripts/*.slurm` is a SLURM array job. **Before running, edit the path block at the top of each script** — the shipped values are for the SLURM cluster used in the paper:

```bash
REPO_ROOT="<REPO_ROOT>"   # → your unzip path's parent
OBKV_ROOT="${REPO_ROOT}/rdkv"                                  # → directory containing this README
SCR="<SCRATCH>"            # → fast scratch for results + Triton cache
EPSILON_PATH="${OBKV_ROOT}/results/epsilon_calibration_mixed_lengths.json"
```

Also adjust `--model-path` (default `${SCR}/hf_models/Llama-3.1-8B-Instruct`) and the `#SBATCH --account` / `--partition` lines.

Sample submission:

```bash
cd rdkv
sbatch scripts/longbench_B1024_pertask_trizone_thinkK_attnlinear.slurm
sbatch scripts/ruler_64k_B1024_pertask_trizone_thinkK_attnlinear.slurm
```

Or run a single LongBench task without SLURM:

```bash
python run_longbench.py \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --token-budget 1024 \
  --k-budget-ratio 0.5 \
  --pool-kernel-size 5 \
  --v-score-type attn_linear \
  --k-score-type think \
  --streaming \
  --eviction-mode joint \
  --epsilon-path results/epsilon_calibration_mixed_lengths.json \
  --num-shards 4 --shard 0 \
  --task narrativeqa \
  --output-dir /tmp/rdkv_demo
```

Aggregate per-task scores using the official LongBench v1 metrics (ships in [external/LongBench/metrics.py](external/LongBench/metrics.py)):

```bash
python eval_longbench.py --pred-dir /tmp/rdkv_demo
```

This prints a per-task table (F1 / ROUGE / classification / retrieval / count / code-similarity, per the official `dataset2metric` mapping) plus an arithmetic mean across the tasks present.

For RULER and InfiniteBench, use the matching `eval_ruler.py` / `eval_infinitebench.py`.

## Varying the configuration

To sweep cells in the paper's main results / ablation tables, change the corresponding flag — the same six scripts cover every batch size and backbone:

| Knob | Flag | Values |
|---|---|---|
| Total token budget | `--token-budget` | 32, 64, 128, 256, 512, 1024, 2048 |
| K/V split | `--k-budget-ratio` | 0.4 – 0.7 |
| Eviction | `--eviction-mode` | `joint` (default), `topk` |
| Streaming on/off | `--streaming` | omit the flag for full-prompt scoring |
| Backbone | `--model-path` | any LLaMA-architecture HF checkpoint |

V-side and K-side scoring functions are fixed to the paper's defaults (`--v-score-type attn_linear`, `--k-score-type think`); the corresponding flags accept only those single values.

## License & attribution

[external/LongBench/](external/LongBench/) is the official LongBench v1 scorer (MIT, THUDM/LongBench). All other code in this directory is released under the MIT license for review purposes.
