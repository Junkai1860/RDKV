"""
obkv_accel/fast_decode.py

Hand-written decode loop that bypasses HuggingFace model.forward() to
eliminate per-layer Python dispatch overhead (~35ms/tok -> target <15ms/tok).

Semantically equivalent to greedy_decode_packed() from decode_hook.py,
but operates directly on weight tensors and Triton kernels without
nn.Module.__call__ overhead.

Compatible with HuggingFace transformers >= 4.40 (kvpress_probe_py311 venv).
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# NVTX profiling gate: set OBKV_NVTX=1 to enable NVTX range markers.
# When disabled (default), the only overhead is a single bool check per step.
_NVTX_ENABLED = os.environ.get("OBKV_NVTX", "0") == "1"

# Per-component wall-clock diagnostic (eager path only): set OBKV_DECODE_DIAG=1
# to accumulate sync'd ms per label across all eager decode steps of a greedy
# call. Summary printed from greedy_decode_fast() at exit. Adds ~1 sync per
# block per step, so it's slow — use with --num-tokens small and CG=0.
_DECODE_DIAG_ENABLED = os.environ.get("OBKV_DECODE_DIAG", "0") == "1"
_DECODE_DIAG_TIMES: dict = {}
_DECODE_DIAG_COUNTS: dict = {}


def _diag_reset() -> None:
    _DECODE_DIAG_TIMES.clear()
    _DECODE_DIAG_COUNTS.clear()


def _diag_accum(label: str, dt_ms: float) -> None:
    _DECODE_DIAG_TIMES[label] = _DECODE_DIAG_TIMES.get(label, 0.0) + dt_ms
    _DECODE_DIAG_COUNTS[label] = _DECODE_DIAG_COUNTS.get(label, 0) + 1


def _diag_tic():
    if _DECODE_DIAG_ENABLED:
        torch.cuda.synchronize()
        return time.perf_counter()
    return None


def _diag_toc(label: str, tic) -> None:
    if tic is not None:
        torch.cuda.synchronize()
        _diag_accum(label, (time.perf_counter() - tic) * 1000.0)


def _diag_summary(header: str) -> None:
    if not _DECODE_DIAG_ENABLED or not _DECODE_DIAG_TIMES:
        return
    total = sum(_DECODE_DIAG_TIMES.values())
    print(f"[DECODE_DIAG] {header}")
    print(f"[DECODE_DIAG] {'label':<28}  {'calls':>6}  {'total_ms':>10}  "
          f"{'per_call_us':>12}  {'share':>6}")
    for label, t_ms in sorted(_DECODE_DIAG_TIMES.items(), key=lambda kv: -kv[1]):
        n = _DECODE_DIAG_COUNTS[label]
        per = (t_ms * 1000.0) / max(n, 1)
        share = t_ms / max(total, 1e-9) * 100.0
        print(f"[DECODE_DIAG] {label:<28}  {n:>6}  {t_ms:>10.2f}  "
              f"{per:>12.1f}  {share:>5.1f}%")
    print(f"[DECODE_DIAG] {'TOTAL':<28}  {'':>6}  {total:>10.2f}")


# CG-replay graph-level timing: set OBKV_GRAPH_TIMING=1 to time each Graph A/
# append/Graph B/Graph C replay slot per layer in steady-state CG steps,
# capturing both CPU duration (perf_counter around .replay()) and GPU duration
# (cuda.Event around .replay()). Compares CPU vs GPU per slot to determine
# whether the CG replay loop is host-bound (Python sync on .replay() ≈ GPU
# duration) or GPU-idle-bound (CPU returns immediately, GPU has gaps).
# Active only on steps in [_GRAPH_TIMING_STEP_LO, _GRAPH_TIMING_STEP_HI].
_GRAPH_TIMING_ENABLED = os.environ.get("OBKV_GRAPH_TIMING", "0") == "1"
_GRAPH_TIMING_STEP_LO = 4
_GRAPH_TIMING_STEP_HI = 9
# {label: [(cpu_us, gpu_us), ...]}
_GT_RECORDS: dict = {}
# Pre-allocated cuda.Event pool, lazy-initialized to size n_layers*3+1.
_GT_EV_START: list = []
_GT_EV_END: list = []
# Per-step scratch for pending CPU times and event-pair ids in record order.
_GT_PENDING: list = []  # list of (label, cpu_us, ev_idx)
_GT_NEXT_EV: int = 0
# Per-step wall clock (perf_counter) for full CG-replay step body
_GT_STEP_WALL_US: list = []


def _gt_ensure_pool(needed: int) -> None:
    """Grow the cuda.Event pool to at least `needed` event pairs."""
    while len(_GT_EV_START) < needed:
        _GT_EV_START.append(torch.cuda.Event(enable_timing=True))
        _GT_EV_END.append(torch.cuda.Event(enable_timing=True))


def _gt_step_begin() -> float:
    """Reset per-step state, return wall-clock t0."""
    global _GT_NEXT_EV
    _GT_NEXT_EV = 0
    _GT_PENDING.clear()
    return time.perf_counter()


def _gt_record(label: str, cpu_t0: float):
    """Pre-replay: claim an event pair, record start, return (ev_idx, ev_end).

    Caller pattern:
        ev_idx, ev_end = _gt_record(label, t0)
        graph.replay()
        cpu_us = (time.perf_counter() - t0) * 1e6
        ev_end.record()
        _gt_pending_finish(label, cpu_us, ev_idx)
    """
    global _GT_NEXT_EV
    idx = _GT_NEXT_EV
    _gt_ensure_pool(idx + 1)
    _GT_EV_START[idx].record()
    _GT_NEXT_EV += 1
    return idx, _GT_EV_END[idx]


def _gt_pending_finish(label: str, cpu_us: float, ev_idx: int) -> None:
    _GT_PENDING.append((label, cpu_us, ev_idx))


def _gt_step_end(step_t0: float) -> None:
    """Sync to materialize event times, accumulate, record step wall."""
    torch.cuda.synchronize()
    step_wall_us = (time.perf_counter() - step_t0) * 1e6
    _GT_STEP_WALL_US.append(step_wall_us)
    for label, cpu_us, ev_idx in _GT_PENDING:
        gpu_us = _GT_EV_START[ev_idx].elapsed_time(_GT_EV_END[ev_idx]) * 1e3
        _GT_RECORDS.setdefault(label, []).append((cpu_us, gpu_us))


# Intra-Graph-B segment timing: set OBKV_INTRA_B_TIMING=1 to insert cuda.Event
# nodes inside Graph B at 5 segment boundaries (qk_new, softmax, attn_out,
# o_proj, mlp). Events fire on every replay; after sync we read elapsed_time
# per segment per layer. Decides which sub-block of cg_B (the 71% chunk) to
# attack — V kernel, MLP, or attention block.
_INTRA_B_TIMING_ENABLED = os.environ.get("OBKV_INTRA_B_TIMING", "0") == "1"
_INTRA_B_LABELS = ("B_qk", "B_softmax", "B_attn_out", "B_oproj", "B_mlp")
# {(bucket_len, layer_i): [ev0, ev1, ev2, ev3, ev4, ev5]}
_IB_EVENTS: dict = {}
# {label: [gpu_us per replay]}
_IB_RECORDS: dict = {label: [] for label in _INTRA_B_LABELS}


def _ib_make_events_for_layer(bucket_len: int, layer_i: int):
    """Allocate 6 timing events for a (bucket_len, layer_i) Graph B."""
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(_INTRA_B_LABELS) + 1)]
    _IB_EVENTS[(bucket_len, layer_i)] = evs
    return evs


def _ib_collect(bucket_len: int) -> None:
    """Sync once, then accumulate per-segment elapsed_time for all layers in bucket_len."""
    torch.cuda.synchronize()
    for (bl, li), evs in _IB_EVENTS.items():
        if bl != bucket_len:
            continue
        for k, label in enumerate(_INTRA_B_LABELS):
            _IB_RECORDS[label].append(evs[k].elapsed_time(evs[k + 1]) * 1e3)  # ms→us


def _ib_summary(header: str) -> None:
    if not _INTRA_B_TIMING_ENABLED or not any(_IB_RECORDS[l] for l in _INTRA_B_LABELS):
        return
    print(f"[INTRA_B] {header}")
    print(f"[INTRA_B] {'segment':<12}  {'n':>5}  {'us(med)':>10}  {'us(sum)':>12}  {'share':>6}")
    sums = {l: sum(_IB_RECORDS[l]) for l in _INTRA_B_LABELS}
    total = sum(sums.values())
    for label in _INTRA_B_LABELS:
        recs = _IB_RECORDS[label]
        if not recs:
            continue
        n = len(recs)
        med = sorted(recs)[n // 2]
        s = sums[label]
        share = s / max(total, 1e-9) * 100.0
        print(f"[INTRA_B] {label:<12}  {n:>5}  {med:>10.2f}  {s:>12.1f}  {share:>5.1f}%")
    print(f"[INTRA_B] {'TOTAL':<12}  {'':>5}  {'':>10}  {total:>12.1f}")


def _gt_summary(header: str) -> None:
    if not _GRAPH_TIMING_ENABLED or not _GT_RECORDS:
        return
    print(f"[GRAPH_TIMING] {header}")
    print(f"[GRAPH_TIMING] {'label':<14}  {'n':>4}  "
          f"{'cpu_us(med)':>12}  {'gpu_us(med)':>12}  "
          f"{'cpu_us(sum)':>12}  {'gpu_us(sum)':>12}  "
          f"{'cpu/gpu':>8}")
    # Aggregate by collapsing layer index in label suffix (e.g., cg_A_0..31 → cg_A).
    agg: dict = {}
    for label, recs in _GT_RECORDS.items():
        bucket = label.rsplit("_", 1)[0] if label[-1].isdigit() else label
        agg.setdefault(bucket, []).extend(recs)
    for bucket in sorted(agg.keys()):
        recs = agg[bucket]
        cpu = sorted(r[0] for r in recs)
        gpu = sorted(r[1] for r in recs)
        n = len(recs)
        cpu_med = cpu[n // 2]
        gpu_med = gpu[n // 2]
        cpu_sum = sum(cpu)
        gpu_sum = sum(gpu)
        ratio = (cpu_sum / gpu_sum) if gpu_sum > 0 else float("nan")
        print(f"[GRAPH_TIMING] {bucket:<14}  {n:>4}  "
              f"{cpu_med:>12.2f}  {gpu_med:>12.2f}  "
              f"{cpu_sum:>12.1f}  {gpu_sum:>12.1f}  "
              f"{ratio:>8.3f}")
    if _GT_STEP_WALL_US:
        sw = sorted(_GT_STEP_WALL_US)
        ns = len(sw)
        print(f"[GRAPH_TIMING] step_wall      {ns:>4}  "
              f"{sw[ns//2]:>12.2f}  {'':>12}  "
              f"{sum(sw):>12.1f}  {'':>12}  {'':>8}")

from obkv_accel.triton_rope import fused_rope

from obkv_accel.packing import DualZoneCache, PackedKVLayer
from obkv_accel.triton_k_kernel import (
    k_mixed_qk_dot, k_mixed_qk_dot_into,
    k_diag_enabled, k_diag_reset, k_diag_snapshot,
)
from obkv_accel.triton_v_kernel import (
    _get_v16_dummy,
    v_dequant_weighted_sum_dispatch,
    v_dequant_weighted_sum_dispatch_into,
)

_SQRT_D = math.sqrt(128.0)
_INV_SQRT_D = 1.0 / _SQRT_D
_GQA_FACTOR = 4  # H_q // H_kv for Llama-3.1-8B


@triton.jit
def _fused_split_softmax_kernel(
    scores_old_ptr, scores_new_ptr,
    w_old_ptr, w_new_ptr,
    T_old, T_new,
    stride_so_h, stride_so_t,
    stride_sn_h, stride_sn_t,
    stride_wo_h, stride_wo_t,
    stride_wn_h, stride_wn_t,
    BLOCK_T: tl.constexpr,
):
    """Fused softmax over cat([scores_old, scores_new]) without materializing the cat.

    Reads from two separate score tensors, computes global softmax,
    writes w_old and w_new to separate output buffers.
    3-pass: (1) max, (2) exp+sum, (3) normalize+store.
    """
    pid_h = tl.program_id(0)

    # Pass 1: global max
    max_val = float('-inf')
    for start in range(0, T_old, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_old
        s = tl.load(scores_old_ptr + pid_h * stride_so_h + offs * stride_so_t,
                     mask=mask, other=float('-inf'))
        block_max = tl.max(s, axis=0)
        max_val = tl.where(block_max > max_val, block_max, max_val)
    for start in range(0, T_new, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_new
        s = tl.load(scores_new_ptr + pid_h * stride_sn_h + offs * stride_sn_t,
                     mask=mask, other=float('-inf'))
        block_max = tl.max(s, axis=0)
        max_val = tl.where(block_max > max_val, block_max, max_val)

    # Pass 2: exp + sum
    sum_exp = 0.0
    for start in range(0, T_old, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_old
        s = tl.load(scores_old_ptr + pid_h * stride_so_h + offs * stride_so_t,
                     mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(s - max_val) * mask.to(tl.float32), axis=0)
    for start in range(0, T_new, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_new
        s = tl.load(scores_new_ptr + pid_h * stride_sn_h + offs * stride_sn_t,
                     mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(s - max_val) * mask.to(tl.float32), axis=0)

    inv_sum = 1.0 / sum_exp

    # Pass 3: normalize + write
    for start in range(0, T_old, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_old
        s = tl.load(scores_old_ptr + pid_h * stride_so_h + offs * stride_so_t,
                     mask=mask, other=float('-inf'))
        w = tl.exp(s - max_val) * inv_sum
        tl.store(w_old_ptr + pid_h * stride_wo_h + offs * stride_wo_t, w, mask=mask)
    for start in range(0, T_new, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < T_new
        s = tl.load(scores_new_ptr + pid_h * stride_sn_h + offs * stride_sn_t,
                     mask=mask, other=float('-inf'))
        w = tl.exp(s - max_val) * inv_sum
        tl.store(w_new_ptr + pid_h * stride_wn_h + offs * stride_wn_t, w, mask=mask)


def _fused_split_softmax(scores_old, scores_new, w_old_out, T_old):
    """Fused cat-free softmax. Writes w_old into w_old_out, returns w_new."""
    H_q = scores_old.shape[0]
    T_new = scores_new.shape[1]
    w_new = torch.empty(H_q, T_new, dtype=torch.float32, device=scores_old.device)
    BLOCK_T = 256
    _fused_split_softmax_kernel[(H_q,)](
        scores_old, scores_new,
        w_old_out, w_new,
        T_old, T_new,
        scores_old.stride(0), scores_old.stride(1),
        scores_new.stride(0), scores_new.stride(1),
        w_old_out.stride(0), w_old_out.stride(1),
        w_new.stride(0), w_new.stride(1),
        BLOCK_T=BLOCK_T,
    )
    return w_new


# =====================================================================
# Pre-extracted weight references
# =====================================================================

@dataclass
class _LayerWeights:
    """Weight references for a single decoder layer."""
    input_ln_weight: torch.Tensor
    input_ln_eps: float
    q_proj_weight: torch.Tensor
    k_proj_weight: torch.Tensor
    v_proj_weight: torch.Tensor
    o_proj_weight: torch.Tensor
    # Q/K/V projection biases (Qwen2 has these; Llama / Qwen3 do not).
    # None ⇒ no bias term (F.linear(.., None) is the same as bias=False).
    q_proj_bias: Optional[torch.Tensor] = None
    k_proj_bias: Optional[torch.Tensor] = None
    v_proj_bias: Optional[torch.Tensor] = None
    # Kept for compatibility (not used in decode loop after inline MLP)
    mlp: object = None
    post_ln: object = None
    # Fused QKV weight (pre-concatenated in pre_extract_weights). Set to
    # None when running multi-device (skip the cat to avoid duplicating
    # ~140 GiB of 70B weights across all participating GPUs).
    qkv_weight: Optional[torch.Tensor] = None
    qkv_split_sizes: Tuple[int, int, int] = (0, 0, 0)
    # Fused QKV bias (pre-concatenated alongside qkv_weight). None when
    # the model has no q/k/v bias OR when on the no-cat (multi-device) path.
    qkv_bias: Optional[torch.Tensor] = None
    # Post-attention LayerNorm (inline RMS norm)
    post_ln_weight: torch.Tensor = None
    post_ln_eps: float = 1e-5
    # Fused Gate+Up weight (pre-concatenated, same pattern as QKV merge)
    gate_up_weight: Optional[torch.Tensor] = None  # [2 * intermediate_size, hidden_size]
    gate_up_split_sizes: Tuple[int, int] = (0, 0)
    down_proj_weight: torch.Tensor = None     # [hidden_size, intermediate_size]
    # Separate gate/up weights — populated only on multi-device (no-cat) path
    # so the decode loop can do `F.linear(h, gate_w)` and `F.linear(h, up_w)`
    # rather than `F.linear(h, cat([gate_w,up_w]))`.
    gate_proj_weight: Optional[torch.Tensor] = None
    up_proj_weight: Optional[torch.Tensor] = None
    # Qwen3-style per-head q/k RMSNorm (after q/k_proj reshape, before RoPE).
    # None for models without q/k_norm (e.g. Llama) — decode paths skip the norm.
    q_norm_weight: Optional[torch.Tensor] = None   # [head_dim]
    k_norm_weight: Optional[torch.Tensor] = None   # [head_dim]
    qk_norm_eps: float = 1e-5
    # Per-layer device (set from q_proj.weight.device in pre_extract_weights).
    device: Optional[torch.device] = None


@dataclass
class PreExtractedWeights:
    """Flat references to all model weights, extracted once before decode."""
    embed_tokens: object           # nn.Embedding
    final_norm_weight: torch.Tensor
    final_norm_eps: float
    lm_head_weight: torch.Tensor
    rotary_emb: object             # LlamaRotaryEmbedding (shared)
    layers: List[_LayerWeights] = field(default_factory=list)
    # Model constants
    num_layers: int = 0
    num_heads: int = 0             # H_q
    num_kv_heads: int = 0          # H_kv
    head_dim: int = 0
    hidden_size: int = 0
    attn_output_dim: int = 0       # num_heads * head_dim (= o_proj in_features; may differ from hidden_size on Qwen3)
    num_kv_groups: int = 0         # H_q // H_kv
    # Per-module devices (read from real weight/buffer devices). Used by
    # greedy_decode_fast under multi-device TP to route h/position_ids/cos/sin.
    embed_dev: Optional[torch.device] = None
    norm_dev: Optional[torch.device] = None
    lm_dev: Optional[torch.device] = None
    rotary_dev: Optional[torch.device] = None
    # True iff every model parameter (excluding meta) lives on the same
    # device — gate for the fused QKV / gate_up cat path. Multi-device
    # falls back to 3 separate F.linear (QKV) + 2 separate F.linear (MLP).
    model_single_device: bool = True


def pre_extract_weights(model) -> PreExtractedWeights:
    """Extract all weight references from a LlamaForCausalLM model.

    Called once before the decode loop.  All references point to the
    original model tensors (no copies), so this is O(1) memory.
    """
    cached = getattr(model, "_obkv_pre_extracted_weights", None)
    if cached is not None:
        return cached

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)

    # Source-of-truth for device discovery: real parameter devices, NOT
    # `model.hf_device_map` (which can store mixed types — int / str /
    # torch.device / "cpu" — and is a placement plan rather than ground truth).
    _real_devs = {
        p.device for p in model.parameters() if p.device.type != "meta"
    }
    model_single_device = (len(_real_devs) == 1)
    # Force separate (no-cat) QKV/MLP linears even on single-GPU. Used by
    # Gate 2 to exercise the multi-device numerical path on cheap single-GPU
    # hardware; bit-equivalence vs the cat path then certifies the no-cat
    # branch independently of routing bugs.
    force_no_fused = os.environ.get("OBKV_FORCE_NO_FUSED_WEIGHTS", "0") == "1"
    use_cat = model_single_device and not force_no_fused
    # OBKV_FREE_ORIGINAL_WEIGHTS replaces attn.q/k/v_proj.weight and
    # mlp.gate/up_proj.weight with views into the cat'd qkv_w / gate_up_w
    # tensors, then drops the local references so the original storage is
    # freed by the caching allocator. Saves ~9 GB on Llama-3.1-8B (the cost
    # of the cat copy that would otherwise stay alive alongside the
    # originals). **Default ON**; set ``OBKV_FREE_ORIGINAL_WEIGHTS=0`` to
    # disable (e.g., for A/B comparisons). Only valid under ``use_cat``
    # (multi-device / no-fused path uses the original q/k/v tensors
    # directly in 3 separate F.linears). HF model.forward keeps working
    # because the new Parameters wrap views with the same
    # shape/stride/dtype as the originals, and the cat output is
    # contiguous → views are also contiguous. Bit-exact verified vs.
    # pre-flip code (1024-token greedy decode, token-for-token MATCH).
    free_originals = (
        use_cat and os.environ.get("OBKV_FREE_ORIGINAL_WEIGHTS", "1") != "0"
    )

    layer_weights: List[_LayerWeights] = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp
        q_w = attn.q_proj.weight          # [q_dim, hidden_size]
        k_w = attn.k_proj.weight          # [kv_dim, hidden_size]
        v_w = attn.v_proj.weight          # [kv_dim, hidden_size]
        # Q/K/V biases: present on Qwen2 (attention_bias=True default),
        # absent on Llama / Qwen3 (attention_bias=False). nn.Linear(bias=False)
        # exposes .bias = None, so getattr handles both cases uniformly.
        q_b = getattr(attn.q_proj, "bias", None)
        k_b = getattr(attn.k_proj, "bias", None)
        v_b = getattr(attn.v_proj, "bias", None)
        gate_w = mlp.gate_proj.weight     # [intermediate_size, hidden_size]
        up_w = mlp.up_proj.weight         # [intermediate_size, hidden_size]
        q_norm_mod = getattr(attn, "q_norm", None)
        k_norm_mod = getattr(attn, "k_norm", None)
        q_norm_w = q_norm_mod.weight if q_norm_mod is not None else None
        k_norm_w = k_norm_mod.weight if k_norm_mod is not None else None
        qk_eps = getattr(q_norm_mod, "variance_epsilon", 1e-5) if q_norm_mod is not None else 1e-5

        any_qkv_bias = (q_b is not None) or (k_b is not None) or (v_b is not None)
        if use_cat:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            gate_up_w = torch.cat([gate_w, up_w], dim=0)
            sep_gate_w = None
            sep_up_w = None
            if any_qkv_bias:
                # Models that mix bias=True with bias=False per-projection don't
                # exist in HF today, but be defensive: fill missing slots with
                # zeros so the cat path stays a single F.linear.
                _q_bf = q_b if q_b is not None else q_w.new_zeros(q_w.shape[0])
                _k_bf = k_b if k_b is not None else k_w.new_zeros(k_w.shape[0])
                _v_bf = v_b if v_b is not None else v_w.new_zeros(v_w.shape[0])
                qkv_b = torch.cat([_q_bf, _k_bf, _v_bf], dim=0)
            else:
                qkv_b = None
            if free_originals:
                # Per-layer view replacement: makes attn.q_proj.weight etc.
                # views into qkv_w / gate_up_w, dropping references to the
                # original storage. Done inside the loop so peak overhead is
                # bounded by one layer's cat (~283 MB on Llama-3.1-8B) rather
                # than 32 layers' worth (~9 GB) before the loop ends.
                q_sz = q_w.shape[0]
                k_sz = k_w.shape[0]
                gate_sz = gate_w.shape[0]
                attn.q_proj.weight = torch.nn.Parameter(
                    qkv_w[:q_sz], requires_grad=False,
                )
                attn.k_proj.weight = torch.nn.Parameter(
                    qkv_w[q_sz:q_sz + k_sz], requires_grad=False,
                )
                attn.v_proj.weight = torch.nn.Parameter(
                    qkv_w[q_sz + k_sz:], requires_grad=False,
                )
                mlp.gate_proj.weight = torch.nn.Parameter(
                    gate_up_w[:gate_sz], requires_grad=False,
                )
                mlp.up_proj.weight = torch.nn.Parameter(
                    gate_up_w[gate_sz:], requires_grad=False,
                )
                if any_qkv_bias and qkv_b is not None:
                    attn.q_proj.bias = torch.nn.Parameter(
                        qkv_b[:q_sz], requires_grad=False,
                    )
                    attn.k_proj.bias = torch.nn.Parameter(
                        qkv_b[q_sz:q_sz + k_sz], requires_grad=False,
                    )
                    attn.v_proj.bias = torch.nn.Parameter(
                        qkv_b[q_sz + k_sz:], requires_grad=False,
                    )
                # Update local refs so _LayerWeights below stores the views.
                # The old originals lose their last reference at this rebind
                # (no other holders) and the caching allocator releases them.
                q_w = attn.q_proj.weight
                k_w = attn.k_proj.weight
                v_w = attn.v_proj.weight
                gate_w = mlp.gate_proj.weight
                up_w = mlp.up_proj.weight
                if any_qkv_bias:
                    q_b = getattr(attn.q_proj, "bias", None)
                    k_b = getattr(attn.k_proj, "bias", None)
                    v_b = getattr(attn.v_proj, "bias", None)
        else:
            # Multi-device or forced no-cat path: skip cat to (a) avoid
            # ~140 GiB weight duplication on 70B and (b) certify the
            # 3-linear / 2-linear branches.
            qkv_w = None
            qkv_b = None
            gate_up_w = None
            sep_gate_w = gate_w
            sep_up_w = up_w

        layer_weights.append(_LayerWeights(
            input_ln_weight=layer.input_layernorm.weight,
            input_ln_eps=layer.input_layernorm.variance_epsilon,
            q_proj_weight=q_w,
            k_proj_weight=k_w,
            v_proj_weight=v_w,
            o_proj_weight=attn.o_proj.weight,
            q_proj_bias=q_b,
            k_proj_bias=k_b,
            v_proj_bias=v_b,
            mlp=mlp,
            post_ln=layer.post_attention_layernorm,
            # cat output is guaranteed contiguous -> cuBLAS F.linear takes optimal path
            qkv_weight=qkv_w,
            qkv_split_sizes=(q_w.shape[0], k_w.shape[0], v_w.shape[0]),
            qkv_bias=qkv_b,
            # Post-attention LayerNorm (for inline RMS norm)
            post_ln_weight=layer.post_attention_layernorm.weight,
            post_ln_eps=layer.post_attention_layernorm.variance_epsilon,
            # Gate+Up merge (same pattern as QKV merge)
            gate_up_weight=gate_up_w,
            gate_up_split_sizes=(gate_w.shape[0], up_w.shape[0]),
            down_proj_weight=mlp.down_proj.weight,
            gate_proj_weight=sep_gate_w,
            up_proj_weight=sep_up_w,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            qk_norm_eps=qk_eps,
            device=q_w.device,
        ))

    # rotary_emb: prefer model-level (HF >= 4.48), fallback to layer-level
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = model.model.layers[0].self_attn.rotary_emb
    rotary_buf = next(rotary_emb.buffers(), None)
    rotary_dev = (
        rotary_buf.device
        if rotary_buf is not None
        else model.model.embed_tokens.weight.device
    )

    weights = PreExtractedWeights(
        embed_tokens=model.model.embed_tokens,
        final_norm_weight=model.model.norm.weight,
        final_norm_eps=model.model.norm.variance_epsilon,
        lm_head_weight=model.lm_head.weight,
        rotary_emb=rotary_emb,
        layers=layer_weights,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=cfg.hidden_size,
        attn_output_dim=num_heads * head_dim,
        num_kv_groups=num_heads // num_kv_heads,
        embed_dev=model.model.embed_tokens.weight.device,
        norm_dev=model.model.norm.weight.device,
        lm_dev=model.lm_head.weight.device,
        rotary_dev=rotary_dev,
        model_single_device=model_single_device,
    )
    model._obkv_pre_extracted_weights = weights
    return weights


# =====================================================================
# Inline helpers
# =====================================================================

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Inline RMSNorm — avoids nn.Module.__call__ overhead."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(x.dtype)



def _pre_squeeze_packed(
    packed_layers: Tuple[PackedKVLayer, ...],
) -> List[PackedKVLayer]:
    """Squeeze batch dim from packed layers once (avoid per-step squeeze)."""
    out = []
    for pl in packed_layers:
        out.append(PackedKVLayer(
            K_2bit=pl.K_2bit.squeeze(0) if pl.K_2bit is not None else None,
            K_4bit=pl.K_4bit.squeeze(0) if pl.K_4bit is not None else None,
            K_8bit=pl.K_8bit.squeeze(0) if pl.K_8bit is not None else None,
            K_ch_scale=pl.K_ch_scale.squeeze(0),
            K_ch_zp=pl.K_ch_zp.squeeze(0),
            K_ch_sort_idx=pl.K_ch_sort_idx,
            K_ch_seg_bounds=pl.K_ch_seg_bounds,
            K_ch_perm_padded_idx=pl.K_ch_perm_padded_idx,
            V_2bit=pl.V_2bit.squeeze(0) if pl.V_2bit is not None else None,
            V_4bit=pl.V_4bit.squeeze(0) if pl.V_4bit is not None else None,
            V_8bit=pl.V_8bit.squeeze(0) if pl.V_8bit is not None else None,
            V_scale=pl.V_scale.squeeze(0),
            V_zp=pl.V_zp.squeeze(0),
            sort_idx=pl.sort_idx,
            seg_bounds=pl.seg_bounds,
            T_eff=pl.T_eff,
            # Per-head joint-knapsack metadata. If None on pl (global path),
            # propagated as None so downstream per-head dispatch checks fall
            # back to the legacy scalar-tuple kernel.
            seg_bounds_per_head=pl.seg_bounds_per_head,
            T_eff_per_head=pl.T_eff_per_head,
            softmax_mask=pl.softmax_mask,
            n_fp16_per_head=pl.n_fp16_per_head,
            fp16_gap_mask=pl.fp16_gap_mask,
            # Phase 2 per-head K metadata (None on legacy packs).
            K_ch_sort_idx_per_head=pl.K_ch_sort_idx_per_head,
            K_ch_perm_padded_idx_per_head=pl.K_ch_perm_padded_idx_per_head,
            K_ch_seg_bounds_per_head=pl.K_ch_seg_bounds_per_head,
            K_ch_perm_padded_idx_for_Q=pl.K_ch_perm_padded_idx_for_Q,
            # TriZone (方案 1) K T-dim layout.
            T_eff_k=pl.T_eff_k,
            n_v16=pl.n_v16,
            T_eff_k_per_head=pl.T_eff_k_per_head,
            n_v16_per_head=pl.n_v16_per_head,
            allocation_v_bits=pl.allocation_v_bits,
            allocation_k_bits=pl.allocation_k_bits,
        ))
    return out


def _update_new_scores_mask(mask_buf: torch.Tensor, valid_cols: int) -> None:
    """Mark valid new-cache positions with 0 and invalid tail with -inf."""
    valid_cols = max(0, min(valid_cols, mask_buf.shape[1]))
    mask_buf.fill_(float('-inf'))
    mask_buf[:, :valid_cols] = 0.0


def _update_new_scores_mask_batched(
    mask_buf: torch.Tensor,
    valid_per_layer: torch.Tensor,
    col_ids: torch.Tensor,
) -> None:
    """Vectorised per-layer mask update.

    Two kernel launches regardless of layer count (fill + masked_fill_).

    Args:
        mask_buf: [L, 1, bucket_len] float mask (mutated in place).
        valid_per_layer: [L] long, clamped valid prefix length per layer.
        col_ids: [bucket_len] long arange scratch.
    """
    mask_buf.fill_(float('-inf'))
    # col_ids[None, :] < valid[:, None] → [L, bucket_len] bool; broadcast over
    # the H_row=1 dim of mask_buf.
    valid_mask = col_ids.unsqueeze(0) < valid_per_layer.unsqueeze(1)  # [L, bucket_len]
    mask_buf.masked_fill_(valid_mask.unsqueeze(1), 0.0)


def _build_bucket_sizes(max_len: int, min_bucket: int = 32) -> List[int]:
    """Build monotonically increasing decode-length buckets up to ``max_len``."""
    if max_len <= 0:
        return [1]

    size = max(32, 1 << (min_bucket - 1).bit_length()) if min_bucket > 0 else 32
    buckets: List[int] = []
    while size < max_len:
        buckets.append(size)
        size *= 2
    if not buckets or buckets[-1] != max_len:
        buckets.append(max_len)
    return buckets


# =====================================================================
# Main decode function
# =====================================================================

def greedy_decode_fast(
    model,
    dual_cache: DualZoneCache,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: int,
    use_cuda_graph: bool = False,
    return_logits: bool = False,
    next_token_hidden: Optional[torch.Tensor] = None,
    logits_observer: Optional[
        Callable[[int, torch.Tensor, int, bool], None]
    ] = None,
    hidden_observer: Optional[
        Callable[[int, torch.Tensor, int, bool], None]
    ] = None,
) -> List[int]:
    """Fast greedy decode bypassing HuggingFace model.forward().

    Semantically equivalent to ``greedy_decode_packed()`` but with
    dramatically reduced Python dispatch overhead.

    When *use_cuda_graph* is True, step 0 runs eagerly (to JIT-compile
    Triton kernels and warm cuBLAS), then 64+1 CUDA Graphs are captured
    and replayed for subsequent steps.

    ``logits_observer`` is an optional online side channel used by
    trajectory-replay diagnostics.  Before a token is consumed it receives
    ``(zero_based_step, full_vocabulary_logits, selected_token_id, is_eos)``.
    The observer must reduce or consume the logits immediately; the decoder
    does not retain them.

    ``hidden_observer`` receives the corresponding final-normalized hidden
    state instead of vocabulary logits.  This permits a separate downstream
    job to reconstruct the exact distribution with the unchanged LM head
    without writing full-vocabulary tensors to disk.
    """
    eos_set = {int(tid) for tid in eos_token_ids if tid is not None}
    generated_ids: List[int] = []
    # E2 Gate V-3b: optional per-step last-position FP32 logits collection.
    # CG path does not produce a `logits` tensor (only _next_token_buf),
    # so we disallow the combination.
    if (
        return_logits
        or logits_observer is not None
        or hidden_observer is not None
    ) and use_cuda_graph:
        raise NotImplementedError(
            "logit observation is only supported under eager decode "
            "(use_cuda_graph=False); CG path writes directly to next_token_buf."
        )
    if hidden_observer is not None and next_token_hidden is None:
        raise ValueError("hidden_observer requires next_token_hidden")
    collected_logits: List[torch.Tensor] = [] if return_logits else None

    # ---- Pre-extract weights (once) ----
    W = pre_extract_weights(model)
    # Multi-device topology log + Gate 3 assertion (skipped silently on
    # cached weight extraction). Helpful for verifying balanced TP smoke.
    if not getattr(model, "_obkv_device_topology_logged", False):
        unique_devs = {str(lw.device) for lw in W.layers}
        print(
            f"[obkv][topology] layers split across devices: {sorted(unique_devs)} "
            f"embed={W.embed_dev} norm={W.norm_dev} lm_head={W.lm_dev} "
            f"rotary={W.rotary_dev} model_single_device={W.model_single_device}",
            flush=True,
        )
        if os.environ.get("OBKV_REQUIRE_MULTI_DEVICE", "0") == "1":
            assert len(unique_devs) > 1, (
                "OBKV_REQUIRE_MULTI_DEVICE=1 expected multi-device split, "
                f"got single device {unique_devs} — smoke is testing nothing new"
            )
        model._obkv_device_topology_logged = True
    model_dtype = next(model.parameters()).dtype
    _n_layers = W.num_layers
    _H_q = W.num_heads
    _H_kv = W.num_kv_heads
    _GQA = W.num_kv_groups
    _D = W.head_dim

    # ---- Pre-squeeze packed data (once per DualZoneCache) ----
    squeezed = getattr(dual_cache, "_pre_squeezed_packed", None)
    if squeezed is None:
        squeezed = _pre_squeeze_packed(dual_cache.packed)
        dual_cache._pre_squeezed_packed = squeezed

    # ---- TriZone (方案 1) buffers ----
    # Zone B (prefill v=16 V) is a read-only FP16 tensor from the cache.
    # Zone C (decode-appended K+V) uses an FP32 interleaved buffer sized
    # max_new_tokens (no prefill seed — Zone A K lives in packed and Zone B
    # V lives in new_v_only_buf).
    n_gen = 0  # number of generated tokens so far

    new_v_only_buf: List[Optional[torch.Tensor]] = [
        (v16.contiguous() if v16 is not None else None)
        for v16 in dual_cache.new_v_only
    ]
    n_v16_per_layer: List[int] = [
        (nv.shape[2] if nv is not None else 0) for nv in new_v_only_buf
    ]

    # ``new_kv_buf`` is decode-only under TriZone; historical tokens live in
    # ``packed``, so ``n_existing_per_layer`` is [0]*L by construction.
    new_kv_cache = getattr(dual_cache, "_new_kv_buf_cache", None)
    n_existing_per_layer: List[int] = [0] * W.num_layers
    max_n_existing = 0
    max_total_cap = max_new_tokens

    _buf_key = (
        tuple(n_v16_per_layer),
        max_new_tokens,
        W.num_kv_heads,
        W.head_dim,
        primary_device,
        # Per-layer device tuple invalidates cache when the model is
        # re-placed (e.g. between balanced and single device_map runs).
        tuple(str(lw.device) for lw in W.layers),
    )
    cache_ok = (
        new_kv_cache is not None
        and new_kv_cache.get("key") == _buf_key
    )

    if cache_ok:
        new_kv_buf = new_kv_cache["buffers"]
    else:
        new_kv_buf = []
        for i in range(W.num_layers):
            # Zero-init FP32 buffer; each decode step writes K/V to row n_gen.
            # Allocated on each layer's own device so the per-layer Triton K/V
            # kernel and bmm see same-device tensors under multi-GPU TP.
            buf = torch.zeros(
                1, W.num_kv_heads, max_total_cap, 2 * W.head_dim,
                device=W.layers[i].device, dtype=torch.float32,
            )
            new_kv_buf.append(buf)

        dual_cache._new_kv_buf_cache = {
            "key": _buf_key,
            "buffers": new_kv_buf,
        }

    # ---- Pre-allocate reusable buffers (A1-A3) ----
    # TriZone: old-zone length is T_eff_k (compressed + v=16), falling back
    # to T_eff for pre-TriZone packs. Computed once at function entry — do
    # NOT move inside the layer loop (would make it O(L²) per step).
    T_old_per_layer: List[int] = [
        (sp.T_eff_k if sp.T_eff_k > 0 else sp.T_eff) for sp in squeezed
    ]
    T_old_max = max(T_old_per_layer)
    max_T_new_plus_1 = max_total_cap  # upper bound on max_i(T_new+1)

    # Zone B FP16 v=16 buffers for the fused V kernel. The kernel reads FP16
    # directly and promotes to FP32 inside (no host-side pre-cast). For layers
    # without Zone B we substitute a cached dummy tensor + zero per-head
    # count, so HAS_V16=1 is uniform across all per-head V kernel calls and
    # only one Triton specialization is compiled.
    # Per-device dummy under multi-GPU TP: any layer on cuda:1 with empty
    # V=16 zone would otherwise feed a cuda:0 dummy into a cuda:1 kernel
    # call → device mismatch crash.
    _v16_dummy_per_dev: dict = {}

    def _v16_dummy_for(dev):
        cached = _v16_dummy_per_dev.get(dev)
        if cached is None:
            cached = _get_v16_dummy(dev, W.num_kv_heads)
            _v16_dummy_per_dev[dev] = cached
        return cached

    _V16_bufs: List[torch.Tensor] = []
    _n_v16_ph_bufs: List[torch.Tensor] = []
    for i, (nv, sp) in enumerate(zip(new_v_only_buf, squeezed)):
        lw_dev = W.layers[i].device
        v16_dummy_buf, n_v16_ph_zeros = _v16_dummy_for(lw_dev)
        if nv is not None:
            v16_buf = nv[0]
            if v16_buf.device != lw_dev:
                v16_buf = v16_buf.to(lw_dev)
        else:
            v16_buf = v16_dummy_buf
        _V16_bufs.append(v16_buf)
        if sp.n_v16_per_head is not None:
            n_v16_ph = sp.n_v16_per_head
            if n_v16_ph.device != lw_dev:
                n_v16_ph = n_v16_ph.to(lw_dev)
        else:
            n_v16_ph = n_v16_ph_zeros
        _n_v16_ph_bufs.append(n_v16_ph)

    # ---- CUDA Graph: persistent cache on dual_cache ----
    # Multi-device TP forces eager (CUDA Graph would need cross-device
    # capture which torch.cuda.Graph does not support).
    if use_cuda_graph and not W.model_single_device:
        warnings.warn(
            "CUDA Graph disabled: model is split across multiple devices "
            f"(layer devices: {sorted({str(lw.device) for lw in W.layers})}). "
            "Falling back to eager decode."
        )
        use_cuda_graph = False
    _cg_ok = False
    _cg = getattr(dual_cache, '_cg_cache', None) if use_cuda_graph else None
    # I10 tripwire: softmax_mask width must equal per-layer T_old_i (when
    # non-empty). Legacy packs may set softmax_mask to torch.empty(H_q, 0);
    # that is accepted — the Graph B add site is gated by `T_old > 0` and
    # the mask being non-None+non-empty is orthogonal. If the packer ever
    # emits a mismatched-width non-empty mask, fail here with a clean stack
    # rather than deep inside a captured graph.
    if use_cuda_graph:
        for _i, _sp in enumerate(squeezed):
            if _sp.softmax_mask is not None and _sp.softmax_mask.shape[1] > 0:
                assert _sp.softmax_mask.shape[1] == T_old_per_layer[_i], (
                    f"softmax_mask[{_i}].shape[1]={_sp.softmax_mask.shape[1]} "
                    f"!= T_old_per_layer[{_i}]={T_old_per_layer[_i]}"
                )
    # Composite invalidation key: captured graphs bake per-layer T_old,
    # T_v, n_v16, mask widths, new_kv_buf identities, and bucket layout
    # into their op streams. Recapture whenever any of these change.
    # E2: _V_TC / _K_TC / _V_TC_T_CHUNKS pulled from env so kernel dispatch
    # changes retrigger capture. Defaults ON as of 2026-04-24.
    _V_TC_FLAG = int(os.environ.get("OBKV_V_TC", "1"))
    _K_TC_FLAG = int(os.environ.get("OBKV_K_TC", "1"))
    _V_TC_T_CHUNKS_FLAG = int(os.environ.get("OBKV_V_TC_T_CHUNKS", "16"))
    _cg_key = (
        tuple(n_existing_per_layer),
        tuple(T_old_per_layer),
        tuple(sp.T_eff for sp in squeezed),
        tuple(n_v16_per_layer),
        W.num_layers, W.num_heads, W.num_kv_heads, W.head_dim,
        max_new_tokens,
        _V_TC_FLAG, _K_TC_FLAG, _V_TC_T_CHUNKS_FLAG,  # E2
    )
    if _cg is not None and _cg.get('cg_key') != _cg_key:
        _cg = None
        try:
            del dual_cache._cg_cache
        except Exception:
            pass
    if _cg is not None:
        # Reuse previously captured graphs + buffers
        _h_buf = _cg['h_buf']
        _residual_buf = _cg['residual_buf']
        _cos_half_buf = _cg['cos_half_buf']
        _sin_half_buf = _cg['sin_half_buf']
        _q_2d_buf = _cg['q_2d_buf']
        _kv_2d_buf = _cg['kv_2d_buf']
        _k_2d_buf = _kv_2d_buf[:, :_kv_2d_buf.shape[1] // 2]
        _v_2d_buf = _kv_2d_buf[:, _kv_2d_buf.shape[1] // 2:]
        _w_old_buf = _cg['w_old_buf']
        _out_new_buf = _cg['out_new_buf']
        _out_old_buf = _cg['out_old_buf']
        _all_scores_bufs = _cg['all_scores_bufs']
        _scores_old_bufs = _cg['scores_old_bufs']
        _bucket_sizes = _cg['bucket_sizes']
        _scores_new_bufs = _cg['scores_new_bufs']
        _scores_new_mask_bufs = _cg['scores_new_mask_bufs']
        _mask_col_ids = _cg['mask_col_ids']
        _mask_valid_buf = _cg['mask_valid_buf']
        _n_existing_buf = _cg['n_existing_buf']
        _next_token_buf = _cg['next_token_buf']
        _graphs_a = _cg['graphs_a']
        _graphs_b = _cg['graphs_b']
        _graph_c = _cg['graph_c']
        _cg_ok = True
    elif use_cuda_graph:
        # First call: allocate buffers, will capture after step 0
        _h_buf = torch.empty(1, 1, W.hidden_size, dtype=model_dtype, device=primary_device)
        _residual_buf = torch.empty(1, 1, W.hidden_size, dtype=model_dtype, device=primary_device)
        _cos_half_buf = torch.empty(W.head_dim // 2, dtype=torch.float32, device=primary_device)
        _sin_half_buf = torch.empty(W.head_dim // 2, dtype=torch.float32, device=primary_device)
        _q_2d_buf = torch.empty(W.num_heads, W.head_dim, dtype=torch.float32, device=primary_device)
        _kv_2d_buf = torch.empty(W.num_kv_heads, 2 * W.head_dim, dtype=model_dtype, device=primary_device)
        _k_2d_buf = _kv_2d_buf[:, :W.head_dim]   # view
        _v_2d_buf = _kv_2d_buf[:, W.head_dim:]    # view
        _w_old_buf = torch.empty(W.num_heads, T_old_max, dtype=torch.float32, device=primary_device)
        _out_new_buf = torch.empty(W.num_heads, W.head_dim, dtype=torch.float32, device=primary_device)
        _out_old_buf = torch.empty(W.num_heads, W.head_dim, dtype=torch.float32, device=primary_device)
        # Pre-allocate combined [old|new] score buffers per layer so that
        #   (a) `torch.cat([scores_old, scores_new])` becomes a pure copy, and
        #   (b) softmax can run in-place, avoiding two ~[H_q, T_old+T_new] allocations
        #       per layer per step.
        # Width per layer is T_old_i = T_eff_k (Zone A+B) — the K kernel
        # writes T_old_i columns, so `_scores_old_bufs[i]` is a view on the
        # first T_old_i columns of _all_scores_bufs[i].
        _all_scores_bufs = [
            torch.empty(
                W.num_heads, T_old_i + max_T_new_plus_1,
                dtype=torch.float32, device=primary_device,
            )
            for T_old_i in T_old_per_layer
        ]
        _scores_old_bufs = [
            buf[:, :T_old_i]
            for buf, T_old_i in zip(_all_scores_bufs, T_old_per_layer)
        ]
        # Joint per-head knapsack can make per-layer max_n_fp16 exceed the
        # default minimum bucket (32). fp16_gap_mask[:, :_mw] must fit in
        # _scores_new_buf[:, :bucket_len]; floor buckets at max(max_n_fp16_i).
        _max_fp16_gap = max(
            (sp.fp16_gap_mask.shape[1] for sp in squeezed if sp.fp16_gap_mask is not None),
            default=0,
        )
        _bucket_sizes = _build_bucket_sizes(
            max(max_T_new_plus_1, _max_fp16_gap),
            min_bucket=_max_fp16_gap,
        )
        _scores_new_bufs = {
            bucket_len: torch.full(
                (_H_q, bucket_len), float('-inf'),
                dtype=torch.float32, device=primary_device,
            )
            for bucket_len in _bucket_sizes
        }
        # Per-layer masks: each layer may have a different valid prefix when
        # n_existing_per_layer differs. Shape [L, 1, bucket_len] so a single
        # fill_/masked_fill_ pair updates all layers (2 kernel launches per
        # step instead of 2·L). Broadcasts over H_q at Graph B add-time.
        _scores_new_mask_bufs = {
            bucket_len: torch.full(
                (_n_layers, 1, bucket_len), float('-inf'),
                dtype=torch.float32, device=primary_device,
            )
            for bucket_len in _bucket_sizes
        }
        # Device-resident scratch for vectorised mask update. Allocated once.
        _mask_col_ids = {
            bucket_len: torch.arange(bucket_len, device=primary_device, dtype=torch.long)
            for bucket_len in _bucket_sizes
        }
        _mask_valid_buf = torch.empty(_n_layers, dtype=torch.long, device=primary_device)
        # Persistent GPU copy of n_existing_per_layer so the hot-path
        # mask update can avoid CPU→GPU list-to-tensor copies every step.
        _n_existing_buf = torch.tensor(
            n_existing_per_layer, dtype=torch.long, device=primary_device,
        )
        _next_token_buf = torch.empty(1, 1, dtype=torch.long, device=primary_device)
        _graphs_a = [None] * W.num_layers
        _graphs_b = {
            bucket_len: [None] * W.num_layers
            for bucket_len in _bucket_sizes
        }
        _graph_c = None

    # ---- First token from prefill logits ----
    if next_token_logits.dim() == 1:
        next_token_logits = next_token_logits.unsqueeze(0)
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

    # ---- Decode loop ----
    # Single-GPU keeps the historical fast-path: one position_ids tensor on
    # primary_device, one cos/sin compute on primary_device. Multi-device
    # uses per-device dicts pre-populated for primary_device + rotary_dev so
    # the in-step fill loop covers them BEFORE the first rotary_emb call.
    # Pre-population is critical: lazy creation produced a zeros buffer that
    # rotary_emb consumed on step 0 with position=0 instead of decode_position_start,
    # which silently corrupted RoPE for the first generated token (and for
    # every K written into new_kv_buf at step 0). Reproducible only when
    # rotary_dev != primary_device (i.e. balanced TP across multiple GPUs).
    position_ids = torch.zeros((1, 1), dtype=torch.long, device=primary_device)
    _pos_ids_per_dev: dict = {primary_device: position_ids}
    if W.rotary_dev is not None and W.rotary_dev != primary_device:
        _pos_ids_per_dev[W.rotary_dev] = torch.zeros(
            (1, 1), dtype=torch.long, device=W.rotary_dev,
        )

    def _pos_ids_for_dev(dev):
        buf = _pos_ids_per_dev.get(dev)
        if buf is None:
            # Should not happen for rotary_dev (pre-populated above); kept as
            # a safety net for any future caller. Initialize with the value
            # of `position_ids` so the buffer is always fill_(pos)-correct
            # for the current step.
            buf = position_ids.to(dev)
            _pos_ids_per_dev[dev] = buf
        return buf

    _k_diag_on = k_diag_enabled()
    _k_diag_target_step = 1  # skip step 0 (JIT warmup); capture steady-state step 1
    _diag_reset()
    _diag_steps_timed = 0
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if _k_diag_on and step == _k_diag_target_step:
                k_diag_reset()
            # Per-component diag: reset accumulators at step 1 so step-0 JIT
            # warmup (kernel compilation) doesn't dominate the totals.
            if _DECODE_DIAG_ENABLED and step == 1:
                _diag_reset()
                _diag_steps_timed = 0
            token_id = int(next_token.item())
            if logits_observer is not None:
                logits_observer(
                    step,
                    next_token_logits if step == 0 else logits[:, -1, :],
                    token_id,
                    token_id in eos_set,
                )
            if hidden_observer is not None:
                hidden_observer(
                    step,
                    next_token_hidden if step == 0 else h[:, -1, :],
                    token_id,
                    token_id in eos_set,
                )
            if token_id in eos_set:
                break
            generated_ids.append(token_id)

            # NVTX: only annotate steps 4-9 (skip warmup)
            _prof = _NVTX_ENABLED and 4 <= step <= 9
            if _prof: torch.cuda.nvtx.range_push("decode_step")

            pos = decode_position_start + step
            position_ids.fill_(pos)
            # Sync every per-device position_ids buffer with the new pos.
            # Single-GPU: this dict has one entry == position_ids — no extra work.
            for _dev, _buf in _pos_ids_per_dev.items():
                if _buf is not position_ids:
                    _buf.fill_(pos)

            # === Embedding (route to embed_dev under multi-device) ===
            _d = _diag_tic()
            h = W.embed_tokens(next_token.to(W.embed_dev))  # [1, 1, hidden_size]
            _diag_toc("embed", _d)

            # === RoPE (computed once per step on rotary_dev, then per-layer
            # to(dev) inside the loop; cache with one-step lifetime since
            # cos/sin depend on `pos`).
            _d = _diag_tic()
            _pos_ids_rot = _pos_ids_for_dev(W.rotary_dev)
            # Build a small fp16 dummy on rotary_dev to drive rotary_emb. The
            # cos/sin output reflects pos through internal multiplication, so
            # h's content is irrelevant — only its dtype/device matter.
            _h_for_rope = h if h.device == W.rotary_dev else h.to(W.rotary_dev)
            if W.rotary_dev is not None and W.rotary_dev.type == "cuda":
                torch.cuda.set_device(W.rotary_dev)
            cos, sin = W.rotary_emb(_h_for_rope, _pos_ids_rot)
            cos_half_root = cos.view(W.head_dim)[:W.head_dim // 2]
            sin_half_root = sin.view(W.head_dim)[:W.head_dim // 2]
            cos_half = cos_half_root  # primary copy used by single-GPU path
            sin_half = sin_half_root
            _rope_this_step: dict = {W.rotary_dev: (cos_half_root, sin_half_root)}

            def _rope_for_dev(dev):
                cached = _rope_this_step.get(dev)
                if cached is not None:
                    return cached
                cached = (cos_half_root.to(dev), sin_half_root.to(dev))
                _rope_this_step[dev] = cached
                return cached
            _diag_toc("rotary_emb_compute", _d)

            # === CUDA Graph: capture after step 0 warmup ===
            if use_cuda_graph and step == 1 and not _cg_ok:
                _cg_stage = "capture:init"
                try:
                    _cg_stage = "capture:create_stream"
                    _cg_stream = torch.cuda.Stream()
                    _cg_stream.wait_stream(torch.cuda.current_stream())
                    # -- warmup in capture stream --
                    with torch.cuda.stream(_cg_stream):
                        _cg_stage = "warmup:stream"
                        for _wi in range(3):
                            _cg_stage = f"warmup:layer0_iter{_wi}"
                            lw0, sp0 = W.layers[0], squeezed[0]
                            T_old0_w = T_old_per_layer[0]
                            T_v0_w = sp0.T_eff
                            _rms_norm(_h_buf, lw0.input_ln_weight, lw0.input_ln_eps)
                            _tmp_qkv = F.linear(_h_buf, lw0.qkv_weight, lw0.qkv_bias)
                            _tmp_q, _tmp_k, _tmp_v = _tmp_qkv.split(lw0.qkv_split_sizes, dim=-1)
                            _tmp_q2 = _tmp_q.view(W.num_heads, W.head_dim)
                            _tmp_k2 = _tmp_k.view(W.num_kv_heads, W.head_dim)
                            _tmp_v2 = _tmp_v.view(W.num_kv_heads, W.head_dim)
                            if lw0.q_norm_weight is not None:
                                _tmp_q2 = _rms_norm(_tmp_q2, lw0.q_norm_weight, lw0.qk_norm_eps)
                            if lw0.k_norm_weight is not None:
                                _tmp_k2 = _rms_norm(_tmp_k2, lw0.k_norm_weight, lw0.qk_norm_eps)
                            fused_rope(_tmp_q2, _tmp_k2, _cos_half_buf, _sin_half_buf)
                            _tmp_q2_f32 = _tmp_q2.float()
                            if T_old0_w > 0:
                                k_mixed_qk_dot_into(
                                    _tmp_q2_f32,
                                    sp0,
                                    T_old0_w,
                                    _scores_old_bufs[0],
                                )
                            _q_2d_buf.copy_(_tmp_q2_f32 * _INV_SQRT_D)
                            _k_2d_buf.copy_(_tmp_k2)
                            _v_2d_buf.copy_(_tmp_v2)
                            if T_old0_w > 0:
                                v_dequant_weighted_sum_dispatch_into(
                                    _w_old_buf[:, :T_old0_w], sp0, _out_old_buf,
                                    V16=_V16_bufs[0],
                                    n_v16_per_head=_n_v16_ph_bufs[0],
                                    max_T_v=T_v0_w,
                                )
                                _out_old_buf.add_(_out_new_buf)
                            else:
                                _out_old_buf.copy_(_out_new_buf)
                            _at = _out_old_buf.to(model_dtype).view(1, 1, W.attn_output_dim)
                            _at = F.linear(_at, lw0.o_proj_weight)
                            _hh = _residual_buf + _at
                            _r2 = _hh.clone()
                            _hh = _rms_norm(_hh, lw0.post_ln_weight, lw0.post_ln_eps)
                            _gu = F.linear(_hh, lw0.gate_up_weight)
                            _g, _u = _gu.split(lw0.gate_up_split_sizes, dim=-1)
                            _hh = F.linear(F.silu(_g) * _u, lw0.down_proj_weight)
                            _h_buf.copy_(_r2 + _hh)

                        # Warm dynamic-shape ops for every replay bucket before capture.
                        T_old0 = T_old_per_layer[0]
                        T_v0 = squeezed[0].T_eff
                        n_v16_0 = T_old0 - T_v0
                        _new_k_full0 = new_kv_buf[0][0, :, :, :_D]
                        _new_v_full0 = new_kv_buf[0][0, :, :, _D:]
                        # Warm the batched mask updater once per bucket so the
                        # kernel variants are JIT-compiled outside capture.
                        for bucket_len in _bucket_sizes:
                            _cg_stage = f"warmup:layer0_bucket{bucket_len}"
                            _scores_new_buf = _scores_new_bufs[bucket_len]
                            # Per-layer mask bank: [L, 1, bucket_len]
                            _scores_new_mask_bank = _scores_new_mask_bufs[bucket_len]
                            torch.clamp(
                                _n_existing_buf + (n_gen + 1),
                                max=bucket_len,
                                out=_mask_valid_buf,
                            )
                            _update_new_scores_mask_batched(
                                _scores_new_mask_bank,
                                _mask_valid_buf,
                                _mask_col_ids[bucket_len],
                            )
                            torch.bmm(
                                _q_2d_buf.view(_H_kv, _GQA, _D),
                                _new_k_full0[:, :bucket_len, :].transpose(-1, -2),
                                out=_scores_new_buf.view(_H_kv, _GQA, bucket_len),
                            )
                            # Warm per-layer slice: [1, bucket_len] broadcast
                            # over H_q into _scores_new_buf [H_q, bucket_len].
                            _scores_new_buf.add_(_scores_new_mask_bank[0])
                            # Warm per-head masks for layer 0 so Triton JIT
                            # caches any kernels before graph capture.
                            _sp0 = squeezed[0]
                            if _sp0.fp16_gap_mask is not None and _sp0.fp16_gap_mask.shape[1] > 0:
                                _mw0 = _sp0.fp16_gap_mask.shape[1]
                                _scores_new_buf[:, :_mw0].add_(_sp0.fp16_gap_mask)
                            _all_scores_buf = _all_scores_bufs[0][:, :T_old0 + bucket_len]
                            _all_scores_buf[:, :T_old0].copy_(_scores_old_bufs[0][:, :T_old0])
                            _all_scores_buf[:, T_old0:T_old0 + bucket_len].copy_(_scores_new_buf)
                            if _sp0.softmax_mask is not None and T_old0 > 0:
                                _all_scores_buf[:, :T_old0].add_(_sp0.softmax_mask)
                            _all_w = F.softmax(_all_scores_buf, dim=-1)
                            # Zone A+B fused V kernel pre-touch: primes Triton
                            # HAS_V16=1 variant so the first invocation inside
                            # the captured graph does not JIT-compile.
                            if T_old0 > 0:
                                v_dequant_weighted_sum_dispatch_into(
                                    _all_w[:, :T_old0], squeezed[0], _out_old_buf,
                                    V16=_V16_bufs[0],
                                    n_v16_per_head=_n_v16_ph_bufs[0],
                                    max_T_v=T_v0,
                                )
                            torch.bmm(
                                _all_w[:, T_old0:T_old0 + bucket_len].view(_H_kv, _GQA, bucket_len),
                                _new_v_full0[:, :bucket_len, :],
                                out=_out_new_buf.view(_H_kv, _GQA, _D),
                            )

                        # Per-layer Triton JIT pre-warm. In per-head mode each layer
                        # has its own T_eff and K/V seg_bounds, so k_mixed_qk_dot_into
                        # and v_dequant_weighted_sum_dispatch_into would otherwise
                        # JIT-compile new kernel variants on first call *inside* the
                        # capture, which transitively touches the CUDA allocator and
                        # trips `captures_underway == 0 INTERNAL ASSERT` on Python-side
                        # empty_cache() later. Pre-fill Triton's kernel cache here for
                        # every layer's specific packed-layer signature.
                        _q_scratch_f32 = _q_2d_buf.float().view(W.num_heads, W.head_dim)
                        for _i_warm in range(W.num_layers):
                            _cg_stage = f"warmup:layer{_i_warm}_packed_kernels"
                            _sp_w = squeezed[_i_warm]
                            T_old_w = T_old_per_layer[_i_warm]
                            T_v_w = _sp_w.T_eff
                            if T_old_w > 0:
                                k_mixed_qk_dot_into(
                                    _q_scratch_f32,
                                    _sp_w,
                                    T_old_w,
                                    _scores_old_bufs[_i_warm],
                                )
                                # Fused V kernel pre-warm (Zone A+B). Use
                                # _all_scores_bufs[_i_warm][:, :T_old_w] as a
                                # weights scratch — valid shape, content is
                                # junk (output overwritten before consumption).
                                v_dequant_weighted_sum_dispatch_into(
                                    _all_scores_bufs[_i_warm][:, :T_old_w],
                                    _sp_w,
                                    _out_old_buf,
                                    V16=_V16_bufs[_i_warm],
                                    n_v16_per_head=_n_v16_ph_bufs[_i_warm],
                                    max_T_v=T_v_w,
                                )
                            _new_k_full_w = new_kv_buf[_i_warm][0, :, :, :_D]
                            _new_v_full_w = new_kv_buf[_i_warm][0, :, :, _D:]
                            for bucket_len in _bucket_sizes:
                                _cg_stage = f"warmup:layer{_i_warm}_bucket{bucket_len}_graphb"
                                _scores_new_buf = _scores_new_bufs[bucket_len]
                                _scores_new_mask_buf_i = _scores_new_mask_bufs[bucket_len][_i_warm]
                                _scores_new_3d = _scores_new_buf.view(_H_kv, _GQA, bucket_len)
                                torch.bmm(
                                    _q_2d_buf.view(_H_kv, _GQA, _D),
                                    _new_k_full_w[:, :bucket_len, :].transpose(-1, -2),
                                    out=_scores_new_3d,
                                )
                                _scores_new_buf.add_(_scores_new_mask_buf_i)
                                if _sp_w.fp16_gap_mask is not None and _sp_w.fp16_gap_mask.shape[1] > 0:
                                    _mw_w = _sp_w.fp16_gap_mask.shape[1]
                                    _scores_new_buf[:, :_mw_w].add_(_sp_w.fp16_gap_mask)
                                _all_scores_buf = _all_scores_bufs[_i_warm][:, :T_old_w + bucket_len]
                                if T_old_w > 0:
                                    _all_scores_buf[:, :T_old_w].copy_(_scores_old_bufs[_i_warm][:, :T_old_w])
                                _all_scores_buf[:, T_old_w:T_old_w + bucket_len].copy_(_scores_new_buf)
                                if _sp_w.softmax_mask is not None and T_old_w > 0:
                                    _all_scores_buf[:, :T_old_w].add_(_sp_w.softmax_mask)
                                _all_w = F.softmax(_all_scores_buf, dim=-1)
                                if T_old_w > 0:
                                    v_dequant_weighted_sum_dispatch_into(
                                        _all_w[:, :T_old_w],
                                        _sp_w,
                                        _out_old_buf,
                                        V16=_V16_bufs[_i_warm],
                                        n_v16_per_head=_n_v16_ph_bufs[_i_warm],
                                        max_T_v=T_v_w,
                                    )
                                else:
                                    _out_old_buf.zero_()
                                torch.bmm(
                                    _all_w[:, T_old_w:T_old_w + bucket_len].view(_H_kv, _GQA, bucket_len),
                                    _new_v_full_w[:, :bucket_len, :],
                                    out=_out_new_buf.view(_H_kv, _GQA, _D),
                                )
                    torch.cuda.current_stream().wait_stream(_cg_stream)

                    for i in range(W.num_layers):
                        _cg_stage = f"capture:graph_a_layer{i}"
                        lw, sp = W.layers[i], squeezed[i]
                        T_old = T_old_per_layer[i]       # Zone A+B length
                        T_v = sp.T_eff                   # Zone A length
                        # -- Graph A: rms_norm → qkv → rope → K kernel --
                        ga = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(ga, stream=_cg_stream):
                            _residual_buf.copy_(_h_buf)
                            _ha = _rms_norm(_h_buf, lw.input_ln_weight, lw.input_ln_eps)
                            _qkv = F.linear(_ha, lw.qkv_weight, lw.qkv_bias)
                            _q, _k, _v = _qkv.split(lw.qkv_split_sizes, dim=-1)
                            _q2 = _q.view(W.num_heads, W.head_dim)
                            _k2 = _k.view(W.num_kv_heads, W.head_dim)
                            _v2 = _v.view(W.num_kv_heads, W.head_dim)
                            if lw.q_norm_weight is not None:
                                _q2 = _rms_norm(_q2, lw.q_norm_weight, lw.qk_norm_eps)
                            if lw.k_norm_weight is not None:
                                _k2 = _rms_norm(_k2, lw.k_norm_weight, lw.qk_norm_eps)
                            fused_rope(_q2, _k2, _cos_half_buf, _sin_half_buf)
                            _q2_f32 = _q2.float()
                            # K kernel covers packed K stream = T_v compressed
                            # + n_v16 FP16-V tokens. Skip when T_old == 0 (pre-
                            # TriZone fallback with empty compressed zone).
                            if T_old > 0:
                                k_mixed_qk_dot_into(
                                    _q2_f32,
                                    sp,
                                    T_old,
                                    _scores_old_bufs[i],
                                )
                            _q_2d_buf.copy_(_q2_f32 * _INV_SQRT_D)  # pre-scaled f32
                            _k_2d_buf.copy_(_k2)
                            _v_2d_buf.copy_(_v2)
                        _graphs_a[i] = ga

                        # -- Graph B: capture one replay graph per new-cache bucket --
                        for bucket_len in _bucket_sizes:
                            _cg_stage = f"capture:graph_b_layer{i}_bucket{bucket_len}"
                            _scores_new_buf = _scores_new_bufs[bucket_len]
                            # Per-layer mask slice: [1, bucket_len] broadcasts
                            # over H_q. Captured by address, so per-step updates
                            # to _scores_new_mask_bufs[bucket_len][i] are visible
                            # on replay.
                            _scores_new_mask_buf_i = _scores_new_mask_bufs[bucket_len][i]
                            _ib_evs = (_ib_make_events_for_layer(bucket_len, i)
                                       if _INTRA_B_TIMING_ENABLED else None)
                            gb = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(gb, stream=_cg_stream):
                                if _ib_evs is not None: _ib_evs[0].record()
                                _scores_new_3d = _scores_new_buf.view(_H_kv, _GQA, bucket_len)
                                _new_k_full = new_kv_buf[i][0, :, :bucket_len, :_D]
                                _new_v_full = new_kv_buf[i][0, :, :bucket_len, _D:]
                                torch.bmm(
                                    _q_2d_buf.view(_H_kv, _GQA, _D),
                                    _new_k_full.transpose(-1, -2),
                                    out=_scores_new_3d,
                                )
                                _scores_new_buf.add_(_scores_new_mask_buf_i)
                                # Per-head joint-knapsack: static gap mask for
                                # FP16 new-zone prefill. Covers only
                                # [n_fp16_h, max_n_fp16); decode region past
                                # max_n_fp16 is unaffected (mask width = max_n_fp16).
                                if sp.fp16_gap_mask is not None and sp.fp16_gap_mask.shape[1] > 0:
                                    _mw = sp.fp16_gap_mask.shape[1]
                                    _scores_new_buf[:, :_mw].add_(sp.fp16_gap_mask)
                                _all_scores_buf = _all_scores_bufs[i][:, :T_old + bucket_len]
                                if T_old > 0:
                                    _all_scores_buf[:, :T_old].copy_(_scores_old_bufs[i][:, :T_old])
                                _all_scores_buf[:, T_old:T_old + bucket_len].copy_(_scores_new_buf)
                                # Per-head joint-knapsack: static softmax_mask on
                                # old-zone (compressed) scores — zeros weight at
                                # per-head padded tail positions [T_eff_h, max_T_eff).
                                if sp.softmax_mask is not None and T_old > 0:
                                    _all_scores_buf[:, :T_old].add_(sp.softmax_mask)
                                if _ib_evs is not None: _ib_evs[1].record()  # end B_qk
                                _all_w = F.softmax(_all_scores_buf, dim=-1)
                                if _ib_evs is not None: _ib_evs[2].record()  # end B_softmax
                                # Attention output accumulate: Zone A+B → C.
                                # Zone A+B V: single fused Triton kernel call.
                                # Reads Zone A (quantized) at cols [0, T_v_h) and
                                # Zone B (FP16 v=16) at cols [T_v, T_v+n_v16_h).
                                if T_old > 0:
                                    v_dequant_weighted_sum_dispatch_into(
                                        _all_w[:, :T_old], sp, _out_old_buf,
                                        V16=_V16_bufs[i],
                                        n_v16_per_head=_n_v16_ph_bufs[i],
                                        max_T_v=T_v,
                                    )
                                else:
                                    _out_old_buf.zero_()
                                # Zone C V (new-kv bmm; consumes [T_old, T_old+bucket_len) cols).
                                torch.bmm(
                                    _all_w[:, T_old:T_old + bucket_len].view(_H_kv, _GQA, bucket_len),
                                    _new_v_full,
                                    out=_out_new_buf.view(_H_kv, _GQA, _D),
                                )
                                _out_old_buf.add_(_out_new_buf)
                                if _ib_evs is not None: _ib_evs[3].record()  # end B_attn_out (A+B+C done)
                                _attn = _out_old_buf.to(model_dtype).view(1, 1, W.attn_output_dim)
                                _attn = F.linear(_attn, lw.o_proj_weight)
                                _hb = _residual_buf + _attn
                                _res2 = _hb.clone()
                                _hb = _rms_norm(_hb, lw.post_ln_weight, lw.post_ln_eps)
                                if _ib_evs is not None: _ib_evs[4].record()  # end B_oproj
                                _gu = F.linear(_hb, lw.gate_up_weight)
                                _g, _u = _gu.split(lw.gate_up_split_sizes, dim=-1)
                                _hb = F.linear(F.silu(_g) * _u, lw.down_proj_weight)
                                _h_buf.copy_(_res2 + _hb)
                                if _ib_evs is not None: _ib_evs[5].record()  # end B_mlp
                            _graphs_b[bucket_len][i] = gb

                    # -- Graph C: final norm → LM head → argmax --
                    _cg_stage = "capture:graph_c"
                    _graph_c = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_graph_c, stream=_cg_stream):
                        _hc = _rms_norm(_h_buf, W.final_norm_weight, W.final_norm_eps)
                        _logits = F.linear(_hc, W.lm_head_weight)
                        _next_token_buf.copy_(_logits[:, -1, :].argmax(dim=-1, keepdim=True))

                    torch.cuda.current_stream().wait_stream(_cg_stream)
                    _cg_ok = True
                    # Persist on dual_cache for reuse across calls
                    dual_cache._cg_cache = {
                        'h_buf': _h_buf, 'residual_buf': _residual_buf,
                        'cos_half_buf': _cos_half_buf, 'sin_half_buf': _sin_half_buf,
                        'q_2d_buf': _q_2d_buf, 'kv_2d_buf': _kv_2d_buf,
                        'w_old_buf': _w_old_buf, 'out_new_buf': _out_new_buf,
                        'out_old_buf': _out_old_buf,
                        'all_scores_bufs': _all_scores_bufs,
                        'scores_old_bufs': _scores_old_bufs,
                        'bucket_sizes': _bucket_sizes,
                        'scores_new_bufs': _scores_new_bufs,
                        'scores_new_mask_bufs': _scores_new_mask_bufs,
                        'mask_col_ids': _mask_col_ids,
                        'mask_valid_buf': _mask_valid_buf,
                        'n_existing_buf': _n_existing_buf,
                        'next_token_buf': _next_token_buf,
                        'graphs_a': _graphs_a, 'graphs_b': _graphs_b, 'graph_c': _graph_c,
                        # Composite invalidation key (matches _cg_key built at
                        # function entry). Captures per-layer T_old, T_v,
                        # n_v16, and topology; any change recaptures.
                        'cg_key': _cg_key,
                        # Legacy field kept for backwards compat / introspection.
                        'n_existing_per_layer': tuple(n_existing_per_layer),
                    }
                except Exception as e:
                    warnings.warn(
                        f"CUDA Graph capture failed at {_cg_stage}: {e}. Falling back to eager."
                    )
                    _cg_ok = False

            # ====== CUDA Graph replay path ======
            # step >= 1 after fresh capture, or step >= 0 when reusing cached graphs
            if _cg_ok and (step >= 1 or _cg is not None):
                _h_buf.copy_(h)
                _cos_half_buf.copy_(cos_half)
                _sin_half_buf.copy_(sin_half)
                # Per-layer T_new = n_existing_per_layer[i] + n_gen. Use max
                # across layers to pick a single bucket; per-layer masks zero
                # out positions beyond each layer's own T_new+1.
                T_new_max_plus_1 = max_n_existing + n_gen + 1
                bucket_len = next(
                    (b for b in _bucket_sizes if b >= T_new_max_plus_1),
                    _bucket_sizes[-1],
                )
                # Batched mask update: compute per-layer valid prefix fully
                # on GPU (one clamp), then one fill_ + one masked_fill_ inside
                # the helper. Total 3 launches regardless of L.
                torch.clamp(
                    _n_existing_buf + (n_gen + 1),
                    max=bucket_len,
                    out=_mask_valid_buf,
                )
                _update_new_scores_mask_batched(
                    _scores_new_mask_bufs[bucket_len],
                    _mask_valid_buf,
                    _mask_col_ids[bucket_len],
                )

                _gt_active = (_GRAPH_TIMING_ENABLED and
                              _GRAPH_TIMING_STEP_LO <= step <= _GRAPH_TIMING_STEP_HI)
                _gt_step_t0 = _gt_step_begin() if _gt_active else 0.0

                for i in range(_n_layers):
                    # -- Graph A replay --
                    if _prof: torch.cuda.nvtx.range_push(f"cg_A_{i}")
                    if _gt_active:
                        t0 = time.perf_counter()
                        ev_idx, ev_end = _gt_record(f"cg_A_{i}", t0)
                        _graphs_a[i].replay()
                        cpu_us = (time.perf_counter() - t0) * 1e6
                        ev_end.record()
                        _gt_pending_finish(f"cg_A_{i}", cpu_us, ev_idx)
                    else:
                        _graphs_a[i].replay()
                    if _prof: torch.cuda.nvtx.range_pop()

                    # Append the current token before replaying Graph B.
                    # Per-layer write position: layers with smaller n_existing
                    # write closer to the start of their buffer. This index is
                    # a Python int — not captured — so per-layer values are
                    # visible to each layer's append assignment.
                    write_pos_i = n_existing_per_layer[i] + n_gen
                    if _prof: torch.cuda.nvtx.range_push(f"cg_append_{i}")
                    if _gt_active:
                        t0 = time.perf_counter()
                        ev_idx, ev_end = _gt_record(f"cg_append_{i}", t0)
                        new_kv_buf[i][0, :, write_pos_i, :] = _kv_2d_buf
                        cpu_us = (time.perf_counter() - t0) * 1e6
                        ev_end.record()
                        _gt_pending_finish(f"cg_append_{i}", cpu_us, ev_idx)
                    else:
                        new_kv_buf[i][0, :, write_pos_i, :] = _kv_2d_buf
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Graph B replay --
                    if _prof: torch.cuda.nvtx.range_push(f"cg_B_{i}")
                    if _gt_active:
                        t0 = time.perf_counter()
                        ev_idx, ev_end = _gt_record(f"cg_B_{i}", t0)
                        _graphs_b[bucket_len][i].replay()
                        cpu_us = (time.perf_counter() - t0) * 1e6
                        ev_end.record()
                        _gt_pending_finish(f"cg_B_{i}", cpu_us, ev_idx)
                    else:
                        _graphs_b[bucket_len][i].replay()
                    if _prof: torch.cuda.nvtx.range_pop()

                # -- Graph C replay --
                if _prof: torch.cuda.nvtx.range_push("cg_C")
                if _gt_active:
                    t0 = time.perf_counter()
                    ev_idx, ev_end = _gt_record("cg_C", t0)
                    _graph_c.replay()
                    cpu_us = (time.perf_counter() - t0) * 1e6
                    ev_end.record()
                    _gt_pending_finish("cg_C", cpu_us, ev_idx)
                else:
                    _graph_c.replay()
                if _prof: torch.cuda.nvtx.range_pop()
                next_token = _next_token_buf

                if _gt_active:
                    _gt_step_end(_gt_step_t0)

                # Intra-Graph-B segment collection (independent of OBKV_GRAPH_TIMING).
                if (_INTRA_B_TIMING_ENABLED and
                        _GRAPH_TIMING_STEP_LO <= step <= _GRAPH_TIMING_STEP_HI):
                    _ib_collect(bucket_len)

            else:
                # ====== Eager path (step 0 or fallback) ======
                for i in range(W.num_layers):
                    lw = W.layers[i]
                    sp = squeezed[i]
                    if _prof: torch.cuda.nvtx.range_push(f"layer_{i}")

                    # Multi-device: route h to this layer's device AND set
                    # this device as the active CUDA context. Triton kernel
                    # launches use the active context — without this, a
                    # cuda:1 tensor launched while context is cuda:0 raises
                    # "Pointer argument cannot be accessed from Triton".
                    # Single-GPU: set_device to the only cuda device is a
                    # no-op (it's already current).
                    if lw.device.type == "cuda":
                        torch.cuda.set_device(lw.device)
                    if h.device != lw.device:
                        h = h.to(lw.device).contiguous()
                    cos_half_d, sin_half_d = _rope_for_dev(lw.device)

                    # -- Input LayerNorm (inline) --
                    if _prof: torch.cuda.nvtx.range_push("input_layernorm")
                    _d = _diag_tic()
                    residual = h
                    h = _rms_norm(h, lw.input_ln_weight, lw.input_ln_eps)
                    _diag_toc("input_layernorm", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Q/K/V projection (fused or split based on lw.qkv_weight) --
                    if _prof: torch.cuda.nvtx.range_push("qkv_proj")
                    _d = _diag_tic()
                    if lw.qkv_weight is not None:
                        qkv = F.linear(h, lw.qkv_weight, lw.qkv_bias)
                        q, k, v = qkv.split(lw.qkv_split_sizes, dim=-1)
                    else:
                        q = F.linear(h, lw.q_proj_weight, lw.q_proj_bias)
                        k = F.linear(h, lw.k_proj_weight, lw.k_proj_bias)
                        v = F.linear(h, lw.v_proj_weight, lw.v_proj_bias)
                    _diag_toc("qkv_proj", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    q_2d = q.view(W.num_heads, W.head_dim)
                    k_2d = k.view(W.num_kv_heads, W.head_dim)
                    v_2d = v.view(W.num_kv_heads, W.head_dim)

                    if lw.q_norm_weight is not None:
                        q_2d = _rms_norm(q_2d, lw.q_norm_weight, lw.qk_norm_eps)
                    if lw.k_norm_weight is not None:
                        k_2d = _rms_norm(k_2d, lw.k_norm_weight, lw.qk_norm_eps)

                    # -- RoPE --
                    if _prof: torch.cuda.nvtx.range_push("rope")
                    _d = _diag_tic()
                    if os.environ.get("OBKV_DEBUG", "0") == "1":
                        # OBKV_DEBUG sentinel: catch CPU/multi-device tensor leaks
                        # before they hit Triton's opaque "Pointer argument cannot
                        # be accessed" error. Prints layer index and per-tensor
                        # devices on first mismatch and falls through to the
                        # kernel for full traceback.
                        if not (q_2d.device == k_2d.device == cos_half_d.device
                                == sin_half_d.device == lw.device):
                            print(
                                f"[OBKV_DEBUG] layer={i} step={step} "
                                f"lw.device={lw.device} h.device={h.device} "
                                f"q_2d.device={q_2d.device} k_2d.device={k_2d.device} "
                                f"cos_half_d.device={cos_half_d.device} "
                                f"sin_half_d.device={sin_half_d.device} "
                                f"q.is_contig={q_2d.is_contiguous()} "
                                f"q_proj_w.device={lw.q_proj_weight.device} "
                                f"input_ln_w.device={lw.input_ln_weight.device} "
                                f"embed_dev={W.embed_dev} rotary_dev={W.rotary_dev}",
                                flush=True,
                            )
                    fused_rope(q_2d, k_2d, cos_half_d, sin_half_d)
                    _diag_toc("rope", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- TriZone three-zone attention (方案 1) --
                    # Zone A: compressed tokens in packed (T_v cols, V Triton kernel).
                    # Zone B: prefill v=16 tokens — K in packed, V in new_v_only_buf[i].
                    # Zone C: decode-appended tokens — K/V in new_kv_buf[i].
                    if _prof: torch.cuda.nvtx.range_push("triton_k_kernel")
                    _d = _diag_tic()
                    T_v = sp.T_eff                                   # Zone A K length
                    T_old = sp.T_eff_k if sp.T_eff_k > 0 else sp.T_eff   # Zone A+B K length
                    n_v16_i = T_old - T_v                            # Zone B K length
                    q_2d_f32 = q_2d.float()
                    if T_old > 0:
                        # K kernel covers packed K stream = T_v compressed + n_v16 FP16-V tokens.
                        scores_old = k_mixed_qk_dot(q_2d_f32, sp, T_old)
                    else:
                        scores_old = None
                    _diag_toc("triton_k_kernel", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Zone C write + QK --
                    # TriZone decode buffer is decode-only, so write_pos = n_gen.
                    write_pos_i = n_gen
                    _d = _diag_tic()
                    kv_2d = torch.cat([k_2d, v_2d], dim=-1)  # [H_kv, 2*D]
                    new_kv_buf[i][0, :, write_pos_i, :] = kv_2d
                    _diag_toc("new_kv_write", _d)
                    T_new_plus_1 = n_gen + 1

                    if _prof: torch.cuda.nvtx.range_push("new_cache_attn")
                    _d = _diag_tic()
                    K_new = new_kv_buf[i][0, :, :T_new_plus_1, :W.head_dim]  # f32
                    q_gqa = q_2d_f32.view(W.num_kv_heads, W.num_kv_groups, W.head_dim) * _INV_SQRT_D
                    scores_new = torch.bmm(q_gqa, K_new.transpose(-1, -2))
                    scores_new = scores_new.reshape(W.num_heads, -1)
                    _diag_toc("new_cache_qk_bmm", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # Per-head joint-knapsack mask. Under TriZone this covers
                    # both Zone A padding and Zone B stripe padding (Phase 1e
                    # extends softmax_mask to [H_q, max_T_eff_k]). Global
                    # prefill path leaves softmax_mask as None — no-op below.
                    # ``fp16_gap_mask`` is deprecated: v=16 padding is folded
                    # into softmax_mask under TriZone.
                    if sp.softmax_mask is not None and scores_old is not None:
                        scores_old.add_(sp.softmax_mask)

                    # -- Global softmax over [Zone A | Zone B | Zone C] --
                    if _prof: torch.cuda.nvtx.range_push("softmax")
                    _d = _diag_tic()
                    if scores_old is not None:
                        all_scores = torch.cat([scores_old, scores_new], dim=-1)
                    else:
                        all_scores = scores_new
                    all_w = F.softmax(all_scores, dim=-1)
                    w_old = all_w[:, :T_old]         # Zone A + Zone B (fused V kernel)
                    w_new = all_w[:, T_old:]         # Zone C
                    _diag_toc("softmax", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Zone A+B V via fused Triton kernel --
                    if _prof: torch.cuda.nvtx.range_push("triton_v_kernel")
                    _d = _diag_tic()
                    if T_old > 0:
                        out_old = v_dequant_weighted_sum_dispatch(
                            w_old, sp,
                            V16=_V16_bufs[i],
                            n_v16_per_head=_n_v16_ph_bufs[i],
                            max_T_v=T_v,
                        )
                    else:
                        out_old = torch.zeros(
                            (W.num_heads, W.head_dim),
                            dtype=torch.float32, device=lw.device,
                        )
                    _diag_toc("triton_v_kernel", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Zone C V + Sum + O proj --
                    if _prof: torch.cuda.nvtx.range_push("output_accumulate")
                    _d = _diag_tic()
                    V_new = new_kv_buf[i][0, :, :T_new_plus_1, W.head_dim:]  # f32
                    w_gqa = w_new.view(W.num_kv_heads, W.num_kv_groups, T_new_plus_1)
                    out_new = torch.bmm(w_gqa, V_new)
                    out_new = out_new.reshape(W.num_heads, W.head_dim)
                    out_old.add_(out_new)
                    _diag_toc("new_cache_v_bmm", _d)
                    _d = _diag_tic()
                    attn_output = out_old.to(model_dtype).view(1, 1, W.attn_output_dim)
                    attn_output = F.linear(attn_output, lw.o_proj_weight)
                    h = residual + attn_output
                    _diag_toc("o_proj", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    # -- Post-attention LayerNorm + MLP --
                    if _prof: torch.cuda.nvtx.range_push("post_ln_mlp")
                    _d = _diag_tic()
                    residual = h
                    h = _rms_norm(h, lw.post_ln_weight, lw.post_ln_eps)
                    _diag_toc("post_layernorm", _d)
                    _d = _diag_tic()
                    if lw.gate_up_weight is not None:
                        gate_up = F.linear(h, lw.gate_up_weight)
                        gate, up = gate_up.split(lw.gate_up_split_sizes, dim=-1)
                    else:
                        gate = F.linear(h, lw.gate_proj_weight)
                        up = F.linear(h, lw.up_proj_weight)
                    h = F.linear(F.silu(gate) * up, lw.down_proj_weight)
                    h = residual + h
                    _diag_toc("mlp", _d)
                    if _prof: torch.cuda.nvtx.range_pop()

                    if _prof: torch.cuda.nvtx.range_pop()  # close layer_{i}

                # === Final Norm + LM Head (route to norm_dev / lm_dev) ===
                _d = _diag_tic()
                if W.norm_dev is not None and W.norm_dev.type == "cuda":
                    torch.cuda.set_device(W.norm_dev)
                if h.device != W.norm_dev:
                    h = h.to(W.norm_dev)
                h = _rms_norm(h, W.final_norm_weight, W.final_norm_eps)
                _diag_toc("final_norm", _d)
                _d = _diag_tic()
                if W.lm_dev is not None and W.lm_dev.type == "cuda":
                    torch.cuda.set_device(W.lm_dev)
                if h.device != W.lm_dev:
                    h = h.to(W.lm_dev)
                logits = F.linear(h, W.lm_head_weight)
                if collected_logits is not None:
                    # [1, 1, V] -> [V] FP32 CPU. Clone before cast so later
                    # argmax on the same tensor is unaffected.
                    collected_logits.append(
                        logits[:, -1, :].detach().float().clone().cpu()
                    )
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                _diag_toc("lm_head_argmax", _d)

            if _prof: torch.cuda.nvtx.range_pop()  # close decode_step
            if _k_diag_on and step == _k_diag_target_step:
                snap = k_diag_snapshot()
                print(
                    f"[K_DIAG step={step}] n_calls={snap['n_calls']} "
                    f"index_select_total={snap['index_select_ms']:.3f}ms "
                    f"kernel_total={snap['kernel_ms']:.3f}ms",
                    flush=True,
                )
            if _DECODE_DIAG_ENABLED and step >= 1:
                _diag_steps_timed += 1
            n_gen += 1

    _diag_summary(f"greedy_decode_fast eager-path "
                  f"({_diag_steps_timed} timed steps, skipped step 0)")
    _gt_summary(f"greedy_decode_fast CG-replay path "
                f"(steps {_GRAPH_TIMING_STEP_LO}..{_GRAPH_TIMING_STEP_HI})")
    _ib_summary(f"Graph B intra-segment timing "
                f"(steps {_GRAPH_TIMING_STEP_LO}..{_GRAPH_TIMING_STEP_HI})")

    # Token-return contract is unchanged by the logit side channel (I21 —
    # tokens match between return_logits=False and return_logits=True paths).
    if return_logits:
        return generated_ids, collected_logits
    return generated_ids
