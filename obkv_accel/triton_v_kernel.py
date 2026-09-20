"""
Triton kernels for V-side: fused N-bit dequantization + weighted sum.

Three separate kernels for 2-bit, 4-bit, and 8-bit segments, plus a Python
dispatch function that splits attention weights by segment and sums results.

V packing layouts (HALF-SPLIT / QUARTER-SPLIT, matching compress_function.py):

  4-bit (HALF-SPLIT):
    byte[i] = val[i] | (val[i+64] << 4)        for i in [0, 64)
    Low nibble  (& 0xF) -> channels 0-63
    High nibble (>> 4)  -> channels 64-127

  2-bit (QUARTER-SPLIT):
    byte[i] = val[i] | (val[i+32]<<2) | (val[i+64]<<4) | (val[i+96]<<6)
                                                 for i in [0, 32)
    bits 0-1 (& 0x3)        -> channels 0-31
    bits 2-3 ((>>2) & 0x3)  -> channels 32-63
    bits 4-5 ((>>4) & 0x3)  -> channels 64-95
    bits 6-7 ((>>6) & 0x3)  -> channels 96-127

  8-bit: no packing, direct uint8.

V quantization: per-token (scale/zp broadcast over channels).
  V_scale: [H_kv, T_seg, 1]  FP16
  V_zp:    [H_kv, T_seg, 1]  FP16

Dequant formula:  v_fp = (v_uint.float() - zp) * scale

GQA mapping: h_kv = h_q // GQA_FACTOR  (Llama-3.1-8B: 32 Q heads, 8 KV heads)
             attn_weights are indexed by h_q; V data by h_kv.
"""

import math
import os
import torch

# Env-tunable V-kernel launch params (helps A/B num_warps / num_stages sweeps
# without editing code). Defaults match the previous hard-coded values.
_V_NUM_WARPS = int(os.environ.get("OBKV_V_NUM_WARPS", "8"))
_V_NUM_STAGES = int(os.environ.get("OBKV_V_NUM_STAGES", "4"))
# D-quarter parallelism: when 1 (default), launch the unified V kernel with
# grid=(H_q, 4) so 4 programs per Q-head split the D=128 output channels.
# Each program holds 1 accumulator per segment instead of 4, freeing register
# pressure and lifting A100 SM occupancy from ~30% (32 programs) to ~100%
# (128 programs > 108 SMs). Set OBKV_V_DQUARTER=0 to fall back to the legacy
# (H_q,) kernel for A/B comparison.
_V_DQUARTER = int(os.environ.get("OBKV_V_DQUARTER", "1")) == 1

# E2: FP16 tensor-core V kernel. When 1, wrapper dispatches to the
# `*_tc_kernel` which uses `tl.dot(mma.m16n8k16)` on a 4-GQA + 12-pad M=16
# tile, split-T accumulator with atomic_add writeback. A100 SM 80 compatible.
# Default ON as of 2026-04-24 after GH200 gates green (TPOT 18.56->10.39 ms/tok
# on 128K streaming); set OBKV_V_TC=0 to revert to the scalar path.
_V_TC = int(os.environ.get("OBKV_V_TC", "1")) == 1
# Split-T granularity (programs per H_kv per D-block). V1 default 16.
_V_TC_T_CHUNKS = int(os.environ.get("OBKV_V_TC_T_CHUNKS", "16"))

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 128
# GQA_FACTOR is the legacy module-level default kept for backward compat
# (Llama-3.1-8B / Mistral-7B / Qwen3-4B are GQA=4). Wrappers below derive
# the *real* value at runtime from V_packed.shape so MHA and other GQA
# ratios (Llama-2-13B GQA=1, Llama-3-70B GQA=8) work correctly.
GQA_FACTOR = 4  # legacy constant; do NOT use in new wrappers


def _runtime_gqa(attn_weights, V_scale):
    """Infer the GQA factor at runtime from input shapes.

    attn_weights : [H_q, T_seg] or higher rank — first dim is H_q
    V_scale      : [H_kv, T_seg, 1] (3D, post-squeeze) or
                   [1, H_kv, T_seg, 1] (4D, raw build_packed_layer output)
    Using V_scale instead of V_packed because the packed segment tensors
    may be None (per-segment optionals); V_scale is always present.
    """
    H_q = attn_weights.shape[0]
    H_kv = V_scale.shape[1] if V_scale.dim() == 4 else V_scale.shape[0]
    return H_q // H_kv


# Cache for zero-sized dummy tensors (avoids repeated torch.empty calls)
_dummy_cache = {}

def _get_dummy_tensors(device, H_kv):
    """Return cached zero-sized dummy tensors for missing V segments."""
    key = (str(device), H_kv)
    if key not in _dummy_cache:
        _dummy_cache[key] = (
            torch.empty((H_kv, 0, D // 4), dtype=torch.uint8, device=device),
            torch.empty((H_kv, 0, D // 2), dtype=torch.uint8, device=device),
            torch.empty((H_kv, 0, D),      dtype=torch.uint8, device=device),
        )
    return _dummy_cache[key]


# Cache for Zone B dummies (V16 FP16 value + per-head n_v16 zeros).
_v16_dummy_cache = {}

def _get_v16_dummy(device, H_kv):
    """Return cached dummy V16 tensor + per-head n_v16=0 tensor.

    Used when the caller has no Zone B (legacy DualZone or pre-TriZone pack);
    the kernel still needs valid pointers but the loop iterates 0 times.
    """
    key = (str(device), H_kv)
    if key not in _v16_dummy_cache:
        _v16_dummy_cache[key] = (
            torch.empty((H_kv, 1, D), dtype=torch.float16, device=device),
            torch.zeros((H_kv,), dtype=torch.int32, device=device),
        )
    return _v16_dummy_cache[key]

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if HAS_TRITON:

    # ===================================================================
    # 4-bit V kernel
    # ===================================================================
    @triton.jit
    def _v_weighted_sum_4bit_kernel(
        # Pointers
        W_ptr,         # [H_q, T_seg]             FP32  (attn weights)
        V_packed_ptr,  # [H_kv, T_seg, D_PACKED]  uint8  (D_PACKED = 64)
        V_scale_ptr,   # [H_kv, T_seg, 1]         FP16
        V_zp_ptr,      # [H_kv, T_seg, 1]         FP16
        Out_ptr,       # [H_q, D]                 FP32  (output)
        # Dimensions
        T_seg,
        # Strides -- W: [H_q, T_seg]
        stride_w_h,
        stride_w_t: tl.constexpr,
        # Strides -- V_packed: [H_kv, T_seg, D_PACKED]
        stride_vp_h,
        stride_vp_t,
        stride_vp_d: tl.constexpr,
        # Strides -- V_scale / V_zp: [H_kv, T_seg, 1]
        stride_vs_h,
        stride_vs_t,
        # Strides -- Out: [H_q, D]
        stride_o_h,
        stride_o_d: tl.constexpr,
        # Constexprs
        BLOCK_T: tl.constexpr,
        D_PACKED_C: tl.constexpr,   # = 64
        D_C: tl.constexpr,          # = 128
        GQA_FACTOR_C: tl.constexpr, # = 4
    ):
        pid_hq = tl.program_id(0)
        h_kv = pid_hq // GQA_FACTOR_C

        # Accumulator for output: [D_C] = [128]
        # We accumulate lo-half and hi-half separately then store concatenated
        d_lo = tl.arange(0, D_PACKED_C)   # [0..63]
        acc_lo = tl.zeros((D_PACKED_C,), dtype=tl.float32)
        acc_hi = tl.zeros((D_PACKED_C,), dtype=tl.float32)

        for t_start in range(0, T_seg, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < T_seg

            # Load attention weights: [BLOCK_T]
            w = tl.load(W_ptr + pid_hq * stride_w_h + t_offs * stride_w_t,
                        mask=t_mask, other=0.0)

            # Load V_packed: [BLOCK_T, D_PACKED_C] uint8
            vp_ptrs = (V_packed_ptr
                       + h_kv * stride_vp_h
                       + t_offs[:, None] * stride_vp_t
                       + d_lo[None, :] * stride_vp_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0)
            packed_i32 = packed.to(tl.int32)

            # Unpack half-split
            v_lo_uint = packed_i32 & 0xF           # channels 0-63
            v_hi_uint = (packed_i32 >> 4) & 0xF    # channels 64-127

            v_lo_f = v_lo_uint.to(tl.float32)  # [BLOCK_T, D_PACKED_C]
            v_hi_f = v_hi_uint.to(tl.float32)

            # Load scale and zp: [BLOCK_T]
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            # Dequant: v_dq = (v_uint - zp) * scale   (per-token: broadcast over D)
            v_lo_dq = (v_lo_f - zp[:, None]) * scale[:, None]  # [BLOCK_T, D_PACKED_C]
            v_hi_dq = (v_hi_f - zp[:, None]) * scale[:, None]

            # Weighted accumulate: acc += sum_t (w[t] * v_dq[t, d])
            acc_lo += tl.sum(w[:, None] * v_lo_dq, axis=0)
            acc_hi += tl.sum(w[:, None] * v_hi_dq, axis=0)

        # Store: channels [0..63] then [64..127]
        d_full = tl.arange(0, D_C)
        out_ptrs = Out_ptr + pid_hq * stride_o_h + d_full * stride_o_d
        # Build contiguous [D_C] output
        # acc_lo covers d_full[0:64], acc_hi covers d_full[64:128]
        out_lo_ptrs = Out_ptr + pid_hq * stride_o_h + d_lo * stride_o_d
        out_hi_ptrs = Out_ptr + pid_hq * stride_o_h + (d_lo + D_PACKED_C) * stride_o_d
        tl.store(out_lo_ptrs, acc_lo)
        tl.store(out_hi_ptrs, acc_hi)

    # ===================================================================
    # 2-bit V kernel
    # ===================================================================
    @triton.jit
    def _v_weighted_sum_2bit_kernel(
        W_ptr,
        V_packed_ptr,  # [H_kv, T_seg, D//4=32] uint8
        V_scale_ptr,
        V_zp_ptr,
        Out_ptr,
        T_seg,
        stride_w_h, stride_w_t: tl.constexpr,
        stride_vp_h, stride_vp_t, stride_vp_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,  # = 32
        D_C: tl.constexpr,          # = 128
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        h_kv = pid_hq // GQA_FACTOR_C

        d_q = tl.arange(0, D_QUARTER_C)  # [0..31]
        acc_q0 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 0-31
        acc_q1 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 32-63
        acc_q2 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 64-95
        acc_q3 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 96-127

        for t_start in range(0, T_seg, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < T_seg

            w = tl.load(W_ptr + pid_hq * stride_w_h + t_offs * stride_w_t,
                        mask=t_mask, other=0.0)

            # Load V_packed: [BLOCK_T, D_QUARTER_C] uint8
            vp_ptrs = (V_packed_ptr
                       + h_kv * stride_vp_h
                       + t_offs[:, None] * stride_vp_t
                       + d_q[None, :] * stride_vp_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0)
            packed_i32 = packed.to(tl.int32)

            # Unpack quarter-split
            v_q0 = (packed_i32 & 0x3).to(tl.float32)           # channels 0-31
            v_q1 = ((packed_i32 >> 2) & 0x3).to(tl.float32)    # channels 32-63
            v_q2 = ((packed_i32 >> 4) & 0x3).to(tl.float32)    # channels 64-95
            v_q3 = ((packed_i32 >> 6) & 0x3).to(tl.float32)    # channels 96-127

            # Load scale, zp: [BLOCK_T]
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            # Dequant + weighted accumulate
            v_q0_dq = (v_q0 - zp[:, None]) * scale[:, None]
            v_q1_dq = (v_q1 - zp[:, None]) * scale[:, None]
            v_q2_dq = (v_q2 - zp[:, None]) * scale[:, None]
            v_q3_dq = (v_q3 - zp[:, None]) * scale[:, None]

            acc_q0 += tl.sum(w[:, None] * v_q0_dq, axis=0)
            acc_q1 += tl.sum(w[:, None] * v_q1_dq, axis=0)
            acc_q2 += tl.sum(w[:, None] * v_q2_dq, axis=0)
            acc_q3 += tl.sum(w[:, None] * v_q3_dq, axis=0)

        # Store [128]: channels 0-31, 32-63, 64-95, 96-127
        base = Out_ptr + pid_hq * stride_o_h
        tl.store(base + d_q * stride_o_d, acc_q0)
        tl.store(base + (d_q + D_QUARTER_C) * stride_o_d, acc_q1)
        tl.store(base + (d_q + 2 * D_QUARTER_C) * stride_o_d, acc_q2)
        tl.store(base + (d_q + 3 * D_QUARTER_C) * stride_o_d, acc_q3)

    # ===================================================================
    # 8-bit V kernel
    # ===================================================================
    @triton.jit
    def _v_weighted_sum_8bit_kernel(
        W_ptr,
        V_ptr,        # [H_kv, T_seg, D] uint8  (no packing)
        V_scale_ptr,
        V_zp_ptr,
        Out_ptr,
        T_seg,
        stride_w_h, stride_w_t: tl.constexpr,
        stride_v_h, stride_v_t, stride_v_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_C: tl.constexpr,           # = 128
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        h_kv = pid_hq // GQA_FACTOR_C

        d_offs = tl.arange(0, D_C)
        acc = tl.zeros((D_C,), dtype=tl.float32)

        for t_start in range(0, T_seg, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < T_seg

            w = tl.load(W_ptr + pid_hq * stride_w_h + t_offs * stride_w_t,
                        mask=t_mask, other=0.0)

            # Load V: [BLOCK_T, D_C] uint8
            v_ptrs = (V_ptr
                      + h_kv * stride_v_h
                      + t_offs[:, None] * stride_v_t
                      + d_offs[None, :] * stride_v_d)
            v_uint = tl.load(v_ptrs, mask=t_mask[:, None], other=0)
            v_f = v_uint.to(tl.float32)

            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            v_dq = (v_f - zp[:, None]) * scale[:, None]
            acc += tl.sum(w[:, None] * v_dq, axis=0)

        tl.store(Out_ptr + pid_hq * stride_o_h + d_offs * stride_o_d, acc)

    # ===================================================================
    # Unified V kernel (2-bit + 4-bit + 8-bit in single launch)
    # ===================================================================
    @triton.jit
    def _v_weighted_sum_unified_kernel(
        # Pointers
        W_ptr,           # [H_q, T_eff]                   FP32
        V_2bit_ptr,      # [H_kv, N_2, D//4=32]           uint8
        V_4bit_ptr,      # [H_kv, N_4, D//2=64]           uint8
        V_8bit_ptr,      # [H_kv, N_8, D=128]             uint8
        V_scale_ptr,     # [H_kv, T_eff, 1]               FP16
        V_zp_ptr,        # [H_kv, T_eff, 1]               FP16
        Out_ptr,         # [H_q, D]                        FP32
        # Dimensions
        N_2, N_4, N_8,
        # Strides -- W: [H_q, T_eff]
        stride_w_h,
        stride_w_t: tl.constexpr,
        # Strides -- V_2bit: [H_kv, N_2, D//4]
        stride_v2_h, stride_v2_t, stride_v2_d: tl.constexpr,
        # Strides -- V_4bit: [H_kv, N_4, D//2]
        stride_v4_h, stride_v4_t, stride_v4_d: tl.constexpr,
        # Strides -- V_8bit: [H_kv, N_8, D]
        stride_v8_h, stride_v8_t, stride_v8_d: tl.constexpr,
        # Strides -- V_scale/V_zp: [H_kv, T_eff, 1]
        stride_vs_h, stride_vs_t,
        # Strides -- Out: [H_q, D]
        stride_o_h, stride_o_d: tl.constexpr,
        # Constexprs
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,   # = 32
        D_PACKED_C: tl.constexpr,    # = 64
        D_C: tl.constexpr,           # = 128
        GQA_FACTOR_C: tl.constexpr,  # = 4
    ):
        pid_hq = tl.program_id(0)
        h_kv = pid_hq // GQA_FACTOR_C
        base_out = Out_ptr + pid_hq * stride_o_h

        # ===== 2-bit segment: 4 quarter accumulators =====
        d_q = tl.arange(0, D_QUARTER_C)   # [0..31]
        acc2_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)

        for t_start in range(0, N_2, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_2

            w = tl.load(W_ptr + pid_hq * stride_w_h + t_offs * stride_w_t,
                        mask=t_mask, other=0.0)

            vp_ptrs = (V_2bit_ptr
                       + h_kv * stride_v2_h
                       + t_offs[:, None] * stride_v2_t
                       + d_q[None, :] * stride_v2_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0)
            packed_i32 = packed.to(tl.int32)

            v_q0 = (packed_i32 & 0x3).to(tl.float32)
            v_q1 = ((packed_i32 >> 2) & 0x3).to(tl.float32)
            v_q2 = ((packed_i32 >> 4) & 0x3).to(tl.float32)
            v_q3 = ((packed_i32 >> 6) & 0x3).to(tl.float32)

            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            v_q0_dq = (v_q0 - zp[:, None]) * scale[:, None]
            v_q1_dq = (v_q1 - zp[:, None]) * scale[:, None]
            v_q2_dq = (v_q2 - zp[:, None]) * scale[:, None]
            v_q3_dq = (v_q3 - zp[:, None]) * scale[:, None]

            acc2_a += tl.sum(w[:, None] * v_q0_dq, axis=0)
            acc2_b += tl.sum(w[:, None] * v_q1_dq, axis=0)
            acc2_c += tl.sum(w[:, None] * v_q2_dq, axis=0)
            acc2_d += tl.sum(w[:, None] * v_q3_dq, axis=0)

        # ===== 4-bit segment: 4 quarter accumulators =====
        acc4_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 0-31
        acc4_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 32-63
        acc4_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 64-95
        acc4_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 96-127

        for t_start in range(0, N_4, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_4

            # W offset: skip N_2 tokens in attn_weights
            w = tl.load(W_ptr + pid_hq * stride_w_h + (t_offs + N_2) * stride_w_t,
                        mask=t_mask, other=0.0)

            vp_ptrs_lo = (V_4bit_ptr
                          + h_kv * stride_v4_h
                          + t_offs[:, None] * stride_v4_t
                          + d_q[None, :] * stride_v4_d)
            vp_ptrs_hi = (V_4bit_ptr
                          + h_kv * stride_v4_h
                          + t_offs[:, None] * stride_v4_t
                          + (d_q + D_QUARTER_C)[None, :] * stride_v4_d)
            packed_lo = tl.load(vp_ptrs_lo, mask=t_mask[:, None], other=0).to(tl.int32)
            packed_hi = tl.load(vp_ptrs_hi, mask=t_mask[:, None], other=0).to(tl.int32)

            v_a = (packed_lo & 0xF).to(tl.float32)
            v_b = (packed_hi & 0xF).to(tl.float32)
            v_c = ((packed_lo >> 4) & 0xF).to(tl.float32)
            v_d = ((packed_hi >> 4) & 0xF).to(tl.float32)

            # Scale/zp offset: skip N_2 tokens
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + (t_offs + N_2) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + (t_offs + N_2) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            v_a_dq = (v_a - zp[:, None]) * scale[:, None]
            v_b_dq = (v_b - zp[:, None]) * scale[:, None]
            v_c_dq = (v_c - zp[:, None]) * scale[:, None]
            v_d_dq = (v_d - zp[:, None]) * scale[:, None]

            acc4_a += tl.sum(w[:, None] * v_a_dq, axis=0)
            acc4_b += tl.sum(w[:, None] * v_b_dq, axis=0)
            acc4_c += tl.sum(w[:, None] * v_c_dq, axis=0)
            acc4_d += tl.sum(w[:, None] * v_d_dq, axis=0)

        # ===== 8-bit segment: 4 quarter accumulators =====
        acc8_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 0-31
        acc8_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 32-63
        acc8_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 64-95
        acc8_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)  # channels 96-127

        for t_start in range(0, N_8, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_8

            # W offset: skip N_2 + N_4 tokens
            w = tl.load(W_ptr + pid_hq * stride_w_h + (t_offs + N_2 + N_4) * stride_w_t,
                        mask=t_mask, other=0.0)

            v_ptrs_a = (V_8bit_ptr
                        + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + d_q[None, :] * stride_v8_d)
            v_ptrs_b = (V_8bit_ptr
                        + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + D_QUARTER_C)[None, :] * stride_v8_d)
            v_ptrs_c = (V_8bit_ptr
                        + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + 2 * D_QUARTER_C)[None, :] * stride_v8_d)
            v_ptrs_d = (V_8bit_ptr
                        + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + 3 * D_QUARTER_C)[None, :] * stride_v8_d)
            v_a = tl.load(v_ptrs_a, mask=t_mask[:, None], other=0).to(tl.float32)
            v_b = tl.load(v_ptrs_b, mask=t_mask[:, None], other=0).to(tl.float32)
            v_c = tl.load(v_ptrs_c, mask=t_mask[:, None], other=0).to(tl.float32)
            v_d = tl.load(v_ptrs_d, mask=t_mask[:, None], other=0).to(tl.float32)

            # Scale/zp offset: skip N_2 + N_4 tokens
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + (t_offs + N_2 + N_4) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + (t_offs + N_2 + N_4) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)

            v_a_dq = (v_a - zp[:, None]) * scale[:, None]
            v_b_dq = (v_b - zp[:, None]) * scale[:, None]
            v_c_dq = (v_c - zp[:, None]) * scale[:, None]
            v_d_dq = (v_d - zp[:, None]) * scale[:, None]
            acc8_a += tl.sum(w[:, None] * v_a_dq, axis=0)
            acc8_b += tl.sum(w[:, None] * v_b_dq, axis=0)
            acc8_c += tl.sum(w[:, None] * v_c_dq, axis=0)
            acc8_d += tl.sum(w[:, None] * v_d_dq, axis=0)

        # Final store: combine 2/4/8-bit accumulators in registers and write once.
        tl.store(base_out + d_q * stride_o_d, acc2_a + acc4_a + acc8_a)
        tl.store(
            base_out + (d_q + D_QUARTER_C) * stride_o_d,
            acc2_b + acc4_b + acc8_b,
        )
        tl.store(
            base_out + (d_q + 2 * D_QUARTER_C) * stride_o_d,
            acc2_c + acc4_c + acc8_c,
        )
        tl.store(
            base_out + (d_q + 3 * D_QUARTER_C) * stride_o_d,
            acc2_d + acc4_d + acc8_d,
        )

    # ===================================================================
    # D-quarter parallel unified V kernel
    # ===================================================================
    # Grid (H_q, 4): pid_dq selects one of the 4 D-quarters of the output;
    # each program holds 1 accumulator per segment (3 total) instead of 4×3.
    # 2-bit:  same 32-byte packed load shared across pid_dq, only shift differs.
    # 4-bit:  byte_offset = (pid_dq & 1) * 32, nibble_shift = (pid_dq & 2) * 2.
    # 8-bit:  pointer offset (d_q + pid_dq * 32).
    @triton.jit
    def _v_weighted_sum_unified_dquarter_kernel(
        W_ptr, V_2bit_ptr, V_4bit_ptr, V_8bit_ptr, V_scale_ptr, V_zp_ptr, Out_ptr,
        N_2, N_4, N_8,
        stride_w_h, stride_w_t: tl.constexpr,
        stride_v2_h, stride_v2_t, stride_v2_d: tl.constexpr,
        stride_v4_h, stride_v4_t, stride_v4_d: tl.constexpr,
        stride_v8_h, stride_v8_t, stride_v8_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_dq = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C
        d_q = tl.arange(0, D_QUARTER_C)
        d_q_out_offset = pid_dq * D_QUARTER_C

        base_w = W_ptr + pid_hq * stride_w_h
        base_vs = V_scale_ptr + h_kv * stride_vs_h
        base_vz = V_zp_ptr + h_kv * stride_vs_h

        # ----- 2-bit segment -----
        acc2 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        shift_2bit = pid_dq * 2
        for t_start in range(0, N_2, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_2
            w = tl.load(base_w + t_offs * stride_w_t, mask=t_mask, other=0.0)
            vp_ptrs = (V_2bit_ptr + h_kv * stride_v2_h
                       + t_offs[:, None] * stride_v2_t
                       + d_q[None, :] * stride_v2_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
            v_q = ((packed >> shift_2bit) & 0x3).to(tl.float32)
            scale = tl.load(base_vs + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v_q - zp[:, None]) * scale[:, None]
            acc2 += tl.sum(w[:, None] * v_dq, axis=0)

        # ----- 4-bit segment -----
        acc4 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        byte_offset_4bit = (pid_dq & 1) * D_QUARTER_C
        nibble_shift = (pid_dq & 2) * 2
        for t_start in range(0, N_4, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_4
            w = tl.load(base_w + (t_offs + N_2) * stride_w_t,
                        mask=t_mask, other=0.0)
            vp_ptrs = (V_4bit_ptr + h_kv * stride_v4_h
                       + t_offs[:, None] * stride_v4_t
                       + (d_q + byte_offset_4bit)[None, :] * stride_v4_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
            v_q = ((packed >> nibble_shift) & 0xF).to(tl.float32)
            scale = tl.load(base_vs + (t_offs + N_2) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + (t_offs + N_2) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v_q - zp[:, None]) * scale[:, None]
            acc4 += tl.sum(w[:, None] * v_dq, axis=0)

        # ----- 8-bit segment -----
        acc8 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        for t_start in range(0, N_8, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_8
            w = tl.load(base_w + (t_offs + N_2 + N_4) * stride_w_t,
                        mask=t_mask, other=0.0)
            v_ptrs = (V_8bit_ptr + h_kv * stride_v8_h
                      + t_offs[:, None] * stride_v8_t
                      + (d_q + d_q_out_offset)[None, :] * stride_v8_d)
            v = tl.load(v_ptrs, mask=t_mask[:, None], other=0).to(tl.float32)
            scale = tl.load(base_vs + (t_offs + N_2 + N_4) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + (t_offs + N_2 + N_4) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v - zp[:, None]) * scale[:, None]
            acc8 += tl.sum(w[:, None] * v_dq, axis=0)

        tl.store(Out_ptr + pid_hq * stride_o_h + (d_q + d_q_out_offset) * stride_o_d,
                 acc2 + acc4 + acc8)

    # ===================================================================
    # Per-head D-quarter unified V kernel (joint-knapsack / padded path)
    # ===================================================================
    # Identical to the D-quarter unified kernel except:
    #   - N_2, N_4, N_8 are read per-head from `seg_bounds_ptr: [H_kv, 3]`
    #     at runtime (one 3-element tl.load per program).
    #   - V_2bit/V_4bit/V_8bit rows are padded per-head to max_h(N_i^h).
    #     Stride addresses along T are head-local: (t_offs + N_2_h), etc.
    #     Padded rows are never read (loop bounds use per-head N_i_h).
    #   - V_scale/V_zp rows are padded per-head to max_h(T_eff_h) and addressed
    #     with head-local offsets (t_offs, t_offs + N_2_h, t_offs + N_2_h + N_4_h).
    #   - W is [H_q, max_T_eff] shared layout. Head h's valid weights live at
    #     [0, T_eff_h); padded tail weights were masked to 0 by softmax_mask.
    @triton.jit
    def _v_weighted_sum_unified_dquarter_perhead_kernel(
        W_ptr, V_2bit_ptr, V_4bit_ptr, V_8bit_ptr, V_scale_ptr, V_zp_ptr, Out_ptr,
        seg_bounds_ptr,   # [H_kv, 3] int32
        # Zone B (v=16) params: FP16 stripe fused into the same kernel so we
        # avoid per-layer torch.bmm launches. When HAS_V16=0 the loop is
        # emitted-but-skipped (n_v16_h=0 dummy), costing only the compile.
        V16_ptr,          # [H_kv, max_n_v16, D] FP16
        n_v16_ph_ptr,     # [H_kv] int32  (per-head valid v=16 length)
        MAX_T_V,          # int32  runtime — col offset in W to reach Zone B
        stride_w_h, stride_w_t: tl.constexpr,
        stride_v2_h, stride_v2_t, stride_v2_d: tl.constexpr,
        stride_v4_h, stride_v4_t, stride_v4_d: tl.constexpr,
        stride_v8_h, stride_v8_t, stride_v8_d: tl.constexpr,
        stride_v16_h, stride_v16_t, stride_v16_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,
        GQA_FACTOR_C: tl.constexpr,
        HAS_V16: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_dq = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C
        # Per-head segment counts
        N_2 = tl.load(seg_bounds_ptr + h_kv * 3 + 0)
        N_4 = tl.load(seg_bounds_ptr + h_kv * 3 + 1)
        N_8 = tl.load(seg_bounds_ptr + h_kv * 3 + 2)

        d_q = tl.arange(0, D_QUARTER_C)
        d_q_out_offset = pid_dq * D_QUARTER_C

        base_w = W_ptr + pid_hq * stride_w_h
        base_vs = V_scale_ptr + h_kv * stride_vs_h
        base_vz = V_zp_ptr + h_kv * stride_vs_h

        # ----- 2-bit segment -----
        acc2 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        shift_2bit = pid_dq * 2
        for t_start in range(0, N_2, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_2
            w = tl.load(base_w + t_offs * stride_w_t, mask=t_mask, other=0.0)
            vp_ptrs = (V_2bit_ptr + h_kv * stride_v2_h
                       + t_offs[:, None] * stride_v2_t
                       + d_q[None, :] * stride_v2_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
            v_q = ((packed >> shift_2bit) & 0x3).to(tl.float32)
            scale = tl.load(base_vs + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v_q - zp[:, None]) * scale[:, None]
            acc2 += tl.sum(w[:, None] * v_dq, axis=0)

        # ----- 4-bit segment -----
        acc4 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        byte_offset_4bit = (pid_dq & 1) * D_QUARTER_C
        nibble_shift = (pid_dq & 2) * 2
        for t_start in range(0, N_4, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_4
            w = tl.load(base_w + (t_offs + N_2) * stride_w_t,
                        mask=t_mask, other=0.0)
            vp_ptrs = (V_4bit_ptr + h_kv * stride_v4_h
                       + t_offs[:, None] * stride_v4_t
                       + (d_q + byte_offset_4bit)[None, :] * stride_v4_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
            v_q = ((packed >> nibble_shift) & 0xF).to(tl.float32)
            scale = tl.load(base_vs + (t_offs + N_2) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + (t_offs + N_2) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v_q - zp[:, None]) * scale[:, None]
            acc4 += tl.sum(w[:, None] * v_dq, axis=0)

        # ----- 8-bit segment -----
        acc8 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        for t_start in range(0, N_8, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_8
            w = tl.load(base_w + (t_offs + N_2 + N_4) * stride_w_t,
                        mask=t_mask, other=0.0)
            v_ptrs = (V_8bit_ptr + h_kv * stride_v8_h
                      + t_offs[:, None] * stride_v8_t
                      + (d_q + d_q_out_offset)[None, :] * stride_v8_d)
            v = tl.load(v_ptrs, mask=t_mask[:, None], other=0).to(tl.float32)
            scale = tl.load(base_vs + (t_offs + N_2 + N_4) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(base_vz + (t_offs + N_2 + N_4) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            v_dq = (v - zp[:, None]) * scale[:, None]
            acc8 += tl.sum(w[:, None] * v_dq, axis=0)

        # ----- Zone B FP16 (v=16) segment -----
        # Weights at W[pid_hq, MAX_T_V + t]; V16 is FP16 stored directly, so
        # no dequant — just promote to FP32 in the tl.sum.
        acc16 = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        if HAS_V16:
            N_V16 = tl.load(n_v16_ph_ptr + h_kv)
            for t_start in range(0, N_V16, BLOCK_T):
                t_offs = t_start + tl.arange(0, BLOCK_T)
                t_mask = t_offs < N_V16
                w = tl.load(base_w + (t_offs + MAX_T_V) * stride_w_t,
                            mask=t_mask, other=0.0)
                v_ptrs = (V16_ptr + h_kv * stride_v16_h
                          + t_offs[:, None] * stride_v16_t
                          + (d_q + d_q_out_offset)[None, :] * stride_v16_d)
                v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
                acc16 += tl.sum(w[:, None] * v, axis=0)

        tl.store(Out_ptr + pid_hq * stride_o_h + (d_q + d_q_out_offset) * stride_o_d,
                 acc2 + acc4 + acc8 + acc16)

    # ===================================================================
    # Per-head non-D-quarter unified V kernel (eager path fallback if
    # OBKV_V_DQUARTER=0). Same changes as above applied to the
    # quad-accumulator unified kernel.
    # ===================================================================
    @triton.jit
    def _v_weighted_sum_unified_perhead_kernel(
        W_ptr,
        V_2bit_ptr, V_4bit_ptr, V_8bit_ptr,
        V_scale_ptr, V_zp_ptr,
        Out_ptr,
        seg_bounds_ptr,
        # Zone B FP16 (v=16) stripe — fused to avoid per-layer torch.bmm.
        V16_ptr,          # [H_kv, max_n_v16, D] FP16
        n_v16_ph_ptr,     # [H_kv] int32
        MAX_T_V,          # int32 runtime — W col offset where Zone B begins
        stride_w_h, stride_w_t: tl.constexpr,
        stride_v2_h, stride_v2_t, stride_v2_d: tl.constexpr,
        stride_v4_h, stride_v4_t, stride_v4_d: tl.constexpr,
        stride_v8_h, stride_v8_t, stride_v8_d: tl.constexpr,
        stride_v16_h, stride_v16_t, stride_v16_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,
        D_PACKED_C: tl.constexpr,
        D_C: tl.constexpr,
        GQA_FACTOR_C: tl.constexpr,
        HAS_V16: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        h_kv = pid_hq // GQA_FACTOR_C
        N_2 = tl.load(seg_bounds_ptr + h_kv * 3 + 0)
        N_4 = tl.load(seg_bounds_ptr + h_kv * 3 + 1)
        N_8 = tl.load(seg_bounds_ptr + h_kv * 3 + 2)
        base_out = Out_ptr + pid_hq * stride_o_h

        d_q = tl.arange(0, D_QUARTER_C)
        acc2_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc2_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)

        for t_start in range(0, N_2, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_2
            w = tl.load(W_ptr + pid_hq * stride_w_h + t_offs * stride_w_t,
                        mask=t_mask, other=0.0)
            vp_ptrs = (V_2bit_ptr + h_kv * stride_v2_h
                       + t_offs[:, None] * stride_v2_t
                       + d_q[None, :] * stride_v2_d)
            packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
            v_q0 = (packed & 0x3).to(tl.float32)
            v_q1 = ((packed >> 2) & 0x3).to(tl.float32)
            v_q2 = ((packed >> 4) & 0x3).to(tl.float32)
            v_q3 = ((packed >> 6) & 0x3).to(tl.float32)
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + t_offs * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            acc2_a += tl.sum(w[:, None] * ((v_q0 - zp[:, None]) * scale[:, None]), axis=0)
            acc2_b += tl.sum(w[:, None] * ((v_q1 - zp[:, None]) * scale[:, None]), axis=0)
            acc2_c += tl.sum(w[:, None] * ((v_q2 - zp[:, None]) * scale[:, None]), axis=0)
            acc2_d += tl.sum(w[:, None] * ((v_q3 - zp[:, None]) * scale[:, None]), axis=0)

        acc4_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc4_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc4_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc4_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        for t_start in range(0, N_4, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_4
            w = tl.load(W_ptr + pid_hq * stride_w_h + (t_offs + N_2) * stride_w_t,
                        mask=t_mask, other=0.0)
            vp_ptrs_lo = (V_4bit_ptr + h_kv * stride_v4_h
                          + t_offs[:, None] * stride_v4_t
                          + d_q[None, :] * stride_v4_d)
            vp_ptrs_hi = (V_4bit_ptr + h_kv * stride_v4_h
                          + t_offs[:, None] * stride_v4_t
                          + (d_q + D_QUARTER_C)[None, :] * stride_v4_d)
            packed_lo = tl.load(vp_ptrs_lo, mask=t_mask[:, None], other=0).to(tl.int32)
            packed_hi = tl.load(vp_ptrs_hi, mask=t_mask[:, None], other=0).to(tl.int32)
            v_a = (packed_lo & 0xF).to(tl.float32)
            v_b = (packed_hi & 0xF).to(tl.float32)
            v_c = ((packed_lo >> 4) & 0xF).to(tl.float32)
            v_d = ((packed_hi >> 4) & 0xF).to(tl.float32)
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + (t_offs + N_2) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + (t_offs + N_2) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            acc4_a += tl.sum(w[:, None] * ((v_a - zp[:, None]) * scale[:, None]), axis=0)
            acc4_b += tl.sum(w[:, None] * ((v_b - zp[:, None]) * scale[:, None]), axis=0)
            acc4_c += tl.sum(w[:, None] * ((v_c - zp[:, None]) * scale[:, None]), axis=0)
            acc4_d += tl.sum(w[:, None] * ((v_d - zp[:, None]) * scale[:, None]), axis=0)

        acc8_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc8_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc8_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc8_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        for t_start in range(0, N_8, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_8
            w = tl.load(W_ptr + pid_hq * stride_w_h + (t_offs + N_2 + N_4) * stride_w_t,
                        mask=t_mask, other=0.0)
            v_ptrs_a = (V_8bit_ptr + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + d_q[None, :] * stride_v8_d)
            v_ptrs_b = (V_8bit_ptr + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + D_QUARTER_C)[None, :] * stride_v8_d)
            v_ptrs_c = (V_8bit_ptr + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + 2 * D_QUARTER_C)[None, :] * stride_v8_d)
            v_ptrs_d = (V_8bit_ptr + h_kv * stride_v8_h
                        + t_offs[:, None] * stride_v8_t
                        + (d_q + 3 * D_QUARTER_C)[None, :] * stride_v8_d)
            v_a = tl.load(v_ptrs_a, mask=t_mask[:, None], other=0).to(tl.float32)
            v_b = tl.load(v_ptrs_b, mask=t_mask[:, None], other=0).to(tl.float32)
            v_c = tl.load(v_ptrs_c, mask=t_mask[:, None], other=0).to(tl.float32)
            v_d = tl.load(v_ptrs_d, mask=t_mask[:, None], other=0).to(tl.float32)
            scale = tl.load(V_scale_ptr + h_kv * stride_vs_h + (t_offs + N_2 + N_4) * stride_vs_t,
                            mask=t_mask, other=1.0).to(tl.float32)
            zp = tl.load(V_zp_ptr + h_kv * stride_vs_h + (t_offs + N_2 + N_4) * stride_vs_t,
                         mask=t_mask, other=0.0).to(tl.float32)
            acc8_a += tl.sum(w[:, None] * ((v_a - zp[:, None]) * scale[:, None]), axis=0)
            acc8_b += tl.sum(w[:, None] * ((v_b - zp[:, None]) * scale[:, None]), axis=0)
            acc8_c += tl.sum(w[:, None] * ((v_c - zp[:, None]) * scale[:, None]), axis=0)
            acc8_d += tl.sum(w[:, None] * ((v_d - zp[:, None]) * scale[:, None]), axis=0)

        # ----- Zone B FP16 (v=16) segment -----
        acc16_a = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc16_b = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc16_c = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        acc16_d = tl.zeros((D_QUARTER_C,), dtype=tl.float32)
        if HAS_V16:
            N_V16 = tl.load(n_v16_ph_ptr + h_kv)
            for t_start in range(0, N_V16, BLOCK_T):
                t_offs = t_start + tl.arange(0, BLOCK_T)
                t_mask = t_offs < N_V16
                w = tl.load(W_ptr + pid_hq * stride_w_h + (t_offs + MAX_T_V) * stride_w_t,
                            mask=t_mask, other=0.0)
                v_ptrs_a = (V16_ptr + h_kv * stride_v16_h
                            + t_offs[:, None] * stride_v16_t
                            + d_q[None, :] * stride_v16_d)
                v_ptrs_b = (V16_ptr + h_kv * stride_v16_h
                            + t_offs[:, None] * stride_v16_t
                            + (d_q + D_QUARTER_C)[None, :] * stride_v16_d)
                v_ptrs_c = (V16_ptr + h_kv * stride_v16_h
                            + t_offs[:, None] * stride_v16_t
                            + (d_q + 2 * D_QUARTER_C)[None, :] * stride_v16_d)
                v_ptrs_d = (V16_ptr + h_kv * stride_v16_h
                            + t_offs[:, None] * stride_v16_t
                            + (d_q + 3 * D_QUARTER_C)[None, :] * stride_v16_d)
                v_a = tl.load(v_ptrs_a, mask=t_mask[:, None], other=0.0).to(tl.float32)
                v_b = tl.load(v_ptrs_b, mask=t_mask[:, None], other=0.0).to(tl.float32)
                v_c = tl.load(v_ptrs_c, mask=t_mask[:, None], other=0.0).to(tl.float32)
                v_d = tl.load(v_ptrs_d, mask=t_mask[:, None], other=0.0).to(tl.float32)
                acc16_a += tl.sum(w[:, None] * v_a, axis=0)
                acc16_b += tl.sum(w[:, None] * v_b, axis=0)
                acc16_c += tl.sum(w[:, None] * v_c, axis=0)
                acc16_d += tl.sum(w[:, None] * v_d, axis=0)

        tl.store(base_out + d_q * stride_o_d, acc2_a + acc4_a + acc8_a + acc16_a)
        tl.store(base_out + (d_q + D_QUARTER_C) * stride_o_d, acc2_b + acc4_b + acc8_b + acc16_b)
        tl.store(base_out + (d_q + 2 * D_QUARTER_C) * stride_o_d, acc2_c + acc4_c + acc8_c + acc16_c)
        tl.store(base_out + (d_q + 3 * D_QUARTER_C) * stride_o_d, acc2_d + acc4_d + acc8_d + acc16_d)

    # ===================================================================
    # E2: FP16 tensor-core V kernel (A100 SM 80 compatible)
    #
    # Strategy
    # --------
    # * M-dim carries 4 GQA H_q heads per h_kv + 12 padding -> M_TC=16,
    #   which is the smallest M allowed by `mma.m16n8k16`.
    # * N-dim carries BLOCK_D=32 = D_QUARTER output channels (matches the
    #   scalar kernel's per-program D-slice).
    # * K-dim carries BLOCK_T=64 tokens per tl.dot.
    # * Grid: (H_kv, D // BLOCK_D, T_CHUNK_COUNT) = (8, 4, 16) = 512
    #   programs on default Llama-3.1-8B config. T is split across
    #   T_CHUNK_COUNT programs via a BLOCK_T-strided loop so each chunk
    #   writes into shared output columns — the partial results are
    #   combined with `tl.atomic_add`. The wrapper zeros `Out` before
    #   launch so atomic sums land on a clean slate (required!).
    # * Zone B (HAS_V16) reuses the same acc_partial with V16 FP16 tile.
    # ===================================================================
    from obkv_accel.triton_dequant_utils import (  # noqa: E402
        dequant_2bit_v_tile as _dq_2bit_tile,
        dequant_4bit_v_tile as _dq_4bit_tile,
        dequant_8bit_v_tile as _dq_8bit_tile,
    )

    @triton.jit
    def _v_weighted_sum_unified_dquarter_perhead_tc_kernel(
        W_ptr, V_2bit_ptr, V_4bit_ptr, V_8bit_ptr,
        V_scale_ptr, V_zp_ptr, Out_ptr,
        seg_bounds_ptr,
        V16_ptr,
        n_v16_ph_ptr,
        MAX_T_V,
        stride_w_h, stride_w_t: tl.constexpr,
        stride_v2_h, stride_v2_t, stride_v2_d: tl.constexpr,
        stride_v4_h, stride_v4_t, stride_v4_d: tl.constexpr,
        stride_v8_h, stride_v8_t, stride_v8_d: tl.constexpr,
        stride_v16_h, stride_v16_t, stride_v16_d: tl.constexpr,
        stride_vs_h, stride_vs_t,
        stride_o_h, stride_o_d: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_QUARTER_C: tl.constexpr,
        M_TC_C: tl.constexpr,            # = 16
        GQA_FACTOR_C: tl.constexpr,      # = 4
        T_CHUNK_COUNT_C: tl.constexpr,
        HAS_V16: tl.constexpr,
    ):
        pid_hkv = tl.program_id(0)
        pid_dq = tl.program_id(1)
        pid_tc = tl.program_id(2)

        # Per-head segment counts
        N_2 = tl.load(seg_bounds_ptr + pid_hkv * 3 + 0)
        N_4 = tl.load(seg_bounds_ptr + pid_hkv * 3 + 1)
        N_8 = tl.load(seg_bounds_ptr + pid_hkv * 3 + 2)

        # M-dim: 4 valid rows (h_kv*GQA + {0..3}) padded to M_TC=16.
        m = tl.arange(0, M_TC_C)
        m_valid = m < GQA_FACTOR_C
        # For invalid rows, bind hq to a safe valid H_q (h_kv*GQA) so pointer
        # arithmetic stays in-range even though the mask zero-fills.
        hq_safe = tl.where(m_valid, pid_hkv * GQA_FACTOR_C + m,
                           pid_hkv * GQA_FACTOR_C)

        # Per-program partial accumulator: [M=16, N=BLOCK_D=32] FP32.
        acc_partial = tl.zeros((M_TC_C, D_QUARTER_C), dtype=tl.float32)

        base_vs = V_scale_ptr + pid_hkv * stride_vs_h
        base_vz = V_zp_ptr + pid_hkv * stride_vs_h

        # ----- 2-bit segment -----
        # Strided loop: program pid_tc owns tiles starting at pid_tc*BLOCK_T
        # with stride T_CHUNK_COUNT * BLOCK_T. This guarantees tiles never
        # cross segment boundaries.
        shift_2bit = pid_dq * 2
        for t_start in range(pid_tc * BLOCK_T, N_2, T_CHUNK_COUNT_C * BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_2
            # W tile: [M, K] = [16, BLOCK_T] FP32 -> FP16
            w_fp32 = tl.load(
                W_ptr + hq_safe[:, None] * stride_w_h
                + t_offs[None, :] * stride_w_t,
                mask=m_valid[:, None] & t_mask[None, :], other=0.0,
            )
            w_tile = w_fp32.to(tl.float16)
            # V tile: [K=BLOCK_T, N=BLOCK_D=32] FP16 via helper.
            v_tile = _dq_2bit_tile(
                V_2bit_ptr, base_vs, base_vz, pid_hkv,
                t_offs, t_mask, 0,
                stride_v2_h, stride_v2_t, stride_v2_d,
                stride_vs_t,
                SHIFT_2BIT_C=shift_2bit,
                SUB_D=D_QUARTER_C, BLOCK_T=BLOCK_T,
            )
            acc_partial = tl.dot(w_tile, v_tile, acc=acc_partial,
                                 out_dtype=tl.float32)

        # ----- 4-bit segment -----
        byte_offset_4bit = (pid_dq & 1) * D_QUARTER_C
        nibble_shift = (pid_dq & 2) * 2
        for t_start in range(pid_tc * BLOCK_T, N_4, T_CHUNK_COUNT_C * BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_4
            # W column offset by N_2 (segment origin in W shared layout).
            w_fp32 = tl.load(
                W_ptr + hq_safe[:, None] * stride_w_h
                + (t_offs + N_2)[None, :] * stride_w_t,
                mask=m_valid[:, None] & t_mask[None, :], other=0.0,
            )
            w_tile = w_fp32.to(tl.float16)
            v_tile = _dq_4bit_tile(
                V_4bit_ptr, base_vs, base_vz, pid_hkv,
                t_offs, t_mask, N_2,
                stride_v4_h, stride_v4_t, stride_v4_d,
                stride_vs_t,
                NIBBLE_SHIFT_C=nibble_shift,
                BYTE_OFFSET_4BIT_C=byte_offset_4bit,
                SUB_D=D_QUARTER_C, BLOCK_T=BLOCK_T,
            )
            acc_partial = tl.dot(w_tile, v_tile, acc=acc_partial,
                                 out_dtype=tl.float32)

        # ----- 8-bit segment -----
        d_q_out_offset = pid_dq * D_QUARTER_C
        for t_start in range(pid_tc * BLOCK_T, N_8, T_CHUNK_COUNT_C * BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < N_8
            w_fp32 = tl.load(
                W_ptr + hq_safe[:, None] * stride_w_h
                + (t_offs + N_2 + N_4)[None, :] * stride_w_t,
                mask=m_valid[:, None] & t_mask[None, :], other=0.0,
            )
            w_tile = w_fp32.to(tl.float16)
            v_tile = _dq_8bit_tile(
                V_8bit_ptr, base_vs, base_vz, pid_hkv,
                t_offs, t_mask, N_2 + N_4,
                stride_v8_h, stride_v8_t, stride_v8_d,
                stride_vs_t,
                D_OFFSET_C=d_q_out_offset,
                SUB_D=D_QUARTER_C, BLOCK_T=BLOCK_T,
            )
            acc_partial = tl.dot(w_tile, v_tile, acc=acc_partial,
                                 out_dtype=tl.float32)

        # ----- Zone B FP16 (v=16) segment -----
        if HAS_V16:
            N_V16 = tl.load(n_v16_ph_ptr + pid_hkv)
            for t_start in range(pid_tc * BLOCK_T, N_V16,
                                 T_CHUNK_COUNT_C * BLOCK_T):
                t_offs = t_start + tl.arange(0, BLOCK_T)
                t_mask = t_offs < N_V16
                w_fp32 = tl.load(
                    W_ptr + hq_safe[:, None] * stride_w_h
                    + (t_offs + MAX_T_V)[None, :] * stride_w_t,
                    mask=m_valid[:, None] & t_mask[None, :], other=0.0,
                )
                w_tile = w_fp32.to(tl.float16)
                # V16 tile: raw FP16 values, no dequant.
                d_sub = tl.arange(0, D_QUARTER_C)
                v_ptrs = (V16_ptr + pid_hkv * stride_v16_h
                          + t_offs[:, None] * stride_v16_t
                          + (d_sub + d_q_out_offset)[None, :] * stride_v16_d)
                v_tile = tl.load(v_ptrs, mask=t_mask[:, None],
                                 other=0.0).to(tl.float16)
                acc_partial = tl.dot(w_tile, v_tile, acc=acc_partial,
                                     out_dtype=tl.float32)

        # ----- writeback (atomic) -----
        # Only the 4 valid rows (m=0..3) carry real data; each row maps to
        # h_q = h_kv*GQA + m. For-range over static GQA factor unrolls.
        d_sub = tl.arange(0, D_QUARTER_C)
        out_d_offs = (d_q_out_offset + d_sub) * stride_o_d
        for i in tl.static_range(GQA_FACTOR_C):
            h_q_i = pid_hkv * GQA_FACTOR_C + i
            # Extract row i of acc_partial.
            row_mask = tl.arange(0, M_TC_C) == i  # [M_TC]
            # Broadcast reduction: sum_m (row_mask[m] * acc[m, :]) picks row i.
            row_vals = tl.sum(tl.where(row_mask[:, None], acc_partial,
                                       0.0), axis=0)
            tl.atomic_add(Out_ptr + h_q_i * stride_o_h + out_d_offs,
                          row_vals)


# ---------------------------------------------------------------------------
# Python wrappers for individual bit-width kernels
# ---------------------------------------------------------------------------

BLOCK_T_DEFAULT = 64


def v_weighted_sum_4bit(attn_weights, V_packed, V_scale, V_zp, T_seg):
    """Weighted sum with 4-bit packed V.

    Args:
        attn_weights: [H_q, T_seg] FP32
        V_packed:     [H_kv, T_seg, D//2=64] uint8
        V_scale:      [H_kv, T_seg, 1] FP16
        V_zp:         [H_kv, T_seg, 1] FP16
        T_seg:        int

    Returns:
        out: [H_q, D=128] FP32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    H_q = attn_weights.shape[0]
    out = torch.zeros((H_q, D), dtype=torch.float32, device=attn_weights.device)

    if T_seg == 0:
        return out

    # Squeeze trailing dim from scale/zp for contiguous stride access
    # scale: [H_kv, T_seg, 1] -> stride_vs_t = scale.stride(1)
    grid = (H_q,)
    _v_weighted_sum_4bit_kernel[grid](
        attn_weights, V_packed, V_scale, V_zp, out,
        T_seg,
        attn_weights.stride(0), attn_weights.stride(1),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2),
        V_scale.stride(0), V_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T_DEFAULT,
        D_PACKED_C=D // 2,
        D_C=D,
        GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
    )
    return out


def v_weighted_sum_2bit(attn_weights, V_packed, V_scale, V_zp, T_seg):
    """Weighted sum with 2-bit packed V.

    Args:
        attn_weights: [H_q, T_seg] FP32
        V_packed:     [H_kv, T_seg, D//4=32] uint8
        V_scale:      [H_kv, T_seg, 1] FP16
        V_zp:         [H_kv, T_seg, 1] FP16
        T_seg:        int

    Returns:
        out: [H_q, D=128] FP32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    H_q = attn_weights.shape[0]
    out = torch.zeros((H_q, D), dtype=torch.float32, device=attn_weights.device)

    if T_seg == 0:
        return out

    grid = (H_q,)
    _v_weighted_sum_2bit_kernel[grid](
        attn_weights, V_packed, V_scale, V_zp, out,
        T_seg,
        attn_weights.stride(0), attn_weights.stride(1),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2),
        V_scale.stride(0), V_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T_DEFAULT,
        D_QUARTER_C=D // 4,
        D_C=D,
        GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
    )
    return out


def v_weighted_sum_8bit(attn_weights, V_packed, V_scale, V_zp, T_seg):
    """Weighted sum with 8-bit V (no packing).

    Args:
        attn_weights: [H_q, T_seg] FP32
        V_packed:     [H_kv, T_seg, D=128] uint8
        V_scale:      [H_kv, T_seg, 1] FP16
        V_zp:         [H_kv, T_seg, 1] FP16
        T_seg:        int

    Returns:
        out: [H_q, D=128] FP32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    H_q = attn_weights.shape[0]
    out = torch.zeros((H_q, D), dtype=torch.float32, device=attn_weights.device)

    if T_seg == 0:
        return out

    grid = (H_q,)
    _v_weighted_sum_8bit_kernel[grid](
        attn_weights, V_packed, V_scale, V_zp, out,
        T_seg,
        attn_weights.stride(0), attn_weights.stride(1),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2),
        V_scale.stride(0), V_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T_DEFAULT,
        D_C=D,
        GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
    )
    return out


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------
def v_dequant_weighted_sum(attn_weights, V_2bit, V_4bit, V_8bit,
                           V_scale, V_zp, seg_bounds):
    """Compute weighted sum over all V segments (2-bit, 4-bit, 8-bit).

    The V cache is sorted by bit-width: first N_2 tokens are 2-bit,
    next N_4 tokens are 4-bit, last N_8 tokens are 8-bit. Scale and zp
    are in this sorted order.

    Args:
        attn_weights: [H_q, T_eff] FP32  (sorted token order, matching V)
        V_2bit:       [H_kv, N_2, D//4=32] uint8  (may have N_2=0)
        V_4bit:       [H_kv, N_4, D//2=64] uint8  (may have N_4=0)
        V_8bit:       [H_kv, N_8, D=128]   uint8  (may have N_8=0)
        V_scale:      [H_kv, T_eff, 1] FP16
        V_zp:         [H_kv, T_eff, 1] FP16
        seg_bounds:   tuple (N_2, N_4, N_8)

    Returns:
        out: [H_q, D=128] FP32
    """
    N_2, N_4, N_8 = seg_bounds
    T_eff = N_2 + N_4 + N_8

    H_q = attn_weights.shape[0]
    device = attn_weights.device

    # Split attention weights by segment
    w_2 = attn_weights[:, :N_2]                      # [H_q, N_2]
    w_4 = attn_weights[:, N_2:N_2 + N_4]             # [H_q, N_4]
    w_8 = attn_weights[:, N_2 + N_4:N_2 + N_4 + N_8] # [H_q, N_8]

    # Split scale/zp by segment
    scale_2 = V_scale[:, :N_2, :]
    scale_4 = V_scale[:, N_2:N_2 + N_4, :]
    scale_8 = V_scale[:, N_2 + N_4:, :]

    zp_2 = V_zp[:, :N_2, :]
    zp_4 = V_zp[:, N_2:N_2 + N_4, :]
    zp_8 = V_zp[:, N_2 + N_4:, :]

    # Call each kernel
    out_2 = v_weighted_sum_2bit(w_2, V_2bit, scale_2, zp_2, N_2)
    out_4 = v_weighted_sum_4bit(w_4, V_4bit, scale_4, zp_4, N_4)
    out_8 = v_weighted_sum_8bit(w_8, V_8bit, scale_8, zp_8, N_8)

    return out_2 + out_4 + out_8


def v_dequant_weighted_sum_dispatch(attn_weights_old, packed_layer,
                                     V16=None, n_v16_per_head=None, max_T_v=0):
    """Compute weighted sum over all V segments using PackedKVLayer.

    Convenience wrapper that extracts fields from the packed_layer dataclass
    and calls v_dequant_weighted_sum_unified. If the layer carries per-head
    metadata (seg_bounds_per_head), dispatches to the per-head variant.

    Args:
        attn_weights_old: [H_q, T_eff] (or [H_q, T_eff_k] when V16 is supplied)
                          FP32 (sorted token order).
        packed_layer:     PackedKVLayer dataclass instance
        V16, n_v16_per_head, max_T_v: Zone B FP16 stripe (per-head variant only).

    Returns:
        out: [H_q, D=128] FP32
    """
    if getattr(packed_layer, "seg_bounds_per_head", None) is not None:
        return v_dequant_weighted_sum_unified_perhead(
            attn_weights_old,
            packed_layer.V_2bit,
            packed_layer.V_4bit,
            packed_layer.V_8bit,
            packed_layer.V_scale,
            packed_layer.V_zp,
            packed_layer.seg_bounds_per_head,
            packed_layer.seg_bounds,
            V16=V16,
            n_v16_per_head=n_v16_per_head,
            max_T_v=max_T_v,
        )
    return v_dequant_weighted_sum_unified(
        attn_weights_old,
        packed_layer.V_2bit,
        packed_layer.V_4bit,
        packed_layer.V_8bit,
        packed_layer.V_scale,
        packed_layer.V_zp,
        packed_layer.seg_bounds,
    )


def v_dequant_weighted_sum_unified(attn_weights, V_2bit, V_4bit, V_8bit,
                                    V_scale, V_zp, seg_bounds):
    """Compute weighted sum over all V segments in a single kernel launch.

    Same interface as v_dequant_weighted_sum, but fuses 2/4/8-bit processing
    into one Triton kernel to eliminate launch + alloc overhead.

    Args:
        attn_weights: [H_q, T_eff] FP32
        V_2bit:       [H_kv, N_2, D//4=32] uint8  (or None if N_2=0)
        V_4bit:       [H_kv, N_4, D//2=64] uint8  (or None if N_4=0)
        V_8bit:       [H_kv, N_8, D=128]   uint8  (or None if N_8=0)
        V_scale:      [H_kv, T_eff, 1] FP16
        V_zp:         [H_kv, T_eff, 1] FP16
        seg_bounds:   tuple (N_2, N_4, N_8)

    Returns:
        out: [H_q, D=128] FP32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    N_2, N_4, N_8 = seg_bounds
    T_eff = N_2 + N_4 + N_8

    H_q = attn_weights.shape[0]
    H_kv = V_scale.shape[1] if V_scale.dim() == 4 else V_scale.shape[0]
    device = attn_weights.device

    if T_eff == 0:
        return torch.zeros((H_q, D), dtype=torch.float32, device=device)

    # Output: torch.empty is safe because 2-bit section always direct-stores
    # the full output (zeros if N_2=0, since accumulators init to zero).
    out = torch.empty((H_q, D), dtype=torch.float32, device=device)

    # For None segments, use cached zero-sized tensors (avoid repeated alloc)
    d2, d4, d8 = _get_dummy_tensors(device, H_kv)
    if V_2bit is None: V_2bit = d2
    if V_4bit is None: V_4bit = d4
    if V_8bit is None: V_8bit = d8

    if _V_DQUARTER:
        grid = (H_q, 4)
        _v_weighted_sum_unified_dquarter_kernel[grid](
            attn_weights,
            V_2bit, V_4bit, V_8bit,
            V_scale, V_zp,
            out,
            N_2, N_4, N_8,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit.stride(0), V_2bit.stride(1), V_2bit.stride(2),
            V_4bit.stride(0), V_4bit.stride(1), V_4bit.stride(2),
            V_8bit.stride(0), V_8bit.stride(1), V_8bit.stride(2),
            V_scale.stride(0), V_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    else:
        grid = (H_q,)
        _v_weighted_sum_unified_kernel[grid](
            attn_weights,
            V_2bit, V_4bit, V_8bit,
            V_scale, V_zp,
            out,
            N_2, N_4, N_8,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit.stride(0), V_2bit.stride(1), V_2bit.stride(2),
            V_4bit.stride(0), V_4bit.stride(1), V_4bit.stride(2),
            V_8bit.stride(0), V_8bit.stride(1), V_8bit.stride(2),
            V_scale.stride(0), V_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            D_PACKED_C=D // 2,
            D_C=D,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    return out


def v_dequant_weighted_sum_unified_into(attn_weights, V_2bit, V_4bit, V_8bit,
                                         V_scale, V_zp, seg_bounds, out):
    """Same as v_dequant_weighted_sum_unified but writes into pre-allocated ``out``.

    Args:
        attn_weights, V_2bit, V_4bit, V_8bit, V_scale, V_zp, seg_bounds:
            same as v_dequant_weighted_sum_unified.
        out: [H_q, D] FP32 — pre-allocated output buffer.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    N_2, N_4, N_8 = seg_bounds
    T_eff = N_2 + N_4 + N_8

    H_q = attn_weights.shape[0]
    H_kv = V_scale.shape[1] if V_scale.dim() == 4 else V_scale.shape[0]

    if T_eff == 0:
        out.zero_()
        return

    d2, d4, d8 = _get_dummy_tensors(attn_weights.device, H_kv)
    if V_2bit is None: V_2bit = d2
    if V_4bit is None: V_4bit = d4
    if V_8bit is None: V_8bit = d8

    if _V_DQUARTER:
        grid = (H_q, 4)
        _v_weighted_sum_unified_dquarter_kernel[grid](
            attn_weights,
            V_2bit, V_4bit, V_8bit,
            V_scale, V_zp,
            out,
            N_2, N_4, N_8,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit.stride(0), V_2bit.stride(1), V_2bit.stride(2),
            V_4bit.stride(0), V_4bit.stride(1), V_4bit.stride(2),
            V_8bit.stride(0), V_8bit.stride(1), V_8bit.stride(2),
            V_scale.stride(0), V_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    else:
        grid = (H_q,)
        _v_weighted_sum_unified_kernel[grid](
            attn_weights,
            V_2bit, V_4bit, V_8bit,
            V_scale, V_zp,
            out,
            N_2, N_4, N_8,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit.stride(0), V_2bit.stride(1), V_2bit.stride(2),
            V_4bit.stride(0), V_4bit.stride(1), V_4bit.stride(2),
            V_8bit.stride(0), V_8bit.stride(1), V_8bit.stride(2),
            V_scale.stride(0), V_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            D_PACKED_C=D // 2,
            D_C=D,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )


def v_dequant_weighted_sum_unified_perhead(
    attn_weights, V_2bit, V_4bit, V_8bit,
    V_scale, V_zp, seg_bounds_per_head, seg_bounds_max,
    V16=None, n_v16_per_head=None, max_T_v=0,
):
    """Per-head variant: N_2/N_4/N_8 read from seg_bounds_per_head[h_kv] at runtime.

    Args:
        attn_weights:        [H_q, max_T_eff] or [H_q, max_T_eff_k] FP32 (if V16 is not None).
                             Head h valid at [0, T_eff_h) for Zone A and
                             [max_T_v, max_T_v + n_v16_h) for Zone B.
        V_2bit/4/8:          [1, H_kv, max_N_i, sub_D] uint8 or None
        V_scale/V_zp:        [1, H_kv, max_T_eff, 1] FP16
        seg_bounds_per_head: [H_kv, 3] int32 — N_2, N_4, N_8 per head
        seg_bounds_max:      (max_N_2, max_N_4, max_N_8) — for dummy-tensor sizing
        V16:                 [H_kv, max_n_v16, D] FP16 or None (no Zone B)
        n_v16_per_head:      [H_kv] int32 or None
        max_T_v:             int, col offset in attn_weights where Zone B starts

    Returns:
        out: [H_q, D] FP32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    max_N_2, max_N_4, max_N_8 = seg_bounds_max
    T_eff_max = max_N_2 + max_N_4 + max_N_8
    H_q = attn_weights.shape[0]
    H_kv = V_scale.shape[1] if V_scale.dim() == 4 else V_scale.shape[0]
    device = attn_weights.device

    has_v16 = V16 is not None

    if T_eff_max == 0 and not has_v16:
        return torch.zeros((H_q, D), dtype=torch.float32, device=device)

    # TC path uses atomic_add writeback -> allocate zeros, not empty.
    if _V_TC and _V_DQUARTER:
        out = torch.zeros((H_q, D), dtype=torch.float32, device=device)
    else:
        out = torch.empty((H_q, D), dtype=torch.float32, device=device)

    # Squeeze leading batch dim from stacked [1, H_kv, ...] → [H_kv, ...]
    V_2bit_k = V_2bit.squeeze(0) if V_2bit is not None else None
    V_4bit_k = V_4bit.squeeze(0) if V_4bit is not None else None
    V_8bit_k = V_8bit.squeeze(0) if V_8bit is not None else None
    V_scale_k = V_scale.squeeze(0) if V_scale.dim() == 4 else V_scale
    V_zp_k = V_zp.squeeze(0) if V_zp.dim() == 4 else V_zp

    d2, d4, d8 = _get_dummy_tensors(device, H_kv)
    if V_2bit_k is None: V_2bit_k = d2
    if V_4bit_k is None: V_4bit_k = d4
    if V_8bit_k is None: V_8bit_k = d8

    if has_v16:
        V16_k = V16
        n_v16_ph = n_v16_per_head
    else:
        V16_k, n_v16_ph = _get_v16_dummy(device, H_kv)

    if _V_TC and _V_DQUARTER:
        grid = (H_kv, D // (D // 4), _V_TC_T_CHUNKS)
        _v_weighted_sum_unified_dquarter_perhead_tc_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            M_TC_C=16,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            T_CHUNK_COUNT_C=_V_TC_T_CHUNKS,
            HAS_V16=(1 if has_v16 else 0),
            num_warps=4,
            num_stages=3,
        )
    elif _V_DQUARTER:
        grid = (H_q, 4)
        _v_weighted_sum_unified_dquarter_perhead_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            HAS_V16=(1 if has_v16 else 0),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    else:
        grid = (H_q,)
        _v_weighted_sum_unified_perhead_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            D_PACKED_C=D // 2,
            D_C=D,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            HAS_V16=(1 if has_v16 else 0),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    return out


def v_dequant_weighted_sum_unified_perhead_into(
    attn_weights, V_2bit, V_4bit, V_8bit,
    V_scale, V_zp, seg_bounds_per_head, seg_bounds_max, out,
    V16=None, n_v16_per_head=None, max_T_v=0,
):
    """Same as v_dequant_weighted_sum_unified_perhead but writes into ``out``.

    When ``V16`` / ``n_v16_per_head`` / ``max_T_v`` are provided the fused Zone B
    FP16 stripe is accumulated into ``out`` as part of the same kernel launch,
    eliminating the separate per-layer ``torch.bmm`` for Zone B.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    max_N_2, max_N_4, max_N_8 = seg_bounds_max
    T_eff_max = max_N_2 + max_N_4 + max_N_8
    H_q = attn_weights.shape[0]
    H_kv = V_scale.shape[1] if V_scale.dim() == 4 else V_scale.shape[0]

    has_v16 = V16 is not None

    if T_eff_max == 0 and not has_v16:
        out.zero_()
        return

    V_2bit_k = V_2bit.squeeze(0) if V_2bit is not None else None
    V_4bit_k = V_4bit.squeeze(0) if V_4bit is not None else None
    V_8bit_k = V_8bit.squeeze(0) if V_8bit is not None else None
    V_scale_k = V_scale.squeeze(0) if V_scale.dim() == 4 else V_scale
    V_zp_k = V_zp.squeeze(0) if V_zp.dim() == 4 else V_zp

    d2, d4, d8 = _get_dummy_tensors(attn_weights.device, H_kv)
    if V_2bit_k is None: V_2bit_k = d2
    if V_4bit_k is None: V_4bit_k = d4
    if V_8bit_k is None: V_8bit_k = d8

    if has_v16:
        V16_k = V16
        n_v16_ph = n_v16_per_head
    else:
        V16_k, n_v16_ph = _get_v16_dummy(attn_weights.device, H_kv)

    if _V_TC and _V_DQUARTER:
        # atomic_add writeback requires a clean output buffer.
        out.zero_()
        grid = (H_kv, D // (D // 4), _V_TC_T_CHUNKS)  # (H_kv, 4, T_chunks)
        _v_weighted_sum_unified_dquarter_perhead_tc_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            M_TC_C=16,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            T_CHUNK_COUNT_C=_V_TC_T_CHUNKS,
            HAS_V16=(1 if has_v16 else 0),
            num_warps=4,
            num_stages=3,
        )
    elif _V_DQUARTER:
        grid = (H_q, 4)
        _v_weighted_sum_unified_dquarter_perhead_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            HAS_V16=(1 if has_v16 else 0),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )
    else:
        grid = (H_q,)
        _v_weighted_sum_unified_perhead_kernel[grid](
            attn_weights,
            V_2bit_k, V_4bit_k, V_8bit_k,
            V_scale_k, V_zp_k,
            out,
            seg_bounds_per_head,
            V16_k, n_v16_ph, max_T_v,
            attn_weights.stride(0), attn_weights.stride(1),
            V_2bit_k.stride(0), V_2bit_k.stride(1), V_2bit_k.stride(2),
            V_4bit_k.stride(0), V_4bit_k.stride(1), V_4bit_k.stride(2),
            V_8bit_k.stride(0), V_8bit_k.stride(1), V_8bit_k.stride(2),
            V16_k.stride(0), V16_k.stride(1), V16_k.stride(2),
            V_scale_k.stride(0), V_scale_k.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_T=BLOCK_T_DEFAULT,
            D_QUARTER_C=D // 4,
            D_PACKED_C=D // 2,
            D_C=D,
            GQA_FACTOR_C=_runtime_gqa(attn_weights, V_scale),
            HAS_V16=(1 if has_v16 else 0),
            num_warps=_V_NUM_WARPS,
            num_stages=_V_NUM_STAGES,
        )


def v_dequant_weighted_sum_dispatch_into(attn_weights_old, packed_layer, out,
                                          V16=None, n_v16_per_head=None, max_T_v=0):
    """Convenience wrapper: same as v_dequant_weighted_sum_dispatch but into ``out``.

    When ``V16`` is supplied, Zone B FP16 stripe is fused into the same kernel.
    """
    if getattr(packed_layer, "seg_bounds_per_head", None) is not None:
        v_dequant_weighted_sum_unified_perhead_into(
            attn_weights_old,
            packed_layer.V_2bit,
            packed_layer.V_4bit,
            packed_layer.V_8bit,
            packed_layer.V_scale,
            packed_layer.V_zp,
            packed_layer.seg_bounds_per_head,
            packed_layer.seg_bounds,
            out,
            V16=V16,
            n_v16_per_head=n_v16_per_head,
            max_T_v=max_T_v,
        )
        return
    v_dequant_weighted_sum_unified_into(
        attn_weights_old,
        packed_layer.V_2bit,
        packed_layer.V_4bit,
        packed_layer.V_8bit,
        packed_layer.V_scale,
        packed_layer.V_zp,
        packed_layer.seg_bounds,
        out,
    )


# ---------------------------------------------------------------------------
# Pure PyTorch reference functions (for testing)
# ---------------------------------------------------------------------------

def _unpack_2bit_ref(packed):
    """Unpack 2-bit quarter-split layout.

    Args:
        packed: [..., D//4=32] uint8

    Returns:
        unpacked: [..., D=128] float32
    """
    p = packed.to(torch.int32)
    q0 = (p & 0x3).to(torch.float32)
    q1 = ((p >> 2) & 0x3).to(torch.float32)
    q2 = ((p >> 4) & 0x3).to(torch.float32)
    q3 = ((p >> 6) & 0x3).to(torch.float32)
    return torch.cat([q0, q1, q2, q3], dim=-1)


def _unpack_4bit_ref(packed):
    """Unpack 4-bit half-split layout.

    Args:
        packed: [..., D//2=64] uint8

    Returns:
        unpacked: [..., D=128] float32
    """
    p = packed.to(torch.int32)
    lo = (p & 0xF).to(torch.float32)
    hi = ((p >> 4) & 0xF).to(torch.float32)
    return torch.cat([lo, hi], dim=-1)


def _unpack_8bit_ref(packed):
    """Unpack 8-bit (identity).

    Args:
        packed: [..., D=128] uint8

    Returns:
        unpacked: [..., D=128] float32
    """
    return packed.to(torch.float32)


def v_weighted_sum_ref(attn_weights, V_packed, V_scale, V_zp, T_seg, n_bits):
    """Pure PyTorch reference for a single-bitwidth V weighted sum.

    Args:
        attn_weights: [H_q, T_seg] FP32
        V_packed:     [H_kv, T_seg, packed_D] uint8
        V_scale:      [H_kv, T_seg, 1] FP16
        V_zp:         [H_kv, T_seg, 1] FP16
        T_seg:        int
        n_bits:       int in {2, 4, 8}

    Returns:
        out: [H_q, D=128] FP32
    """
    H_q = attn_weights.shape[0]
    device = attn_weights.device

    if T_seg == 0:
        return torch.zeros((H_q, D), dtype=torch.float32, device=device)

    # Unpack
    if n_bits == 2:
        v_uint = _unpack_2bit_ref(V_packed)
    elif n_bits == 4:
        v_uint = _unpack_4bit_ref(V_packed)
    elif n_bits == 8:
        v_uint = _unpack_8bit_ref(V_packed)
    else:
        raise ValueError(f"Unsupported n_bits={n_bits}")

    # v_uint: [H_kv, T_seg, D]
    scale = V_scale.float()  # [H_kv, T_seg, 1]
    zp = V_zp.float()        # [H_kv, T_seg, 1]
    v_dequant = (v_uint - zp) * scale  # [H_kv, T_seg, D]

    # GQA expand (runtime factor; was hardcoded to GQA_FACTOR=4)
    v_expanded = v_dequant.repeat_interleave(
        _runtime_gqa(attn_weights, V_scale), dim=0,
    )  # [H_q, T_seg, D]

    # Weighted sum: out[h, d] = sum_t (w[h, t] * v[h, t, d])
    w = attn_weights.unsqueeze(-1)  # [H_q, T_seg, 1]
    out = (w * v_expanded).sum(dim=1)  # [H_q, D]

    return out


def v_dequant_weighted_sum_ref(attn_weights, V_2bit, V_4bit, V_8bit,
                                V_scale, V_zp, seg_bounds):
    """Pure PyTorch reference for v_dequant_weighted_sum.

    Same interface as the Triton dispatch version.
    """
    N_2, N_4, N_8 = seg_bounds

    w_2 = attn_weights[:, :N_2]
    w_4 = attn_weights[:, N_2:N_2 + N_4]
    w_8 = attn_weights[:, N_2 + N_4:N_2 + N_4 + N_8]

    scale_2 = V_scale[:, :N_2, :]
    scale_4 = V_scale[:, N_2:N_2 + N_4, :]
    scale_8 = V_scale[:, N_2 + N_4:, :]

    zp_2 = V_zp[:, :N_2, :]
    zp_4 = V_zp[:, N_2:N_2 + N_4, :]
    zp_8 = V_zp[:, N_2 + N_4:, :]

    out_2 = v_weighted_sum_ref(w_2, V_2bit, scale_2, zp_2, N_2, n_bits=2)
    out_4 = v_weighted_sum_ref(w_4, V_4bit, scale_4, zp_4, N_4, n_bits=4)
    out_8 = v_weighted_sum_ref(w_8, V_8bit, scale_8, zp_8, N_8, n_bits=8)

    return out_2 + out_4 + out_8
