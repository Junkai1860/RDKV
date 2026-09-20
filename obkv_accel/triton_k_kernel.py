"""
Triton kernel for K-side: per-channel mixed-precision (2/4/8-bit) QK dot product.

K channels are grouped by bit-width after sort (2-bit → 4-bit → 8-bit) and padded
so each group satisfies its bit-packing alignment constraint:
  - 2-bit: quarter-split, N_ch_2_padded must be a multiple of 4
  - 4-bit: half-split,    N_ch_4_padded must be a multiple of 2
  - 8-bit: no pack, no alignment requirement
Pad channels have scale=zp=0 and Q_perm=0 so their dequant contribution is 0.

Uniform K (e.g. all 4-bit) is the natural degenerate case where two of the
three segment sizes are 0. The unified kernel handles it without branching.

Dequant formula:  k_fp = (k_uint - zp) * scale
Bias trick: precompute qs=q*scale, bias=sum(zp*qs); then
    acc[t] = sum_d codes[t,d] * qs[d] - bias
GQA mapping: h_kv = h_q // GQA_FACTOR (Llama-3.1-8B: H_q=32, H_kv=8, factor=4).
"""

from __future__ import annotations

import math
import os
import time
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Env-gated fine-grained timing for the K decode kernel. When enabled,
# torch.cuda.synchronize() is inserted around Q.index_select() and the
# Triton launch so we can separate wrapper vs kernel cost.
_DIAG_ENABLED = os.environ.get("OBKV_K_DIAG", "0") == "1"
_DIAG = {"index_select_ms": 0.0, "kernel_ms": 0.0, "n_calls": 0}

# One-shot per-layer HAS_2 / HAS_4 / HAS_8 print, gated by OBKV_K_HAS_X_DIAG=1.
_HAS_X_DIAG_ENABLED = os.environ.get("OBKV_K_HAS_X_DIAG", "0") == "1"
_HAS_X_PRINTED: dict[int, int] = {}

# E2: FP16 tensor-core K kernels. When 1, wrapper dispatches to the `*_tc*`
# variants that batch 4 GQA H_q rows + 12 padding rows into M_TC=16 and use
# `tl.dot(mma.m16n8k16)` (A100 SM 80 compatible). Default ON as of 2026-04-24
# after GH200 gates green; set OBKV_K_TC=0 to revert to the scalar path.
_K_TC = int(os.environ.get("OBKV_K_TC", "1")) == 1
# Min T_eff threshold: if a layer's T_eff is smaller than this, fall back to
# scalar (MMA launch overhead > arithmetic savings on tiny segments).
_K_TC_MIN_T_EFF = int(os.environ.get("OBKV_K_TC_MIN_T_EFF", "0"))


def k_diag_reset() -> None:
    _DIAG["index_select_ms"] = 0.0
    _DIAG["kernel_ms"] = 0.0
    _DIAG["n_calls"] = 0


def k_diag_snapshot() -> dict:
    return dict(_DIAG)


def k_diag_enabled() -> bool:
    return _DIAG_ENABLED


D = 128            # head dimension
# GQA_FACTOR is the legacy module-level default kept for backward compat with
# any caller that imports the constant. Wrappers below derive the *real*
# value at runtime from K_scale.shape so MHA (Llama-2-13B, GQA=1), GQA=4
# (Llama-3.1-8B / Mistral-7B / Qwen3-4B), and GQA=8 (Llama-3-70B) all
# produce a correctly-parameterised TC kernel without source edits.
GQA_FACTOR = 4     # legacy constant (Llama-3.1-8B); do NOT use in wrappers
SQRT_D = math.sqrt(float(D))
INV_SQRT_D = 1.0 / SQRT_D  # = 0.08838834764831845


def _runtime_gqa(Q, K_scale):
    """Infer the runtime GQA factor from input shapes.

    Replaces the historical hardcoded ``GQA_FACTOR = 4`` in TC kernel
    wrappers. K_scale is either ``[1, H_kv, 1, D_padded]`` (4D, from
    ``build_packed_layer``) or ``[H_kv, 1, D_padded]`` (3D, post-squeeze).
    """
    H_q = Q.shape[0]
    H_kv = K_scale.shape[1] if K_scale.dim() == 4 else K_scale.shape[0]
    return H_q // H_kv


# ---------------------------------------------------------------------------
# Unified mixed-precision K kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _k_mixed_qk_dot_unified_kernel(
        # Pointers
        Q_ptr,           # [H_q, D_padded]            FP32 (permuted + padded)
        K_2bit_ptr,      # [H_kv, T_alloc, sub_2]     uint8
        K_4bit_ptr,      # [H_kv, T_alloc, sub_4]     uint8
        K_8bit_ptr,      # [H_kv, T_alloc, N_ch_8]    uint8
        K_scale_ptr,     # [H_kv, 1, D_padded]        FP16
        K_zp_ptr,        # [H_kv, 1, D_padded]        FP16
        QK_ptr,          # [H_q, T_alloc]             FP32
        # Dimensions
        T_eff,
        N_ch_2_pad,      # multiple of 4
        N_ch_4_pad,      # multiple of 2
        N_ch_8,
        sub_2,           # = N_ch_2_pad // 4
        sub_4,           # = N_ch_4_pad // 2
        # Strides -- Q: [H_q, D_padded]
        stride_q_h, stride_q_d,
        # Strides -- K_2bit: [H_kv, T_alloc, sub_2]
        stride_k2_h, stride_k2_t, stride_k2_d,
        # Strides -- K_4bit: [H_kv, T_alloc, sub_4]
        stride_k4_h, stride_k4_t, stride_k4_d,
        # Strides -- K_8bit: [H_kv, T_alloc, N_ch_8]
        stride_k8_h, stride_k8_t, stride_k8_d,
        # Strides -- scale/zp: [H_kv, 1, D_padded]
        stride_ks_h, stride_ks_d,
        # Strides -- QK: [H_q, T_alloc]
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        BLOCK_D2_SUB_C: tl.constexpr,   # max sub_2 = 32
        BLOCK_D4_SUB_C: tl.constexpr,   # max sub_4 = 64
        BLOCK_D8_C: tl.constexpr,       # max N_ch_8 = 128
        GQA_FACTOR_C: tl.constexpr,
        HAS_2: tl.constexpr,
        HAS_4: tl.constexpr,
        HAS_8: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_bt = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        q_base = Q_ptr + pid_hq * stride_q_h
        s_base = K_scale_ptr + h_kv * stride_ks_h
        z_base = K_zp_ptr + h_kv * stride_ks_h

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # ================================================================
        # 2-bit segment (quarter-split: 4 channel positions per byte)
        # Gated by HAS_2 (constexpr): empty-segment layers compile the
        # entire block away instead of running it under a zero mask.
        # ================================================================
        if HAS_2:
            d_q = tl.arange(0, BLOCK_D2_SUB_C)
            d_q_mask = d_q < sub_2

            # Q / scale / zp for 4 quarters of the 2-bit group
            q0 = tl.load(q_base + d_q * stride_q_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            q1 = tl.load(q_base + (d_q + sub_2) * stride_q_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            q2 = tl.load(q_base + (d_q + 2 * sub_2) * stride_q_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            q3 = tl.load(q_base + (d_q + 3 * sub_2) * stride_q_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)

            s0 = tl.load(s_base + d_q * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            s1 = tl.load(s_base + (d_q + sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            s2 = tl.load(s_base + (d_q + 2 * sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            s3 = tl.load(s_base + (d_q + 3 * sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)

            z0 = tl.load(z_base + d_q * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            z1 = tl.load(z_base + (d_q + sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            z2 = tl.load(z_base + (d_q + 2 * sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)
            z3 = tl.load(z_base + (d_q + 3 * sub_2) * stride_ks_d,
                         mask=d_q_mask, other=0.0).to(tl.float32)

            qs0 = q0 * s0
            qs1 = q1 * s1
            qs2 = q2 * s2
            qs3 = q3 * s3
            bias2 = (tl.sum(z0 * qs0) + tl.sum(z1 * qs1)
                     + tl.sum(z2 * qs2) + tl.sum(z3 * qs3))

            k2_ptrs = (K_2bit_ptr + h_kv * stride_k2_h
                       + t_offs[:, None] * stride_k2_t
                       + d_q[None, :] * stride_k2_d)
            packed2 = tl.load(k2_ptrs,
                              mask=(t_mask[:, None] & d_q_mask[None, :]),
                              other=0)
            p2_i32 = packed2.to(tl.int32)
            k_q0 = (p2_i32 & 0x3).to(tl.float32)
            k_q1 = ((p2_i32 >> 2) & 0x3).to(tl.float32)
            k_q2 = ((p2_i32 >> 4) & 0x3).to(tl.float32)
            k_q3 = ((p2_i32 >> 6) & 0x3).to(tl.float32)

            acc += tl.sum(k_q0 * qs0[None, :], axis=1)
            acc += tl.sum(k_q1 * qs1[None, :], axis=1)
            acc += tl.sum(k_q2 * qs2[None, :], axis=1)
            acc += tl.sum(k_q3 * qs3[None, :], axis=1)
            acc -= bias2

        # ================================================================
        # 4-bit segment (half-split: 2 channel positions per byte)
        # ================================================================
        if HAS_4:
            d_lo = tl.arange(0, BLOCK_D4_SUB_C)
            d_lo_mask = d_lo < sub_4
            off4 = N_ch_2_pad

            ql = tl.load(q_base + (d_lo + off4) * stride_q_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)
            qh = tl.load(q_base + (d_lo + off4 + sub_4) * stride_q_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)

            sl = tl.load(s_base + (d_lo + off4) * stride_ks_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)
            sh = tl.load(s_base + (d_lo + off4 + sub_4) * stride_ks_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)
            zl = tl.load(z_base + (d_lo + off4) * stride_ks_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)
            zh = tl.load(z_base + (d_lo + off4 + sub_4) * stride_ks_d,
                         mask=d_lo_mask, other=0.0).to(tl.float32)

            qsl = ql * sl
            qsh = qh * sh
            bias4 = tl.sum(zl * qsl) + tl.sum(zh * qsh)

            k4_ptrs = (K_4bit_ptr + h_kv * stride_k4_h
                       + t_offs[:, None] * stride_k4_t
                       + d_lo[None, :] * stride_k4_d)
            packed4 = tl.load(k4_ptrs,
                              mask=(t_mask[:, None] & d_lo_mask[None, :]),
                              other=0)
            p4_i32 = packed4.to(tl.int32)
            k_lo = (p4_i32 & 0xF).to(tl.float32)
            k_hi = ((p4_i32 >> 4) & 0xF).to(tl.float32)

            acc += tl.sum(k_lo * qsl[None, :], axis=1)
            acc += tl.sum(k_hi * qsh[None, :], axis=1)
            acc -= bias4

        # ================================================================
        # 8-bit segment (raw uint8, per-channel dequant)
        # ================================================================
        if HAS_8:
            d_f = tl.arange(0, BLOCK_D8_C)
            d_f_mask = d_f < N_ch_8
            off8 = N_ch_2_pad + N_ch_4_pad

            q8 = tl.load(q_base + (d_f + off8) * stride_q_d,
                         mask=d_f_mask, other=0.0).to(tl.float32)
            s8 = tl.load(s_base + (d_f + off8) * stride_ks_d,
                         mask=d_f_mask, other=0.0).to(tl.float32)
            z8 = tl.load(z_base + (d_f + off8) * stride_ks_d,
                         mask=d_f_mask, other=0.0).to(tl.float32)

            qs8 = q8 * s8
            bias8 = tl.sum(z8 * qs8)

            k8_ptrs = (K_8bit_ptr + h_kv * stride_k8_h
                       + t_offs[:, None] * stride_k8_t
                       + d_f[None, :] * stride_k8_d)
            packed8 = tl.load(k8_ptrs,
                              mask=(t_mask[:, None] & d_f_mask[None, :]),
                              other=0)
            k8_f = packed8.to(tl.float32)
            acc += tl.sum(k8_f * qs8[None, :], axis=1)
            acc -= bias8

        # Scale by 1/sqrt(D)
        acc = acc * 0.08838834764831845

        # Store
        out_ptrs = QK_ptr + pid_hq * stride_qk_h + t_offs * stride_qk_t
        tl.store(out_ptrs, acc, mask=t_mask)


BLOCK_T_DEFAULT = 64

# ---------------------------------------------------------------------------
# E2: FP16 tensor-core K kernels (A100 SM 80 compatible).
# Shared design:
#   * Grid = (H_kv, ceil(T_eff / BLOCK_T)). M_TC=16 packs the 4 GQA rows for
#     one h_kv plus 12 rows of padding; m_valid mask zeros padded rows.
#   * Q / scale / zp loaded in FP32 so the bias trick keeps its numerical
#     headroom; `qs = q * scale` then cast FP32->FP16 for the mma input.
#   * K codes: raw 2/4/8-bit integer values cast to FP16 (exact for up to
#     2048) via the helpers in triton_dequant_utils.
#   * tl.dot(qs_f16[M,K], k_codes_f16[K,N]) -> acc_f32[M,N]; subtract bias
#     (broadcast [M,1]) and apply 1/sqrt(D) before store.
#   * Store: only rows m<GQA_FACTOR are valid; T-mask guards the tail.
# ---------------------------------------------------------------------------
if HAS_TRITON:
    from obkv_accel.triton_dequant_utils import (  # noqa: E402
        unpack_2bit_k_codes_kt as _k_unpack_2bit,
        unpack_4bit_k_codes_kt as _k_unpack_4bit,
        unpack_8bit_k_codes_kt as _k_unpack_8bit,
    )

    @triton.jit
    def _k_mixed_unified_tc_kernel(
        # Pointers
        Q_ptr,            # [H_q, D_padded]           FP32
        K_2bit_ptr,       # [H_kv, T_alloc, sub_2]    uint8
        K_4bit_ptr,       # [H_kv, T_alloc, sub_4]    uint8
        K_8bit_ptr,       # [H_kv, T_alloc, N_ch_8]   uint8
        K_scale_ptr,      # [H_kv, 1, D_padded]       FP16
        K_zp_ptr,
        QK_ptr,           # [H_q, T_alloc]            FP32
        # Dimensions
        T_eff,
        N_ch_2_pad,
        N_ch_4_pad,
        N_ch_8,
        sub_2,
        sub_4,
        # Strides
        stride_q_h, stride_q_d,
        stride_k2_h, stride_k2_t, stride_k2_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_k8_h, stride_k8_t, stride_k8_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        BLOCK_K_2: tl.constexpr,          # = 128 max channels in 2-bit seg
        BLOCK_K_4: tl.constexpr,          # = 128
        BLOCK_K_8: tl.constexpr,          # = 16 (hybrid) or 128 (unified)
        M_TC_C: tl.constexpr,
        GQA_FACTOR_C: tl.constexpr,
        HAS_2: tl.constexpr,
        HAS_4: tl.constexpr,
        HAS_8: tl.constexpr,
    ):
        pid_hkv = tl.program_id(0)
        pid_bt = tl.program_id(1)

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        m = tl.arange(0, M_TC_C)
        m_valid = m < GQA_FACTOR_C
        hq = pid_hkv * GQA_FACTOR_C + m
        hq_safe = tl.where(m_valid, hq, pid_hkv * GQA_FACTOR_C)

        q_base_h = Q_ptr + hq_safe[:, None] * stride_q_h
        s_base = K_scale_ptr + pid_hkv * stride_ks_h
        z_base = K_zp_ptr + pid_hkv * stride_ks_h

        acc = tl.zeros((M_TC_C, BLOCK_T), dtype=tl.float32)
        bias_total = tl.zeros((M_TC_C,), dtype=tl.float32)

        # ================= 2-bit segment =================
        if HAS_2:
            d2 = tl.arange(0, BLOCK_K_2)
            d2_mask = d2 < N_ch_2_pad
            # Q / scale / zp
            q2 = tl.load(q_base_h + d2[None, :] * stride_q_d,
                         mask=m_valid[:, None] & d2_mask[None, :],
                         other=0.0).to(tl.float32)
            s2 = tl.load(s_base + d2 * stride_ks_d,
                         mask=d2_mask, other=0.0).to(tl.float32)
            z2 = tl.load(z_base + d2 * stride_ks_d,
                         mask=d2_mask, other=0.0).to(tl.float32)
            qs2_f32 = q2 * s2[None, :]
            bias_total = bias_total + tl.sum(z2[None, :] * qs2_f32, axis=1)
            qs2_f16 = qs2_f32.to(tl.float16)
            # K codes [BLOCK_K_2, BLOCK_T]
            # channel c (< N_ch_2_pad): byte idx = c % sub_2, shift = (c//sub_2)*2
            sub_2_safe = tl.maximum(sub_2, 1)
            d2_byte = d2 % sub_2_safe
            d2_shift = (d2 // sub_2_safe) * 2
            k2_ptrs = (K_2bit_ptr + pid_hkv * stride_k2_h
                       + t_offs[None, :] * stride_k2_t
                       + d2_byte[:, None] * stride_k2_d)
            packed = tl.load(k2_ptrs,
                             mask=d2_mask[:, None] & t_mask[None, :],
                             other=0).to(tl.int32)
            k2_codes = ((packed >> d2_shift[:, None]) & 0x3).to(tl.float16)
            acc = tl.dot(qs2_f16, k2_codes, acc=acc, out_dtype=tl.float32)

        # ================= 4-bit segment =================
        if HAS_4:
            d4 = tl.arange(0, BLOCK_K_4)
            d4_mask = d4 < N_ch_4_pad
            q4 = tl.load(q_base_h + (d4 + N_ch_2_pad)[None, :] * stride_q_d,
                         mask=m_valid[:, None] & d4_mask[None, :],
                         other=0.0).to(tl.float32)
            s4 = tl.load(s_base + (d4 + N_ch_2_pad) * stride_ks_d,
                         mask=d4_mask, other=0.0).to(tl.float32)
            z4 = tl.load(z_base + (d4 + N_ch_2_pad) * stride_ks_d,
                         mask=d4_mask, other=0.0).to(tl.float32)
            qs4_f32 = q4 * s4[None, :]
            bias_total = bias_total + tl.sum(z4[None, :] * qs4_f32, axis=1)
            qs4_f16 = qs4_f32.to(tl.float16)
            sub_4_safe = tl.maximum(sub_4, 1)
            d4_byte = d4 % sub_4_safe
            d4_shift = (d4 // sub_4_safe) * 4
            k4_ptrs = (K_4bit_ptr + pid_hkv * stride_k4_h
                       + t_offs[None, :] * stride_k4_t
                       + d4_byte[:, None] * stride_k4_d)
            packed4 = tl.load(k4_ptrs,
                              mask=d4_mask[:, None] & t_mask[None, :],
                              other=0).to(tl.int32)
            k4_codes = ((packed4 >> d4_shift[:, None]) & 0xF).to(tl.float16)
            acc = tl.dot(qs4_f16, k4_codes, acc=acc, out_dtype=tl.float32)

        # ================= 8-bit segment =================
        if HAS_8:
            d8 = tl.arange(0, BLOCK_K_8)
            d8_mask = d8 < N_ch_8
            off8 = N_ch_2_pad + N_ch_4_pad
            q8 = tl.load(q_base_h + (d8 + off8)[None, :] * stride_q_d,
                         mask=m_valid[:, None] & d8_mask[None, :],
                         other=0.0).to(tl.float32)
            s8 = tl.load(s_base + (d8 + off8) * stride_ks_d,
                         mask=d8_mask, other=0.0).to(tl.float32)
            z8 = tl.load(z_base + (d8 + off8) * stride_ks_d,
                         mask=d8_mask, other=0.0).to(tl.float32)
            qs8_f32 = q8 * s8[None, :]
            bias_total = bias_total + tl.sum(z8[None, :] * qs8_f32, axis=1)
            qs8_f16 = qs8_f32.to(tl.float16)
            k8_ptrs = (K_8bit_ptr + pid_hkv * stride_k8_h
                       + t_offs[None, :] * stride_k8_t
                       + d8[:, None] * stride_k8_d)
            k8_raw = tl.load(k8_ptrs,
                             mask=d8_mask[:, None] & t_mask[None, :],
                             other=0).to(tl.float16)
            acc = tl.dot(qs8_f16, k8_raw, acc=acc, out_dtype=tl.float32)

        acc = (acc - bias_total[:, None]) * 0.08838834764831845

        out_ptrs = (QK_ptr + hq_safe[:, None] * stride_qk_h
                    + t_offs[None, :] * stride_qk_t)
        tl.store(out_ptrs, acc,
                 mask=m_valid[:, None] & t_mask[None, :])

    @triton.jit
    def _k_uniform4_tc_kernel(
        Q_ptr,             # [H_q, D]                FP32 (permuted)
        K_4bit_ptr,        # [H_kv, T_alloc, D//2]   uint8
        K_scale_ptr,       # [H_kv, 1, D]            FP16
        K_zp_ptr,
        QK_ptr,            # [H_q, T_alloc]          FP32
        T_eff,
        stride_q_h, stride_q_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        BLOCK_T: tl.constexpr,
        D_C: tl.constexpr,              # = 128
        D_PACKED_C: tl.constexpr,       # = 64
        M_TC_C: tl.constexpr,           # = 16
        GQA_FACTOR_C: tl.constexpr,     # = 4
    ):
        pid_hkv = tl.program_id(0)
        pid_bt = tl.program_id(1)

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        m = tl.arange(0, M_TC_C)
        m_valid = m < GQA_FACTOR_C
        hq = pid_hkv * GQA_FACTOR_C + m
        hq_safe = tl.where(m_valid, hq, pid_hkv * GQA_FACTOR_C)

        # Load Q / scale / zp in FP32.
        d = tl.arange(0, D_C)
        q_fp32 = tl.load(
            Q_ptr + hq_safe[:, None] * stride_q_h + d[None, :] * stride_q_d,
            mask=m_valid[:, None], other=0.0,
        ).to(tl.float32)                          # [M, D]
        scale = tl.load(K_scale_ptr + pid_hkv * stride_ks_h + d * stride_ks_d
                        ).to(tl.float32)          # [D]
        zp = tl.load(K_zp_ptr + pid_hkv * stride_ks_h + d * stride_ks_d
                     ).to(tl.float32)             # [D]

        qs_f32 = q_fp32 * scale[None, :]          # [M, D]
        bias = tl.sum(zp[None, :] * qs_f32, axis=1, keep_dims=True)  # [M, 1]
        qs_f16 = qs_f32.to(tl.float16)

        # Build K codes tile [D=128, BLOCK_T=64] FP16.
        # Half-split: byte K_4bit[h_kv, t, d_byte] low nibble -> channel d_byte,
        # high nibble -> channel d_byte + D_PACKED_C.
        d_full = tl.arange(0, D_C)
        d_byte = d_full % D_PACKED_C              # 0..63, repeats twice
        d_shift = (d_full // D_PACKED_C) * 4      # 0 or 4
        k4_ptrs = (K_4bit_ptr + pid_hkv * stride_k4_h
                   + t_offs[None, :] * stride_k4_t
                   + d_byte[:, None] * stride_k4_d)
        packed = tl.load(k4_ptrs, mask=t_mask[None, :], other=0).to(tl.int32)
        k_codes = ((packed >> d_shift[:, None]) & 0xF).to(tl.float16)  # [D, BLOCK_T]

        acc = tl.dot(qs_f16, k_codes, out_dtype=tl.float32)            # [M, BLOCK_T]
        acc = acc - bias
        acc = acc * 0.08838834764831845

        # Store 4 valid rows only; padded rows never materialise.
        out_ptrs = (QK_ptr + hq_safe[:, None] * stride_qk_h
                    + t_offs[None, :] * stride_qk_t)
        tl.store(out_ptrs, acc,
                 mask=m_valid[:, None] & t_mask[None, :])


# ---------------------------------------------------------------------------
# Minimal uniform-4bit K kernel — structural A/B against the unified kernel.
# Same dequant + half-split unpack math as the HAS_4-only branch of the unified
# kernel, but stripped of HAS_2/HAS_8 arms, extra pointers, and segment-offset
# arithmetic. Exists so we can isolate "unified kernel params/structure" from
# "4-bit unpack+dot math" as the cause of the 2× slowdown vs old kernel.
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _k_uniform4_qk_dot_kernel(
        # Pointers
        Q_ptr,           # [H_q, D=128]               FP32 (already permuted)
        K_4bit_ptr,      # [H_kv, T_alloc, D//2=64]   uint8 half-split
        K_scale_ptr,     # [H_kv, 1, D]               FP16 per-channel
        K_zp_ptr,        # [H_kv, 1, D]               FP16 per-channel
        QK_ptr,          # [H_q, T_alloc]             FP32
        # Dimensions
        T_eff,
        # Strides
        stride_q_h, stride_q_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        D_PACKED_C: tl.constexpr,    # = 64
        D_C: tl.constexpr,            # = 128
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_bt = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C

        # Channel offsets for half-split layout
        d_lo = tl.arange(0, D_PACKED_C)             # [0..63]
        d_hi = d_lo + D_PACKED_C                    # [64..127]

        # Load Q, scale, zp (unmasked — D=128 always)
        q_base = Q_ptr + pid_hq * stride_q_h
        s_base = K_scale_ptr + h_kv * stride_ks_h
        z_base = K_zp_ptr + h_kv * stride_ks_h

        q_lo = tl.load(q_base + d_lo * stride_q_d).to(tl.float32)
        q_hi = tl.load(q_base + d_hi * stride_q_d).to(tl.float32)
        s_lo = tl.load(s_base + d_lo * stride_ks_d).to(tl.float32)
        s_hi = tl.load(s_base + d_hi * stride_ks_d).to(tl.float32)
        z_lo = tl.load(z_base + d_lo * stride_ks_d).to(tl.float32)
        z_hi = tl.load(z_base + d_hi * stride_ks_d).to(tl.float32)

        qs_lo = q_lo * s_lo
        qs_hi = q_hi * s_hi
        bias = tl.sum(z_lo * qs_lo) + tl.sum(z_hi * qs_hi)

        # Token tile
        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        packed_ptrs = (K_4bit_ptr + h_kv * stride_k4_h
                       + t_offs[:, None] * stride_k4_t
                       + d_lo[None, :] * stride_k4_d)
        packed = tl.load(packed_ptrs, mask=t_mask[:, None], other=0)
        packed_i32 = packed.to(tl.int32)
        k_lo_f = (packed_i32 & 0xF).to(tl.float32)
        k_hi_f = ((packed_i32 >> 4) & 0xF).to(tl.float32)

        acc = tl.sum(k_lo_f * qs_lo[None, :], axis=1)
        acc += tl.sum(k_hi_f * qs_hi[None, :], axis=1)
        acc = (acc - bias) * 0.08838834764831845

        out_ptrs = QK_ptr + pid_hq * stride_qk_h + t_offs * stride_qk_t
        tl.store(out_ptrs, acc, mask=t_mask)


# ---------------------------------------------------------------------------
# Specialised 2-segment kernels, written in the same minimal style as MIN_4.
# Covers the two patterns actually produced by the kr=0.5 knapsack across
# 4K–128K contexts (confirmed by an exhaustive K_HAS_X_DIAG scan):
#   - mixed24: HAS_2 + HAS_4   (8K, 16K, 32K, 64K, 128K)
#   - mixed48: HAS_4 + HAS_8   (4K only; N_ch_8 very small, typically ≤ 8)
# Three-segment or single-segment patterns fall back to _k_mixed_qk_dot_unified_kernel.
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _k_mixed24_qk_dot_kernel(
        # Pointers
        Q_ptr,            # [H_q, D_padded]             FP32 (permuted+padded)
        K_2bit_ptr,       # [H_kv, T_alloc, sub_2]      uint8 quarter-split
        K_4bit_ptr,       # [H_kv, T_alloc, sub_4]      uint8 half-split
        K_scale_ptr,      # [H_kv, 1, D_padded]         FP16 per-channel
        K_zp_ptr,         # [H_kv, 1, D_padded]         FP16 per-channel
        QK_ptr,           # [H_q, T_alloc]              FP32
        # Dimensions
        T_eff,
        N_ch_2_pad,       # multiple of 4 (=4*sub_2)
        sub_2,            # = N_ch_2_pad // 4
        sub_4,            # = N_ch_4_pad // 2
        # Strides
        stride_q_h, stride_q_d,
        stride_k2_h, stride_k2_t, stride_k2_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        BLOCK_D2_SUB_C: tl.constexpr,    # max sub_2
        BLOCK_D4_SUB_C: tl.constexpr,    # max sub_4
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_bt = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        q_base = Q_ptr + pid_hq * stride_q_h
        s_base = K_scale_ptr + h_kv * stride_ks_h
        z_base = K_zp_ptr + h_kv * stride_ks_h

        # ---- 2-bit segment (channels [0, N_ch_2_pad)) ----
        d_q = tl.arange(0, BLOCK_D2_SUB_C)
        d_q_mask = d_q < sub_2

        q0 = tl.load(q_base + d_q * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + (d_q + sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_base + (d_q + 2 * sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q3 = tl.load(q_base + (d_q + 3 * sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)

        s0 = tl.load(s_base + d_q * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s1 = tl.load(s_base + (d_q + sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s2 = tl.load(s_base + (d_q + 2 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s3 = tl.load(s_base + (d_q + 3 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)

        z0 = tl.load(z_base + d_q * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z1 = tl.load(z_base + (d_q + sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z2 = tl.load(z_base + (d_q + 2 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z3 = tl.load(z_base + (d_q + 3 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)

        qs0 = q0 * s0
        qs1 = q1 * s1
        qs2 = q2 * s2
        qs3 = q3 * s3
        bias2 = (tl.sum(z0 * qs0) + tl.sum(z1 * qs1)
                 + tl.sum(z2 * qs2) + tl.sum(z3 * qs3))

        k2_ptrs = (K_2bit_ptr + h_kv * stride_k2_h
                   + t_offs[:, None] * stride_k2_t
                   + d_q[None, :] * stride_k2_d)
        packed2 = tl.load(k2_ptrs,
                          mask=(t_mask[:, None] & d_q_mask[None, :]),
                          other=0)
        p2_i32 = packed2.to(tl.int32)
        k_q0 = (p2_i32 & 0x3).to(tl.float32)
        k_q1 = ((p2_i32 >> 2) & 0x3).to(tl.float32)
        k_q2 = ((p2_i32 >> 4) & 0x3).to(tl.float32)
        k_q3 = ((p2_i32 >> 6) & 0x3).to(tl.float32)

        acc = tl.sum(k_q0 * qs0[None, :], axis=1)
        acc += tl.sum(k_q1 * qs1[None, :], axis=1)
        acc += tl.sum(k_q2 * qs2[None, :], axis=1)
        acc += tl.sum(k_q3 * qs3[None, :], axis=1)
        acc -= bias2

        # ---- 4-bit segment (channels [N_ch_2_pad, N_ch_2_pad + N_ch_4_pad)) ----
        d_lo = tl.arange(0, BLOCK_D4_SUB_C)
        d_lo_mask = d_lo < sub_4
        off4 = N_ch_2_pad

        ql = tl.load(q_base + (d_lo + off4) * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        qh = tl.load(q_base + (d_lo + off4 + sub_4) * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)

        sl = tl.load(s_base + (d_lo + off4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        sh = tl.load(s_base + (d_lo + off4 + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zl = tl.load(z_base + (d_lo + off4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zh = tl.load(z_base + (d_lo + off4 + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)

        qsl = ql * sl
        qsh = qh * sh
        bias4 = tl.sum(zl * qsl) + tl.sum(zh * qsh)

        k4_ptrs = (K_4bit_ptr + h_kv * stride_k4_h
                   + t_offs[:, None] * stride_k4_t
                   + d_lo[None, :] * stride_k4_d)
        packed4 = tl.load(k4_ptrs,
                          mask=(t_mask[:, None] & d_lo_mask[None, :]),
                          other=0)
        p4_i32 = packed4.to(tl.int32)
        k_lo = (p4_i32 & 0xF).to(tl.float32)
        k_hi = ((p4_i32 >> 4) & 0xF).to(tl.float32)

        acc += tl.sum(k_lo * qsl[None, :], axis=1)
        acc += tl.sum(k_hi * qsh[None, :], axis=1)
        acc -= bias4

        acc = acc * 0.08838834764831845

        out_ptrs = QK_ptr + pid_hq * stride_qk_h + t_offs * stride_qk_t
        tl.store(out_ptrs, acc, mask=t_mask)


    @triton.jit
    def _k_mixed48_qk_dot_kernel(
        # Pointers
        Q_ptr,
        K_4bit_ptr,
        K_8bit_ptr,
        K_scale_ptr,
        K_zp_ptr,
        QK_ptr,
        # Dimensions
        T_eff,
        N_ch_4_pad,
        N_ch_8,
        sub_4,            # = N_ch_4_pad // 2
        # Strides
        stride_q_h, stride_q_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_k8_h, stride_k8_t, stride_k8_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        BLOCK_D4_SUB_C: tl.constexpr,    # max sub_4
        BLOCK_D8_C: tl.constexpr,        # max N_ch_8
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_bt = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        q_base = Q_ptr + pid_hq * stride_q_h
        s_base = K_scale_ptr + h_kv * stride_ks_h
        z_base = K_zp_ptr + h_kv * stride_ks_h

        # ---- 4-bit segment (channels [0, N_ch_4_pad)) ----
        d_lo = tl.arange(0, BLOCK_D4_SUB_C)
        d_lo_mask = d_lo < sub_4

        ql = tl.load(q_base + d_lo * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        qh = tl.load(q_base + (d_lo + sub_4) * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        sl = tl.load(s_base + d_lo * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        sh = tl.load(s_base + (d_lo + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zl = tl.load(z_base + d_lo * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zh = tl.load(z_base + (d_lo + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)

        qsl = ql * sl
        qsh = qh * sh
        bias4 = tl.sum(zl * qsl) + tl.sum(zh * qsh)

        k4_ptrs = (K_4bit_ptr + h_kv * stride_k4_h
                   + t_offs[:, None] * stride_k4_t
                   + d_lo[None, :] * stride_k4_d)
        packed4 = tl.load(k4_ptrs,
                          mask=(t_mask[:, None] & d_lo_mask[None, :]),
                          other=0)
        p4_i32 = packed4.to(tl.int32)
        k_lo = (p4_i32 & 0xF).to(tl.float32)
        k_hi = ((p4_i32 >> 4) & 0xF).to(tl.float32)

        acc = tl.sum(k_lo * qsl[None, :], axis=1)
        acc += tl.sum(k_hi * qsh[None, :], axis=1)
        acc -= bias4

        # ---- 8-bit segment (channels [N_ch_4_pad, N_ch_4_pad + N_ch_8)) ----
        d_f = tl.arange(0, BLOCK_D8_C)
        d_f_mask = d_f < N_ch_8
        off8 = N_ch_4_pad

        q8 = tl.load(q_base + (d_f + off8) * stride_q_d, mask=d_f_mask, other=0.0).to(tl.float32)
        s8 = tl.load(s_base + (d_f + off8) * stride_ks_d, mask=d_f_mask, other=0.0).to(tl.float32)
        z8 = tl.load(z_base + (d_f + off8) * stride_ks_d, mask=d_f_mask, other=0.0).to(tl.float32)

        qs8 = q8 * s8
        bias8 = tl.sum(z8 * qs8)

        k8_ptrs = (K_8bit_ptr + h_kv * stride_k8_h
                   + t_offs[:, None] * stride_k8_t
                   + d_f[None, :] * stride_k8_d)
        packed8 = tl.load(k8_ptrs,
                          mask=(t_mask[:, None] & d_f_mask[None, :]),
                          other=0)
        k8_f = packed8.to(tl.float32)
        acc += tl.sum(k8_f * qs8[None, :], axis=1)
        acc -= bias8

        acc = acc * 0.08838834764831845

        out_ptrs = QK_ptr + pid_hq * stride_qk_h + t_offs * stride_qk_t
        tl.store(out_ptrs, acc, mask=t_mask)


    @triton.jit
    def _k_mixed248_qk_dot_kernel(
        # Pointers
        Q_ptr,            # [H_q, D_padded]             FP32 (permuted+padded)
        K_2bit_ptr,       # [H_kv, T_alloc, sub_2]      uint8 quarter-split
        K_4bit_ptr,       # [H_kv, T_alloc, sub_4]      uint8 half-split
        K_8bit_ptr,       # [H_kv, T_alloc, N_ch_8]     uint8 full
        K_scale_ptr,      # [H_kv, 1, D_padded]         FP16 per-channel
        K_zp_ptr,         # [H_kv, 1, D_padded]         FP16 per-channel
        QK_ptr,           # [H_q, T_alloc]              FP32
        # Dimensions
        T_eff,
        N_ch_2_pad,       # multiple of 4 (=4*sub_2)
        N_ch_4_pad,       # multiple of 2 (=2*sub_4)
        N_ch_8,
        sub_2,            # = N_ch_2_pad // 4
        sub_4,            # = N_ch_4_pad // 2
        # Strides
        stride_q_h, stride_q_d,
        stride_k2_h, stride_k2_t, stride_k2_d,
        stride_k4_h, stride_k4_t, stride_k4_d,
        stride_k8_h, stride_k8_t, stride_k8_d,
        stride_ks_h, stride_ks_d,
        stride_qk_h, stride_qk_t,
        # Constexprs
        BLOCK_T: tl.constexpr,
        BLOCK_D2_SUB_C: tl.constexpr,
        BLOCK_D4_SUB_C: tl.constexpr,
        BLOCK_D8_C: tl.constexpr,
        GQA_FACTOR_C: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_bt = tl.program_id(1)
        h_kv = pid_hq // GQA_FACTOR_C

        t_start = pid_bt * BLOCK_T
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T_eff

        q_base = Q_ptr + pid_hq * stride_q_h
        s_base = K_scale_ptr + h_kv * stride_ks_h
        z_base = K_zp_ptr + h_kv * stride_ks_h

        # ---- 2-bit segment (channels [0, N_ch_2_pad)) ----
        d_q = tl.arange(0, BLOCK_D2_SUB_C)
        d_q_mask = d_q < sub_2

        q0 = tl.load(q_base + d_q * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + (d_q + sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_base + (d_q + 2 * sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)
        q3 = tl.load(q_base + (d_q + 3 * sub_2) * stride_q_d, mask=d_q_mask, other=0.0).to(tl.float32)

        s0 = tl.load(s_base + d_q * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s1 = tl.load(s_base + (d_q + sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s2 = tl.load(s_base + (d_q + 2 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        s3 = tl.load(s_base + (d_q + 3 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)

        z0 = tl.load(z_base + d_q * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z1 = tl.load(z_base + (d_q + sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z2 = tl.load(z_base + (d_q + 2 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)
        z3 = tl.load(z_base + (d_q + 3 * sub_2) * stride_ks_d, mask=d_q_mask, other=0.0).to(tl.float32)

        qs0 = q0 * s0
        qs1 = q1 * s1
        qs2 = q2 * s2
        qs3 = q3 * s3
        bias2 = (tl.sum(z0 * qs0) + tl.sum(z1 * qs1)
                 + tl.sum(z2 * qs2) + tl.sum(z3 * qs3))

        k2_ptrs = (K_2bit_ptr + h_kv * stride_k2_h
                   + t_offs[:, None] * stride_k2_t
                   + d_q[None, :] * stride_k2_d)
        packed2 = tl.load(k2_ptrs,
                          mask=(t_mask[:, None] & d_q_mask[None, :]),
                          other=0)
        p2_i32 = packed2.to(tl.int32)
        k_q0 = (p2_i32 & 0x3).to(tl.float32)
        k_q1 = ((p2_i32 >> 2) & 0x3).to(tl.float32)
        k_q2 = ((p2_i32 >> 4) & 0x3).to(tl.float32)
        k_q3 = ((p2_i32 >> 6) & 0x3).to(tl.float32)

        acc = tl.sum(k_q0 * qs0[None, :], axis=1)
        acc += tl.sum(k_q1 * qs1[None, :], axis=1)
        acc += tl.sum(k_q2 * qs2[None, :], axis=1)
        acc += tl.sum(k_q3 * qs3[None, :], axis=1)
        acc -= bias2

        # ---- 4-bit segment (channels [N_ch_2_pad, N_ch_2_pad + N_ch_4_pad)) ----
        d_lo = tl.arange(0, BLOCK_D4_SUB_C)
        d_lo_mask = d_lo < sub_4
        off4 = N_ch_2_pad

        ql = tl.load(q_base + (d_lo + off4) * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        qh = tl.load(q_base + (d_lo + off4 + sub_4) * stride_q_d, mask=d_lo_mask, other=0.0).to(tl.float32)

        sl = tl.load(s_base + (d_lo + off4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        sh = tl.load(s_base + (d_lo + off4 + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zl = tl.load(z_base + (d_lo + off4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)
        zh = tl.load(z_base + (d_lo + off4 + sub_4) * stride_ks_d, mask=d_lo_mask, other=0.0).to(tl.float32)

        qsl = ql * sl
        qsh = qh * sh
        bias4 = tl.sum(zl * qsl) + tl.sum(zh * qsh)

        k4_ptrs = (K_4bit_ptr + h_kv * stride_k4_h
                   + t_offs[:, None] * stride_k4_t
                   + d_lo[None, :] * stride_k4_d)
        packed4 = tl.load(k4_ptrs,
                          mask=(t_mask[:, None] & d_lo_mask[None, :]),
                          other=0)
        p4_i32 = packed4.to(tl.int32)
        k_lo = (p4_i32 & 0xF).to(tl.float32)
        k_hi = ((p4_i32 >> 4) & 0xF).to(tl.float32)

        acc += tl.sum(k_lo * qsl[None, :], axis=1)
        acc += tl.sum(k_hi * qsh[None, :], axis=1)
        acc -= bias4

        # ---- 8-bit segment (channels [N_ch_2_pad+N_ch_4_pad, +N_ch_8)) ----
        d_f = tl.arange(0, BLOCK_D8_C)
        d_f_mask = d_f < N_ch_8
        off8 = N_ch_2_pad + N_ch_4_pad

        q8 = tl.load(q_base + (d_f + off8) * stride_q_d, mask=d_f_mask, other=0.0).to(tl.float32)
        s8 = tl.load(s_base + (d_f + off8) * stride_ks_d, mask=d_f_mask, other=0.0).to(tl.float32)
        z8 = tl.load(z_base + (d_f + off8) * stride_ks_d, mask=d_f_mask, other=0.0).to(tl.float32)

        qs8 = q8 * s8
        bias8 = tl.sum(z8 * qs8)

        k8_ptrs = (K_8bit_ptr + h_kv * stride_k8_h
                   + t_offs[:, None] * stride_k8_t
                   + d_f[None, :] * stride_k8_d)
        packed8 = tl.load(k8_ptrs,
                          mask=(t_mask[:, None] & d_f_mask[None, :]),
                          other=0)
        k8_f = packed8.to(tl.float32)
        acc += tl.sum(k8_f * qs8[None, :], axis=1)
        acc -= bias8

        acc = acc * 0.08838834764831845

        out_ptrs = QK_ptr + pid_hq * stride_qk_h + t_offs * stride_qk_t
        tl.store(out_ptrs, acc, mask=t_mask)


def k_mixed248_qk_dot_into(
    Q: torch.Tensor,           # [H_q, D_padded] FP32 permuted+padded
    K_2bit: torch.Tensor,      # [H_kv, T_alloc, sub_2] uint8
    K_4bit: torch.Tensor,      # [H_kv, T_alloc, sub_4] uint8
    K_8bit: torch.Tensor,      # [H_kv, T_alloc, N_ch_8] uint8
    K_scale: torch.Tensor,     # [H_kv, 1, D_padded] FP16
    K_zp: torch.Tensor,        # [H_kv, 1, D_padded] FP16
    T_eff: int,
    N_ch_2_pad: int,
    N_ch_4_pad: int,
    N_ch_8: int,
    sub_2: int,
    sub_4: int,
    out: torch.Tensor,         # [H_q, T_alloc] FP32
) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_q, triton.cdiv(T_eff, BLOCK_T))
    _k_mixed248_qk_dot_kernel[grid](
        Q, K_2bit, K_4bit, K_8bit, K_scale, K_zp, out,
        T_eff,
        N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4,
        Q.stride(0), Q.stride(1),
        K_2bit.stride(0), K_2bit.stride(1), K_2bit.stride(2),
        K_4bit.stride(0), K_4bit.stride(1), K_4bit.stride(2),
        K_8bit.stride(0), K_8bit.stride(1), K_8bit.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_D2_SUB_C=32,
        BLOCK_D4_SUB_C=64,
        BLOCK_D8_C=16,
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        num_stages=4,
    )


def k_mixed24_qk_dot_into(
    Q: torch.Tensor,           # [H_q, D_padded] FP32 permuted+padded
    K_2bit: torch.Tensor,      # [H_kv, T_alloc, sub_2] uint8
    K_4bit: torch.Tensor,      # [H_kv, T_alloc, sub_4] uint8
    K_scale: torch.Tensor,     # [H_kv, 1, D_padded] FP16
    K_zp: torch.Tensor,        # [H_kv, 1, D_padded] FP16
    T_eff: int,
    N_ch_2_pad: int,
    sub_2: int,
    sub_4: int,
    out: torch.Tensor,         # [H_q, T_alloc] FP32
) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_q, triton.cdiv(T_eff, BLOCK_T))
    _k_mixed24_qk_dot_kernel[grid](
        Q, K_2bit, K_4bit, K_scale, K_zp, out,
        T_eff,
        N_ch_2_pad, sub_2, sub_4,
        Q.stride(0), Q.stride(1),
        K_2bit.stride(0), K_2bit.stride(1), K_2bit.stride(2),
        K_4bit.stride(0), K_4bit.stride(1), K_4bit.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_D2_SUB_C=32,
        BLOCK_D4_SUB_C=64,
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        num_stages=4,
    )


def k_mixed48_qk_dot_into(
    Q: torch.Tensor,           # [H_q, D_padded] FP32 permuted+padded
    K_4bit: torch.Tensor,      # [H_kv, T_alloc, sub_4] uint8
    K_8bit: torch.Tensor,      # [H_kv, T_alloc, N_ch_8] uint8
    K_scale: torch.Tensor,     # [H_kv, 1, D_padded] FP16
    K_zp: torch.Tensor,        # [H_kv, 1, D_padded] FP16
    T_eff: int,
    N_ch_4_pad: int,
    N_ch_8: int,
    sub_4: int,
    out: torch.Tensor,         # [H_q, T_alloc] FP32
) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_q, triton.cdiv(T_eff, BLOCK_T))
    _k_mixed48_qk_dot_kernel[grid](
        Q, K_4bit, K_8bit, K_scale, K_zp, out,
        T_eff,
        N_ch_4_pad, N_ch_8, sub_4,
        Q.stride(0), Q.stride(1),
        K_4bit.stride(0), K_4bit.stride(1), K_4bit.stride(2),
        K_8bit.stride(0), K_8bit.stride(1), K_8bit.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_D4_SUB_C=64,
        BLOCK_D8_C=16,              # small: 4K pattern N_ch_8=4, dispatch enforces ≤16
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        num_stages=4,
    )


def k_uniform4_qk_dot_into(
    Q: torch.Tensor,        # [H_q, D=128]  FP32 (already permuted)
    K_4bit: torch.Tensor,   # [H_kv, T_alloc, 64]  uint8
    K_scale: torch.Tensor,  # [H_kv, 1, 128]  FP16
    K_zp: torch.Tensor,     # [H_kv, 1, 128]  FP16
    T_eff: int,
    out: torch.Tensor,      # [H_q, T_alloc]  FP32
) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_q, triton.cdiv(T_eff, BLOCK_T))
    _k_uniform4_qk_dot_kernel[grid](
        Q, K_4bit, K_scale, K_zp, out,
        T_eff,
        Q.stride(0), Q.stride(1),
        K_4bit.stride(0), K_4bit.stride(1), K_4bit.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        D_PACKED_C=64,
        D_C=D,
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        num_stages=4,
    )


def k_uniform4_qk_dot_tc_into(
    Q: torch.Tensor,
    K_4bit: torch.Tensor,
    K_scale: torch.Tensor,
    K_zp: torch.Tensor,
    T_eff: int,
    out: torch.Tensor,
) -> None:
    """E2 uniform4 TC wrapper. Same signature as k_uniform4_qk_dot_into."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    H_kv = K_scale.shape[1] if K_scale.dim() == 4 else K_scale.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_kv, triton.cdiv(T_eff, BLOCK_T))
    _k_uniform4_tc_kernel[grid](
        Q, K_4bit, K_scale, K_zp, out,
        T_eff,
        Q.stride(0), Q.stride(1),
        K_4bit.stride(0), K_4bit.stride(1), K_4bit.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        D_C=D, D_PACKED_C=D // 2,
        M_TC_C=16,
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        num_warps=4,
        num_stages=3,
    )


def k_mixed_unified_tc_into(
    Q: torch.Tensor,               # [H_q, D_padded] FP32 permuted
    K_2: torch.Tensor,             # 3D [H_kv, T_alloc, sub_2] uint8 (maybe dummy)
    K_4: torch.Tensor,
    K_8: torch.Tensor,
    K_scale: torch.Tensor,         # [H_kv, 1, D_padded] FP16
    K_zp: torch.Tensor,
    T_eff: int,
    N_ch_2_pad: int,
    N_ch_4_pad: int,
    N_ch_8: int,
    sub_2: int,
    sub_4: int,
    out: torch.Tensor,             # [H_q, T_alloc] FP32
    *,
    has_2: bool,
    has_4: bool,
    has_8: bool,
    block_k_8: int = 128,
) -> None:
    """E2 unified mixed-bit TC wrapper. `block_k_8` is 16 for hybrid
    mixed48/mixed248 (small N_ch_8 tail) or 128 for the generic unified path.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")
    H_q = Q.shape[0]
    H_kv = K_scale.shape[1] if K_scale.dim() == 4 else K_scale.shape[0]
    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_kv, triton.cdiv(T_eff, BLOCK_T))
    _k_mixed_unified_tc_kernel[grid](
        Q, K_2, K_4, K_8, K_scale, K_zp, out,
        T_eff,
        N_ch_2_pad, N_ch_4_pad, N_ch_8,
        sub_2, sub_4,
        Q.stride(0), Q.stride(1),
        K_2.stride(0), K_2.stride(1), K_2.stride(2),
        K_4.stride(0), K_4.stride(1), K_4.stride(2),
        K_8.stride(0), K_8.stride(1), K_8.stride(2),
        K_scale.stride(0), K_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_K_2=128,
        BLOCK_K_4=128,
        BLOCK_K_8=block_k_8,
        M_TC_C=16,
        GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
        HAS_2=has_2,
        HAS_4=has_4,
        HAS_8=has_8,
        num_warps=4,
        num_stages=3,
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _seg_padded(n: int, align: int) -> int:
    if n == 0:
        return 0
    return ((n + align - 1) // align) * align


def _as_3d(t):
    """Normalise K-side tensor to 3D [H_kv, T_alloc_or_1, last_dim].

    Accepts either [1, H_kv, ..., last] (build_packed_layer output) or
    [H_kv, ..., last] (_pre_squeeze_packed output).
    """
    if t is None:
        return None
    if t.dim() == 4 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def _ensure_tensor(t, H_kv: int, last_dim: int, device, dtype=torch.uint8):
    """If *t* is None or empty, return a placeholder; else pass-through as 3D."""
    t3 = _as_3d(t)
    if t3 is not None and t3.numel() > 0:
        return t3
    return torch.zeros((H_kv, 1, max(1, last_dim)), dtype=dtype, device=device)


def k_mixed_qk_dot_into(
    Q: torch.Tensor,
    packed_layer,
    T_eff: int,
    out: torch.Tensor,
    T_alloc: int | None = None,
) -> None:
    """Compute QK dot product with per-channel mixed-precision K cache,
    writing into a pre-allocated output buffer.

    Args:
        Q:      [H_q, D] FP16 — single decode query (batch dim squeezed).
        packed_layer: PackedKVLayer with K_2bit/K_4bit/K_8bit + K_ch_*.
        T_eff:  int — number of valid tokens.
        out:    [H_q, T_alloc] FP32 — pre-allocated buffer. Only the first
                T_eff columns are written (kernel t_mask guards the rest).
        T_alloc: optional explicit allocation length for K tensors' token dim.
                 If None, inferred from packed tensor shapes.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available.")

    H_q = Q.shape[0]
    assert Q.shape[1] == D, f"Q head_dim {Q.shape[1]} != {D}"

    N_ch_2, N_ch_4, N_ch_8 = packed_layer.K_ch_seg_bounds
    N_ch_2_pad = _seg_padded(N_ch_2, 4)
    N_ch_4_pad = _seg_padded(N_ch_4, 2)
    D_padded = N_ch_2_pad + N_ch_4_pad + N_ch_8

    sub_2 = N_ch_2_pad // 4
    sub_4 = N_ch_4_pad // 2

    if _HAS_X_DIAG_ENABLED:
        layer_key = id(packed_layer)
        if layer_key not in _HAS_X_PRINTED:
            layer_idx = len(_HAS_X_PRINTED)
            _HAS_X_PRINTED[layer_key] = layer_idx
            print(
                f"[K_HAS_X] layer={layer_idx:02d} "
                f"N_ch_2={int(N_ch_2)} N_ch_4={int(N_ch_4)} N_ch_8={int(N_ch_8)}  "
                f"HAS_2={N_ch_2_pad > 0} HAS_4={N_ch_4_pad > 0} HAS_8={N_ch_8 > 0}",
                flush=True,
            )

    device = Q.device
    # K_ch_scale is [1, H_kv, 1, D] (pre-squeeze) or [H_kv, 1, D] (post-squeeze)
    _sc = packed_layer.K_ch_scale
    H_kv = _sc.shape[1] if _sc.dim() == 4 else _sc.shape[0]

    # Build Q_perm_padded via a single index_select using the precomputed
    # padded permutation. Pad slots point at channel 0; matching K_ch_scale/zp
    # entries are 0 so those positions contribute nothing to the dot.
    q_fp32 = Q.float()
    if D_padded == 0:
        # Degenerate: no kept channels. Write zeros and return.
        out[:H_q, :T_eff].zero_()
        return

    if _DIAG_ENABLED:
        torch.cuda.synchronize()
        _t_is0 = time.perf_counter()
    # Phase 2 per-head Q gather: when the packed layer carries
    # ``K_ch_perm_padded_idx_for_Q [H_q, D_padded]`` (populated by
    # ``build_packed_layer`` when k_bits is [H_kv, D]), use torch.gather along
    # dim=1 so each query head reads its own channel permutation. Otherwise
    # fall back to the shared-permutation index_select (single 1D perm).
    perm_for_Q = getattr(packed_layer, "K_ch_perm_padded_idx_for_Q", None)
    if perm_for_Q is not None:
        # q_fp32 shape: [H_q, D]; perm_for_Q shape: [H_q, D_padded].
        Q_perm_padded = torch.gather(q_fp32, dim=1, index=perm_for_Q)
    else:
        Q_perm_padded = q_fp32.index_select(
            dim=1, index=packed_layer.K_ch_perm_padded_idx,
        )  # [H_q, D_padded] fp32
    if _DIAG_ENABLED:
        torch.cuda.synchronize()
        _DIAG["index_select_ms"] += (time.perf_counter() - _t_is0) * 1000.0

    # Resolve K segment tensors as 3D [H_kv, T_alloc, last_dim]
    K_2 = _ensure_tensor(packed_layer.K_2bit, H_kv, max(1, sub_2), device)
    K_4 = _ensure_tensor(packed_layer.K_4bit, H_kv, max(1, sub_4), device)
    K_8 = _ensure_tensor(packed_layer.K_8bit, H_kv, max(1, N_ch_8), device)

    # Scale/zp as 3D [H_kv, 1, D_padded]
    K_scale = _as_3d(packed_layer.K_ch_scale)
    K_zp = _as_3d(packed_layer.K_ch_zp)

    BLOCK_T = BLOCK_T_DEFAULT
    grid = (H_q, triton.cdiv(T_eff, BLOCK_T))

    # -----------------------------------------------------------------
    # Dispatch to a specialised kernel when the HAS_* pattern matches a
    # known-fast code path. Known-fast kernels (in decreasing speedup):
    #   * uniform4: single 4-bit segment covering full D (N_ch_4_pad==D).
    #               ~2× faster than unified (validated against old kernel).
    #   * mixed24 : HAS_2 + HAS_4.  ~1.1× faster than unified (structural).
    #   * mixed48 : HAS_4 + HAS_8.  ~1.1× faster than unified (structural).
    # Anything else — single 2/8-bit, 3-segment, 1+1 combos other than
    # (2,4)/(4,8), or a 4-bit segment with evicted channels — falls back
    # to the generic unified kernel.
    # -----------------------------------------------------------------
    has_2 = N_ch_2_pad > 0
    has_4 = N_ch_4_pad > 0
    has_8 = N_ch_8 > 0

    if _DIAG_ENABLED:
        torch.cuda.synchronize()
        _t_kn0 = time.perf_counter()

    tc_ok = _K_TC and T_eff >= _K_TC_MIN_T_EFF

    if not has_2 and not has_8 and N_ch_4_pad == D:
        # uniform4 fast path (single 4-bit segment, full head_dim).
        # k_uniform4_qk_dot_into has hardcoded D=128 / D_PACKED_C=64, so we
        # only take this path when N_ch_4_pad == D (== 128). If some channels
        # were evicted (N_ch_4_pad < D), fall back to unified below.
        if tc_ok:
            k_uniform4_qk_dot_tc_into(
                Q_perm_padded, K_4, K_scale, K_zp, T_eff, out,
            )
        else:
            k_uniform4_qk_dot_into(
                Q_perm_padded, K_4, K_scale, K_zp, T_eff, out,
            )
    elif has_2 and has_4 and not has_8:
        # mixed24 fast path
        if tc_ok:
            k_mixed_unified_tc_into(
                Q_perm_padded, K_2, K_4, K_8, K_scale, K_zp, T_eff,
                N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4, out,
                has_2=True, has_4=True, has_8=False, block_k_8=16,
            )
        else:
            k_mixed24_qk_dot_into(
                Q_perm_padded, K_2, K_4, K_scale, K_zp,
                T_eff, N_ch_2_pad, sub_2, sub_4, out,
            )
    elif has_4 and has_8 and not has_2 and N_ch_8 <= 16:
        # mixed48 fast path (BLOCK_D8_C=16 in specialised kernel)
        if tc_ok:
            k_mixed_unified_tc_into(
                Q_perm_padded, K_2, K_4, K_8, K_scale, K_zp, T_eff,
                N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4, out,
                has_2=False, has_4=True, has_8=True, block_k_8=16,
            )
        else:
            k_mixed48_qk_dot_into(
                Q_perm_padded, K_4, K_8, K_scale, K_zp,
                T_eff, N_ch_4_pad, N_ch_8, sub_4, out,
            )
    elif has_2 and has_4 and has_8 and N_ch_8 <= 16:
        # mixed248 fast path: 3-segment with small N_ch_8. Observed at 128K
        # under --streaming where packer emits small 8-bit tails on most layers.
        if tc_ok:
            k_mixed_unified_tc_into(
                Q_perm_padded, K_2, K_4, K_8, K_scale, K_zp, T_eff,
                N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4, out,
                has_2=True, has_4=True, has_8=True, block_k_8=16,
            )
        else:
            k_mixed248_qk_dot_into(
                Q_perm_padded, K_2, K_4, K_8, K_scale, K_zp,
                T_eff, N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4, out,
            )
    else:
        # Fallback: generic unified kernel (supports all 7 non-empty patterns,
        # incl. single-segment and 3-segment edge cases).
        if tc_ok:
            k_mixed_unified_tc_into(
                Q_perm_padded, K_2, K_4, K_8, K_scale, K_zp, T_eff,
                N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4, out,
                has_2=has_2, has_4=has_4, has_8=has_8, block_k_8=128,
            )
        else:
            _k_mixed_qk_dot_unified_kernel[grid](
                Q_perm_padded, K_2, K_4, K_8,
                K_scale, K_zp, out,
                T_eff,
                N_ch_2_pad, N_ch_4_pad, N_ch_8, sub_2, sub_4,
                Q_perm_padded.stride(0), Q_perm_padded.stride(1),
                K_2.stride(0), K_2.stride(1), K_2.stride(2),
                K_4.stride(0), K_4.stride(1), K_4.stride(2),
                K_8.stride(0), K_8.stride(1), K_8.stride(2),
                K_scale.stride(0), K_scale.stride(2),
                out.stride(0), out.stride(1),
                BLOCK_T=BLOCK_T,
                BLOCK_D2_SUB_C=32,
                BLOCK_D4_SUB_C=64,
                BLOCK_D8_C=128,
                GQA_FACTOR_C=_runtime_gqa(Q, K_scale),
                HAS_2=has_2,
                HAS_4=has_4,
                HAS_8=has_8,
                num_stages=4,
            )

    if _DIAG_ENABLED:
        torch.cuda.synchronize()
        _DIAG["kernel_ms"] += (time.perf_counter() - _t_kn0) * 1000.0
        _DIAG["n_calls"] += 1


def k_mixed_qk_dot(
    Q: torch.Tensor,
    packed_layer,
    T_eff: int,
) -> torch.Tensor:
    """Allocating variant of k_mixed_qk_dot_into.

    Returns:
        QK: [H_q, T_eff] FP32.
    """
    H_q = Q.shape[0]
    QK = torch.empty((H_q, T_eff), dtype=torch.float32, device=Q.device)
    k_mixed_qk_dot_into(Q, packed_layer, T_eff, QK)
    return QK


# ---------------------------------------------------------------------------
# Pure PyTorch reference (for correctness testing, Gate 2)
# ---------------------------------------------------------------------------
def k_mixed_qk_dot_ref(
    Q: torch.Tensor,
    packed_layer,
    T_eff: int,
) -> torch.Tensor:
    """Reference: unpack K_mixed → dequantise → standard QK bmm.

    Mirrors the Triton kernel numerics (FP32 accumulation, 1/sqrt(D) scale).
    """
    from obkv_accel.packing import unpack_k_mixed

    K_hat = unpack_k_mixed(packed_layer, D=D)  # [1, H_kv, T_eff, D] FP16
    K_hat = K_hat.squeeze(0)                    # [H_kv, T_eff, D]
    # GQA: broadcast H_kv → H_q
    H_q = Q.shape[0]
    H_kv = K_hat.shape[0]
    assert H_q % H_kv == 0
    K_exp = K_hat.repeat_interleave(H_q // H_kv, dim=0)  # [H_q, T_eff, D]

    q_f = Q.float()
    k_f = K_exp.float()
    qk = torch.einsum("hd,htd->ht", q_f, k_f[:, :T_eff, :])
    qk = qk / SQRT_D
    return qk
