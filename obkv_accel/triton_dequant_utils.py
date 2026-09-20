"""Shared Triton dequant-to-FP16 helpers for K/V tensor-core kernels.

All helpers are `@triton.jit` inner routines meant to be called from the V/K
TC kernels (E2). They produce FP16 tiles that feed directly into `tl.dot` on
`mma.m16n8k16` (A100 SM 80 compatible).

Two tile conventions are exported:

V-side (per-token scale/zp; full dequant applied):
  * `dequant_2bit_v_tile(...)`  -> tile[BLOCK_T, SUB_D] FP16
  * `dequant_4bit_v_tile(...)`  -> tile[BLOCK_T, SUB_D] FP16
  * `dequant_8bit_v_tile(...)`  -> tile[BLOCK_T, SUB_D] FP16

K-side (per-channel scale/zp; bias trick -> codes stay as raw uints cast to
FP16 for the `tl.dot`, scale/zp folded into Q by caller):
  * `unpack_2bit_k_codes_kt(...)`  -> tile[K_ch, BLOCK_T] FP16 (values in 0..3)
  * `unpack_4bit_k_codes_kt(...)`  -> tile[K_ch, BLOCK_T] FP16 (0..15)
  * `unpack_8bit_k_codes_kt(...)`  -> tile[K_ch, BLOCK_T] FP16 (0..255)

Layout reminders (see `triton_v_kernel.py` + `triton_k_kernel.py` headers):

  2-bit V (QUARTER-SPLIT over D): packed[i] holds channels i, i+32, i+64, i+96
    (shifts 0, 2, 4, 6) — so for D_QUARTER output slot selected by
    `SHIFT_2BIT_C` the helper reads packed[..., d_sub] and shifts.

  4-bit V (HALF-SPLIT over D): packed[i] holds channels i, i+64
    (shifts 0, 4) — so `NIBBLE_SHIFT_C` selects low (0) vs high (4) nibble,
    and `BYTE_OFFSET_4BIT_C` offsets within the 64 packed bytes to pick the
    correct D_QUARTER slice.

  2-bit K (QUARTER-SPLIT over D_channel axis): packed[:, :, d] holds 4 channel
    positions per byte (shifts 0,2,4,6 = 4 sub-groups of sub_2 channels each).

  4-bit K (HALF-SPLIT over D_channel axis): packed[:, :, d] holds 2 channel
    positions per byte (shifts 0, 4 = 2 sub-groups of sub_4 each).

All helpers treat masks:
  * `t_mask` (BLOCK_T-bool): out-of-range tokens load as 0 -> dequant 0.
  * For K codes the caller supplies a channel-axis mask via `d_mask`.

The helpers NEVER apply `1/sqrt(D)` scaling; callers fold that at the end.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    # ======================================================================
    # V-side helpers: per-token scale/zp, produce [BLOCK_T, SUB_D] FP16 tile
    # ======================================================================

    @triton.jit
    def dequant_2bit_v_tile(
        V_2bit_ptr,       # [H_kv, T_alloc, D//4] uint8
        scale_ptr,        # per-token scale base (already offset to [h_kv, 0])
        zp_ptr,           # per-token zp base
        h_kv,
        t_offs,           # [BLOCK_T] int
        t_mask,           # [BLOCK_T] bool
        t_row_offset,     # int -- row offset in scale/zp (per-segment origin)
        stride_v_h: tl.constexpr,
        stride_v_t: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_vs_t,
        SHIFT_2BIT_C: tl.constexpr,    # pid_dq * 2  (0/2/4/6)
        SUB_D: tl.constexpr,           # = D_QUARTER_C = 32
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_T, SUB_D] FP16 dequantised 2-bit V tile."""
        d_sub = tl.arange(0, SUB_D)
        vp_ptrs = (V_2bit_ptr + h_kv * stride_v_h
                   + t_offs[:, None] * stride_v_t
                   + d_sub[None, :] * stride_v_d)
        packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
        v_q = ((packed >> SHIFT_2BIT_C) & 0x3).to(tl.float32)
        scale = tl.load(scale_ptr + (t_offs + t_row_offset) * stride_vs_t,
                        mask=t_mask, other=0.0).to(tl.float32)
        zp = tl.load(zp_ptr + (t_offs + t_row_offset) * stride_vs_t,
                     mask=t_mask, other=0.0).to(tl.float32)
        # Mask-safe: for out-of-range tokens, scale=0 -> v_dq=0.
        v_dq = (v_q - zp[:, None]) * scale[:, None]
        return v_dq.to(tl.float16)

    @triton.jit
    def dequant_4bit_v_tile(
        V_4bit_ptr,       # [H_kv, T_alloc, D//2] uint8
        scale_ptr,
        zp_ptr,
        h_kv,
        t_offs,
        t_mask,
        t_row_offset,
        stride_v_h: tl.constexpr,
        stride_v_t: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_vs_t,
        NIBBLE_SHIFT_C: tl.constexpr,   # 0 or 4
        BYTE_OFFSET_4BIT_C: tl.constexpr,   # 0 or SUB_D
        SUB_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_T, SUB_D] FP16 dequantised 4-bit V tile."""
        d_sub = tl.arange(0, SUB_D)
        vp_ptrs = (V_4bit_ptr + h_kv * stride_v_h
                   + t_offs[:, None] * stride_v_t
                   + (d_sub + BYTE_OFFSET_4BIT_C)[None, :] * stride_v_d)
        packed = tl.load(vp_ptrs, mask=t_mask[:, None], other=0).to(tl.int32)
        v_q = ((packed >> NIBBLE_SHIFT_C) & 0xF).to(tl.float32)
        scale = tl.load(scale_ptr + (t_offs + t_row_offset) * stride_vs_t,
                        mask=t_mask, other=0.0).to(tl.float32)
        zp = tl.load(zp_ptr + (t_offs + t_row_offset) * stride_vs_t,
                     mask=t_mask, other=0.0).to(tl.float32)
        v_dq = (v_q - zp[:, None]) * scale[:, None]
        return v_dq.to(tl.float16)

    @triton.jit
    def dequant_8bit_v_tile(
        V_8bit_ptr,       # [H_kv, T_alloc, D] uint8
        scale_ptr,
        zp_ptr,
        h_kv,
        t_offs,
        t_mask,
        t_row_offset,
        stride_v_h: tl.constexpr,
        stride_v_t: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_vs_t,
        D_OFFSET_C: tl.constexpr,      # pid_dq * SUB_D
        SUB_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_T, SUB_D] FP16 dequantised 8-bit V tile."""
        d_sub = tl.arange(0, SUB_D)
        v_ptrs = (V_8bit_ptr + h_kv * stride_v_h
                  + t_offs[:, None] * stride_v_t
                  + (d_sub + D_OFFSET_C)[None, :] * stride_v_d)
        v = tl.load(v_ptrs, mask=t_mask[:, None], other=0).to(tl.float32)
        scale = tl.load(scale_ptr + (t_offs + t_row_offset) * stride_vs_t,
                        mask=t_mask, other=0.0).to(tl.float32)
        zp = tl.load(zp_ptr + (t_offs + t_row_offset) * stride_vs_t,
                     mask=t_mask, other=0.0).to(tl.float32)
        v_dq = (v - zp[:, None]) * scale[:, None]
        return v_dq.to(tl.float16)

    # ======================================================================
    # K-side helpers: raw codes cast to FP16 (bias trick fuses scale/zp into Q)
    # Produce [K_ch, BLOCK_T] FP16 tile suitable for tl.dot(qs, k_codes)
    # ======================================================================

    @triton.jit
    def unpack_2bit_k_codes_kt(
        K_2bit_ptr,       # [H_kv, T_alloc, sub_2] uint8
        h_kv,
        t_offs,           # [BLOCK_T]
        t_mask,           # [BLOCK_T]
        stride_k_h: tl.constexpr,
        stride_k_t: tl.constexpr,
        stride_k_d: tl.constexpr,
        SHIFT_2BIT_C: tl.constexpr,    # 0/2/4/6
        BLOCK_D_SUB_C: tl.constexpr,   # max sub_2 (compile-time upper bound)
        sub_2,                          # runtime true sub_2
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_D_SUB_C, BLOCK_T] FP16 containing raw 2-bit codes
        (0..3). Tokens beyond `t_mask` and channels beyond `sub_2` are 0.
        """
        d_sub = tl.arange(0, BLOCK_D_SUB_C)
        d_mask = d_sub < sub_2
        k2_ptrs = (K_2bit_ptr + h_kv * stride_k_h
                   + t_offs[None, :] * stride_k_t
                   + d_sub[:, None] * stride_k_d)
        packed = tl.load(k2_ptrs,
                         mask=(d_mask[:, None] & t_mask[None, :]),
                         other=0).to(tl.int32)
        codes = ((packed >> SHIFT_2BIT_C) & 0x3).to(tl.float32)
        return codes.to(tl.float16)

    @triton.jit
    def unpack_4bit_k_codes_kt(
        K_4bit_ptr,       # [H_kv, T_alloc, sub_4] uint8
        h_kv,
        t_offs,
        t_mask,
        stride_k_h: tl.constexpr,
        stride_k_t: tl.constexpr,
        stride_k_d: tl.constexpr,
        NIBBLE_SHIFT_C: tl.constexpr,   # 0 or 4
        BLOCK_D_SUB_C: tl.constexpr,    # max sub_4
        sub_4,
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_D_SUB_C, BLOCK_T] FP16 containing raw 4-bit codes (0..15)."""
        d_sub = tl.arange(0, BLOCK_D_SUB_C)
        d_mask = d_sub < sub_4
        k4_ptrs = (K_4bit_ptr + h_kv * stride_k_h
                   + t_offs[None, :] * stride_k_t
                   + d_sub[:, None] * stride_k_d)
        packed = tl.load(k4_ptrs,
                         mask=(d_mask[:, None] & t_mask[None, :]),
                         other=0).to(tl.int32)
        codes = ((packed >> NIBBLE_SHIFT_C) & 0xF).to(tl.float32)
        return codes.to(tl.float16)

    @triton.jit
    def unpack_8bit_k_codes_kt(
        K_8bit_ptr,       # [H_kv, T_alloc, N_ch_8] uint8
        h_kv,
        t_offs,
        t_mask,
        stride_k_h: tl.constexpr,
        stride_k_t: tl.constexpr,
        stride_k_d: tl.constexpr,
        BLOCK_D_C: tl.constexpr,       # max N_ch_8
        N_ch_8,
        BLOCK_T: tl.constexpr,
    ):
        """Return [BLOCK_D_C, BLOCK_T] FP16 containing raw 8-bit codes (0..255)."""
        d = tl.arange(0, BLOCK_D_C)
        d_mask = d < N_ch_8
        k8_ptrs = (K_8bit_ptr + h_kv * stride_k_h
                   + t_offs[None, :] * stride_k_t
                   + d[:, None] * stride_k_d)
        codes = tl.load(k8_ptrs,
                        mask=(d_mask[:, None] & t_mask[None, :]),
                        other=0).to(tl.float32)
        return codes.to(tl.float16)
