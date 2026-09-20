"""
obkv_accel/packing.py

S1: Real Packing -- quantise FP16 KV cache into compressed uint8 format
with mixed-precision V segments (2/4/8 bit, per-token) AND
mixed-precision K segments (2/4/8 bit, per-channel).

Layout conventions follow the existing RDKV compress utilities:
  - 4-bit: half-split  byte[i] = val[i] | (val[i+D//2] << 4)
  - 2-bit: quarter-split byte[i] = val[i] | (val[i+D//4]<<2)
                                    | (val[i+D//2]<<4) | (val[i+3D//4]<<6)

K per-channel mixed precision:
  - Channels with k_bit == 0 are pruned.
  - Channels with k_bit == 16 are clamped to 8 (8-bit K error is negligible).
  - Remaining channels are sorted by bit-width (2 → 4 → 8) and packed per-segment.
  - Each segment may require channel padding (2-bit: pad to multiple of 4,
    4-bit: pad to multiple of 2). Padded channels get scale=zp=0 so their
    dequant contribution is exactly zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from obkv_accel.bitpack import (
    transfer_8bit_to_4bit_batchwise,
    transfer_4bit_to_8bit_batchwise,
    transfer_8bit_to_2bit_batchwise,
    transfer_2bit_to_8bit_batchwise,
)


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class PackedKVLayer:
    # K side (per-channel mixed precision, single path).
    # Shapes use the **shared-max layout** across H_kv heads: every per-head
    # segment is zero-padded to `max_h(N_ch_*_pad)`. This keeps the Phase-2 K
    # Triton kernel (shared scalar bounds) correct — pad lanes have
    # scale = zp = 0 → zero dequant contribution, and Q is gathered through
    # ``K_ch_perm_padded_idx_for_Q`` which has sentinel 0 at pad positions so
    # ``Q[:, sentinel]·K_scale[sentinel]=0``.
    K_2bit: Optional[Tensor]   # [1, H_kv, T_eff, max_h(N_ch_2_pad) // 4] uint8 or None
    K_4bit: Optional[Tensor]   # [1, H_kv, T_eff, max_h(N_ch_4_pad) // 2] uint8 or None
    K_8bit: Optional[Tensor]   # [1, H_kv, T_eff, max_h(N_ch_8)] uint8 or None
    K_ch_scale: Tensor         # [1, H_kv, 1, D_kept_padded_max] FP16 (per-channel, sorted order)
    K_ch_zp: Tensor            # [1, H_kv, 1, D_kept_padded_max] FP16 (per-channel, sorted order)
    K_ch_sort_idx: Tensor      # [D_kept_max] long — head-0 slice for legacy consumers (see note)
    K_ch_seg_bounds: Tuple[int, int, int]  # shared-max (max_N_ch_2, max_N_ch_4, max_N_ch_8)
    # Precomputed permutation for Q at decode: one index_select builds the
    # padded Q layout that matches K_ch_scale/K_ch_zp. Pad slots point at
    # channel 0; the corresponding scale/zp are 0 so contribution is 0.
    K_ch_perm_padded_idx: Tensor  # [D_padded_max] long into original D (head-0 fallback)
    # V side (unchanged)
    V_2bit: Optional[Tensor]   # [1, H_kv, N_2, D//4] uint8 or None
    V_4bit: Optional[Tensor]   # [1, H_kv, N_4, D//2] uint8 or None
    V_8bit: Optional[Tensor]   # [1, H_kv, N_8, D] uint8 or None
    V_scale: Tensor            # [1, H_kv, T_eff, 1] FP16 (per-token, sorted order)
    V_zp: Tensor               # [1, H_kv, T_eff, 1] FP16 (per-token, sorted order)
    sort_idx: Tensor           # [T_eff] long — V token sort
    seg_bounds: Tuple[int, int, int]  # V (N_2, N_4, N_8)
    T_eff: int
    # ── Per-head joint-knapsack path metadata (None for global/legacy path) ──
    # Set only by _pad_and_stack_per_head_layers. When present, V kernel dispatch
    # uses a per-head variant that reads N_2/N_4/N_8 from seg_bounds_per_head at
    # runtime. `T_eff` above is set to max_h(T_eff_h) so old-zone buffers allocate
    # to the per-head maximum; softmax_mask zeros out the padded tail per head.
    seg_bounds_per_head: Optional[Tensor] = None   # [H_kv, 3] int32
    T_eff_per_head: Optional[Tensor] = None        # [H_kv] int32
    softmax_mask: Optional[Tensor] = None          # [H_q, max_T_eff] FP32 additive
    n_fp16_per_head: Optional[Tensor] = None       # [H_kv] int32
    fp16_gap_mask: Optional[Tensor] = None         # [H_kv, max_n_fp16] FP32 additive
    # ── Per-head K bit allocation metadata (Phase 2) ──
    # When populated, the K Triton kernel wrapper uses the per-head Q gather
    # path (``K_ch_perm_padded_idx_for_Q``), so different heads can have
    # different K_ch_sort_idx orderings. None = fall back to legacy single-
    # permutation layout via the shared ``K_ch_perm_padded_idx`` (head-0).
    K_ch_sort_idx_per_head: Optional[Tensor] = None     # [H_kv, D_kept_max] long
    K_ch_perm_padded_idx_per_head: Optional[Tensor] = None  # [H_kv, D_padded_max] long
    K_ch_seg_bounds_per_head: Optional[Tensor] = None   # [H_kv, 3] int32 UNPADDED
    # Decode-time Q permutation: [H_q, D_padded_max] long, derived from
    # K_ch_perm_padded_idx_per_head by repeat-interleave for GQA (each h_q
    # maps to K_ch_perm_padded_idx_per_head[h_q // gqa_factor]).
    K_ch_perm_padded_idx_for_Q: Optional[Tensor] = None  # [H_q, D_padded_max] long
    # ── TriZone (方案 1) K T-dim layout ──
    # K T-dim layout is [0, T_v) compressed tokens ++ [T_v, T_eff_k) v=16 tokens.
    # ``T_eff`` (above) stays as the compressed-token count for V-kernel
    # compatibility — the V kernel only consumes ``w[:, :T_v]``. ``T_eff_k =
    # T_v + n_v16`` is the K-side sequence length that the K kernel / QK dot
    # must consume. ``T_eff_k == 0`` signals legacy (pre-TriZone) layers and
    # callers should treat them as ``T_eff_k := T_eff``.
    T_eff_k: int = 0
    n_v16: int = 0
    T_eff_k_per_head: Optional[Tensor] = None      # [H_kv] int32
    n_v16_per_head: Optional[Tensor] = None        # [H_kv] int32
    # Logical allocator decisions.  These are retained for blockwise decode
    # accounting/debugging; the attention kernels do not consume them.
    allocation_v_bits: Optional[Tensor] = None     # [H_kv, T_original] int16
    allocation_k_bits: Optional[Tensor] = None     # [H_kv, D] int16
    # Transient source used while assembling a per-head layer. The final
    # stacked PackedKVLayer never retains this tensor.
    deferred_k_for_stack: Optional[Tensor] = None


@dataclass
class TriZoneCache:
    packed: Tuple[PackedKVLayer, ...]         # N layers, frozen
    new_v_only: List[Optional[Tensor]]        # [L] [1, H_kv, n_v16, D] FP16 — prefill read-only
    new_both_k: List[Optional[Tensor]]        # [L] [1, H_kv, n_gen, D] FP16 — decode-only
    new_both_v: List[Optional[Tensor]]
    original_seq_len: int

    # Legacy ``.new_k`` / ``.new_v`` alias for read-only consumers. Returns
    # the list reference so ``cache.new_k[i] = X`` still mutates through.
    @property
    def new_k(self):
        return self.new_both_k

    @property
    def new_v(self):
        return self.new_both_v


# Existing ``from obkv_accel.packing import DualZoneCache`` imports keep
# working unchanged. The dataclass *is* TriZoneCache.
DualZoneCache = TriZoneCache


# ── Core helpers ─────────────────────────────────────────────────────


def quantize_and_pack(
    x: Tensor,
    n_bits: int,
    dim: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Uniform asymmetric quantisation along *dim*.

    Returns:
        x_q:        uint8 tensor of quantised codes, same shape as *x*
        scale:      FP16 scale tensor  (keepdim along *dim*)
        zero_point: FP16 zero-point tensor (keepdim along *dim*)
    """
    x_f32 = x.float()
    qmin, qmax = 0, 2 ** n_bits - 1

    x_min = x_f32.amin(dim=dim, keepdim=True)
    x_max = x_f32.amax(dim=dim, keepdim=True)

    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-8)
    zero_point = (qmin - x_min / scale).clamp(qmin, qmax).round()

    x_q = (x_f32 / scale + zero_point).clamp(qmin, qmax).round()
    x_q = x_q.to(torch.uint8)

    return x_q, scale.half(), zero_point.half()


def _seg_padded_len(n: int, align: int) -> int:
    if n == 0:
        return 0
    return ((n + align - 1) // align) * align


# ── K packing (mixed-precision per-channel) ─────────────────────────


def pack_k_mixed(
    K: Tensor,
    k_bits: Tensor,
    *,
    sub_2_target: Optional[int] = None,
    sub_4_target: Optional[int] = None,
    n_ch_8_target: Optional[int] = None,
    segment_counts: Optional[Tuple[int, int, int]] = None,
) -> Tuple[
    Optional[Tensor],  # K_2bit
    Optional[Tensor],  # K_4bit
    Optional[Tensor],  # K_8bit
    Tensor,            # K_ch_scale
    Tensor,            # K_ch_zp
    Tensor,            # ch_sort_idx
    Tensor,            # ch_perm_padded_idx
    Tuple[int, int, int],  # (N_ch_2, N_ch_4, N_ch_8) UNPADDED
]:
    """Pack K with per-channel mixed bit-width assignment.

    Internal helper. External callers in the Phase-2+ per-head pipeline
    should use ``pack_k_mixed_per_head`` instead; this function is invoked
    per head from that wrapper (with explicit ``sub_*_target`` args so each
    head's quarter-split/half-split bytes align to a shared-max channel
    layout across H_kv).

    Args:
        K:      [1, H_kv, T_eff, D] FP16  (typically H_kv=1 in the per-head loop)
        k_bits: [D] LongTensor with values in {0, 2, 4, 8, 16}.
                0 = channel pruned; 16 is clamped to 8 internally.
        sub_2_target, sub_4_target, n_ch_8_target:
            Optional shared-max targets passed by ``pack_k_mixed_per_head``.
            When supplied:
              - ``sub_2_target``: target 2-bit sub-dim (N_ch_2_pad // 4) to align
                to. ``N_ch_2_pad`` becomes ``4 * sub_2_target``.
              - ``sub_4_target``: target 4-bit sub-dim (N_ch_4_pad // 2).
                ``N_ch_4_pad`` becomes ``2 * sub_4_target``.
              - ``n_ch_8_target``: target 8-bit channel count.
            These pad the **sorted channel tensor in channel space** (prior to
            ``quantize_and_pack``), so quarter-/half-split byte packing places
            real channels at their original positions and zero channels occupy
            the pad slots. Pad channels have scale = zp = 0 → dequant = 0.

    Returns:
        K_2bit:     [1, H_kv, T_eff, N_ch_2_pad // 4] uint8 or None
        K_4bit:     [1, H_kv, T_eff, N_ch_4_pad // 2] uint8 or None
        K_8bit:     [1, H_kv, T_eff, N_ch_8_out]       uint8 or None
        K_ch_scale: [1, H_kv, 1, D_kept_padded] FP16 (padded channels = 0)
        K_ch_zp:    [1, H_kv, 1, D_kept_padded] FP16 (padded channels = 0)
        ch_sort_idx:[D_kept] long — indices into original D dim for kept channels
                    in sorted order (2-bit first, then 4, then 8). **UNPADDED**;
                    caller pads (with sentinel 0) to target widths if needed.
        ch_perm_padded_idx: [D_padded] long — Q-side permutation aligning to the
                    padded K layout. Pad slots point to channel 0 so the
                    corresponding scale/zp = 0 slots contribute zero.
        seg_bounds: (N_ch_2, N_ch_4, N_ch_8) UNPADDED per-segment channel counts.
    """
    assert K.dim() == 4 and K.shape[0] == 1, f"Expected K [1,H,T,D], got {tuple(K.shape)}"
    H_kv = K.shape[1]
    T_eff = K.shape[2]
    D = K.shape[3]
    device = K.device

    # 1) Clamp 16 → 8 (unify pipeline; 8-bit K error is negligible)
    k_bits_work = k_bits.to(device=device, dtype=torch.long).clamp(max=8)

    # 2) Channel pruning: drop k_bits == 0
    keep_mask = k_bits_work > 0
    keep_ids = keep_mask.nonzero(as_tuple=True)[0]  # [D_kept]
    k_bits_kept = k_bits_work[keep_ids]              # [D_kept]

    # 3) Stable sort kept channels by bit-width (2 < 4 < 8)
    order = torch.argsort(k_bits_kept, stable=True)
    ch_sort_idx = keep_ids[order].contiguous()       # [D_kept] into original D
    k_bits_sorted = k_bits_kept[order]

    if segment_counts is None:
        N_ch_2 = int((k_bits_sorted == 2).sum().item())
        N_ch_4 = int((k_bits_sorted == 4).sum().item())
        N_ch_8 = int((k_bits_sorted == 8).sum().item())
    else:
        N_ch_2, N_ch_4, N_ch_8 = map(int, segment_counts)
    D_kept = N_ch_2 + N_ch_4 + N_ch_8
    assert D_kept == ch_sort_idx.numel(), (
        f"Segment sizes {N_ch_2}+{N_ch_4}+{N_ch_8} != D_kept={ch_sort_idx.numel()}"
    )

    # 4) Reorder K along channel dim → [1, H_kv, T_eff, D_kept]
    K_sorted = K.index_select(dim=3, index=ch_sort_idx).contiguous()

    # 5) Resolve per-segment padded channel counts. If caller supplied
    #    sub_*_target / n_ch_8_target (per-head path aligning to shared max),
    #    honour those; otherwise default to per-segment alignment.
    N_ch_2_pad_auto = _seg_padded_len(N_ch_2, 4)
    N_ch_4_pad_auto = _seg_padded_len(N_ch_4, 2)
    if sub_2_target is not None:
        N_ch_2_pad = 4 * int(sub_2_target)
        assert N_ch_2_pad >= N_ch_2_pad_auto, (
            f"sub_2_target={sub_2_target} → N_ch_2_pad={N_ch_2_pad} < required "
            f"{N_ch_2_pad_auto} for N_ch_2={N_ch_2}"
        )
    else:
        N_ch_2_pad = N_ch_2_pad_auto
    if sub_4_target is not None:
        N_ch_4_pad = 2 * int(sub_4_target)
        assert N_ch_4_pad >= N_ch_4_pad_auto, (
            f"sub_4_target={sub_4_target} → N_ch_4_pad={N_ch_4_pad} < required "
            f"{N_ch_4_pad_auto} for N_ch_4={N_ch_4}"
        )
    else:
        N_ch_4_pad = N_ch_4_pad_auto
    if n_ch_8_target is not None:
        N_ch_8_out = int(n_ch_8_target)
        assert N_ch_8_out >= N_ch_8, (
            f"n_ch_8_target={n_ch_8_target} < N_ch_8={N_ch_8}"
        )
    else:
        N_ch_8_out = N_ch_8

    # 6) Per-segment quantise + bit-pack. Padding happens in channel space on
    #    the sorted K tensor (BEFORE quantize_and_pack), so quarter-/half-split
    #    byte packing places real channels at their original positions and
    #    zero channels occupy the pad slots. Scale = zp = 0 for pad channels
    #    falls out naturally from quantize_and_pack on a zero column.
    scale_parts: list[Tensor] = []
    zp_parts: list[Tensor] = []

    K_2bit: Optional[Tensor] = None
    if N_ch_2_pad > 0:
        if N_ch_2 > 0:
            seg_unpad = K_sorted[..., :N_ch_2]                  # [1,H,T_eff,N_ch_2]
        else:
            seg_unpad = K_sorted[..., :0]                       # empty slice
        pad = N_ch_2_pad - N_ch_2
        if pad > 0:
            seg = F.pad(seg_unpad, (0, pad), value=0.0)
        else:
            seg = seg_unpad
        codes, sc, zp = quantize_and_pack(seg, 2, dim=2)        # scale/zp: [1,H,1,N_ch_2_pad]
        K_2bit = transfer_8bit_to_2bit_batchwise(codes.contiguous())
        scale_parts.append(sc)
        zp_parts.append(zp)

    K_4bit: Optional[Tensor] = None
    if N_ch_4_pad > 0:
        if N_ch_4 > 0:
            seg_unpad = K_sorted[..., N_ch_2:N_ch_2 + N_ch_4]
        else:
            seg_unpad = K_sorted[..., :0]
        pad = N_ch_4_pad - N_ch_4
        if pad > 0:
            seg = F.pad(seg_unpad, (0, pad), value=0.0)
        else:
            seg = seg_unpad
        codes, sc, zp = quantize_and_pack(seg, 4, dim=2)
        K_4bit = transfer_8bit_to_4bit_batchwise(codes.contiguous())
        scale_parts.append(sc)
        zp_parts.append(zp)

    K_8bit: Optional[Tensor] = None
    if N_ch_8_out > 0:
        if N_ch_8 > 0:
            seg_unpad = K_sorted[..., N_ch_2 + N_ch_4:]
        else:
            seg_unpad = K_sorted[..., :0]
        pad = N_ch_8_out - N_ch_8
        if pad > 0:
            seg = F.pad(seg_unpad, (0, pad), value=0.0)
        else:
            seg = seg_unpad
        codes, sc, zp = quantize_and_pack(seg, 8, dim=2)
        # 8-bit has no packing; keep codes as-is.
        K_8bit = codes.contiguous()
        scale_parts.append(sc)
        zp_parts.append(zp)

    # 7) Concat scale/zp along channel dim → [1, H_kv, 1, D_kept_padded]
    if scale_parts:
        K_ch_scale = torch.cat(scale_parts, dim=3).contiguous()
        K_ch_zp = torch.cat(zp_parts, dim=3).contiguous()
    else:
        K_ch_scale = torch.empty(1, H_kv, 1, 0, dtype=torch.float16, device=device)
        K_ch_zp = torch.empty_like(K_ch_scale)

    # 9) Build padded permutation index: one index_select at decode time produces
    # Q_perm_padded matching the K_ch_scale/K_ch_zp layout. Pad slots use
    # channel 0 as a benign placeholder — scale/zp are zero there.
    D_padded = N_ch_2_pad + N_ch_4_pad + N_ch_8_out
    perm_parts: list[Tensor] = []
    sentinel = torch.zeros(1, dtype=torch.long, device=device)
    if N_ch_2_pad > 0:
        if N_ch_2 > 0:
            perm_parts.append(ch_sort_idx[:N_ch_2])
        if N_ch_2_pad > N_ch_2:
            perm_parts.append(sentinel.expand(N_ch_2_pad - N_ch_2))
    if N_ch_4_pad > 0:
        if N_ch_4 > 0:
            perm_parts.append(ch_sort_idx[N_ch_2:N_ch_2 + N_ch_4])
        if N_ch_4_pad > N_ch_4:
            perm_parts.append(sentinel.expand(N_ch_4_pad - N_ch_4))
    if N_ch_8_out > 0:
        if N_ch_8 > 0:
            perm_parts.append(ch_sort_idx[N_ch_2 + N_ch_4:])
        if N_ch_8_out > N_ch_8:
            perm_parts.append(sentinel.expand(N_ch_8_out - N_ch_8))
    if perm_parts:
        ch_perm_padded_idx = torch.cat(perm_parts, dim=0).contiguous()
    else:
        ch_perm_padded_idx = torch.empty(0, dtype=torch.long, device=device)
    assert ch_perm_padded_idx.numel() == D_padded, (
        f"perm_padded length {ch_perm_padded_idx.numel()} != D_padded={D_padded}"
    )

    return (K_2bit, K_4bit, K_8bit, K_ch_scale, K_ch_zp, ch_sort_idx,
            ch_perm_padded_idx, (N_ch_2, N_ch_4, N_ch_8))


# Alias: `pack_k_mixed_single_head` is the single-head path the per-head wrapper
# calls internally. Same function, named for clarity at call sites.
pack_k_mixed_single_head = pack_k_mixed


def pack_k_mixed_per_head(
    K: Tensor,
    k_bits_ph: Tensor,
) -> Tuple[
    Optional[Tensor],  # K_2bit   [1, H_kv, T_eff, sub_2_max]
    Optional[Tensor],  # K_4bit   [1, H_kv, T_eff, sub_4_max]
    Optional[Tensor],  # K_8bit   [1, H_kv, T_eff, max_N_ch_8]
    Tensor,            # K_ch_scale [1, H_kv, 1, D_kept_padded_max]
    Tensor,            # K_ch_zp    [1, H_kv, 1, D_kept_padded_max]
    Tensor,            # K_ch_sort_idx_per_head       [H_kv, D_kept_max] long, tail = 0
    Tensor,            # K_ch_perm_padded_idx_per_head [H_kv, D_padded_max] long, tail = 0
    Tensor,            # K_ch_seg_bounds_per_head     [H_kv, 3] int32 UNPADDED (n2, n4, n8)
    Tuple[int, int, int],  # shared-max seg bounds: (max_N_ch_2, max_N_ch_4, max_N_ch_8)
]:
    """Per-head variant of ``pack_k_mixed``.

    Two-pass: first count each head's segment sizes via a single-head pack
    probe (discarded), then compute shared-max channel targets, and finally
    re-pack each head with ``sub_2_target / sub_4_target / n_ch_8_target``
    so quarter-/half-split byte packing aligns channels to the shared-max
    layout. Pad channels are zero → scale = zp = 0 → dequant = 0. No byte-
    stream padding happens at any point — all padding is channel-space before
    ``quantize_and_pack``.

    Args:
        K:        [1, H_kv, T_eff, D] FP16
        k_bits_ph: [H_kv, D] LongTensor with values in {0, 2, 4, 8, 16} per head.

    Returns (9-tuple): ``K_ch_seg_bounds_per_head[h]`` stores each head's
    unpadded ``(n2, n4, n8)``; the final tuple is the shared-max bounds used
    by the Phase-2 K kernel (which still reads a scalar triple). Phase 3 will
    switch the kernel to per-head bounds.
    """
    assert K.dim() == 4 and K.shape[0] == 1, (
        f"Expected K [1,H,T,D], got {tuple(K.shape)}"
    )
    H_kv = int(K.shape[1])
    T_eff = int(K.shape[2])
    D = int(K.shape[3])
    device = K.device
    assert k_bits_ph.dim() == 2 and k_bits_ph.shape[0] == H_kv, (
        f"k_bits_ph must be [H_kv={H_kv}, D={D}], got {tuple(k_bits_ph.shape)}"
    )

    # ----------------------------------------------------------------
    # Pass 1: probe per-head segment counts (no quantize, just counting).
    # We derive (n2, n4, n8) from k_bits_h alone — same logic as the
    # pre-quantize sort inside pack_k_mixed, without running the expensive
    # quantize / bit-pack steps.
    # ----------------------------------------------------------------
    if os.environ.get("OBKV_BLOCKWISE_LEGACY_PATH", "0") == "1":
        n2_h, n4_h, n8_h = [], [], []
        for h in range(H_kv):
            kb = k_bits_ph[h].to(
                device=device, dtype=torch.long
            ).clamp(max=8)
            n2_h.append(int((kb == 2).sum().item()))
            n4_h.append(int((kb == 4).sum().item()))
            n8_h.append(int((kb == 8).sum().item()))
    else:
        k_bits_work = k_bits_ph.to(
            device=device, dtype=torch.long
        ).clamp(max=8)
        counts = torch.stack(
            [(k_bits_work == bit).sum(dim=1) for bit in (2, 4, 8)],
            dim=1,
        ).detach().cpu().tolist()
        n2_h = [int(row[0]) for row in counts]
        n4_h = [int(row[1]) for row in counts]
        n8_h = [int(row[2]) for row in counts]

    max_N_ch_2 = max(n2_h) if n2_h else 0
    max_N_ch_4 = max(n4_h) if n4_h else 0
    max_N_ch_8 = max(n8_h) if n8_h else 0
    max_N_ch_2_pad = _seg_padded_len(max_N_ch_2, 4)
    max_N_ch_4_pad = _seg_padded_len(max_N_ch_4, 2)
    sub_2_max = max_N_ch_2_pad // 4 if max_N_ch_2_pad > 0 else 0
    sub_4_max = max_N_ch_4_pad // 2 if max_N_ch_4_pad > 0 else 0
    D_padded_max = max_N_ch_2_pad + max_N_ch_4_pad + max_N_ch_8

    seg_bounds_per_head = torch.tensor(
        [[n2, n4, n8] for n2, n4, n8 in zip(n2_h, n4_h, n8_h)],
        dtype=torch.int32, device=device,
    )  # [H_kv, 3] UNPADDED

    # ----------------------------------------------------------------
    # Pass 2: pack each head targeting the shared-max layout. Padding
    # happens in channel space (F.pad on K_sorted before quantize), so
    # byte stream positions are correct by construction.
    # ----------------------------------------------------------------
    per_head: list = []
    for h in range(H_kv):
        K_h = K[:, h:h + 1, :, :].contiguous()
        kb_h = k_bits_ph[h]
        per_head.append(
            pack_k_mixed_single_head(
                K_h, kb_h,
                sub_2_target=sub_2_max if sub_2_max > 0 else None,
                sub_4_target=sub_4_max if sub_4_max > 0 else None,
                n_ch_8_target=max_N_ch_8 if max_N_ch_8 > 0 else None,
                segment_counts=(n2_h[h], n4_h[h], n8_h[h]),
            )
        )

    # ----------------------------------------------------------------
    # Stack per-head outputs along H_kv. Thanks to Pass 2's shared-max
    # targets, every head's K_2bit / K_4bit / K_8bit / scale / zp already
    # have the same channel shape; we just concat along dim=1.
    # ----------------------------------------------------------------
    def _concat_opt(idx: int) -> Optional[Tensor]:
        parts = [per_head[h][idx] for h in range(H_kv)]
        if all(p is None for p in parts):
            return None
        # If any head has a non-None tensor (which, by shared-max targets,
        # happens iff the corresponding max_N_ch_* > 0), every head must be
        # non-None too (since Pass 2 targets are uniform). Sanity-check:
        assert all(p is not None for p in parts), (
            f"tuple[{idx}]: mixed None/non-None across heads "
            f"— shared-max targeting failed"
        )
        return torch.cat(parts, dim=1).contiguous()  # [1, H_kv, T_eff, sub_max]

    K_2bit = _concat_opt(0)
    K_4bit = _concat_opt(1)
    K_8bit = _concat_opt(2)

    # Scale/zp: single-head already emits [1, 1, 1, D_padded_max] under
    # shared-max targeting (since channel pad happens pre-quantize). Concat
    # directly.
    scale_parts = [per_head[h][3] for h in range(H_kv)]
    zp_parts = [per_head[h][4] for h in range(H_kv)]
    if D_padded_max > 0:
        K_ch_scale = torch.cat(scale_parts, dim=1).contiguous()  # [1, H_kv, 1, D_padded_max]
        K_ch_zp = torch.cat(zp_parts, dim=1).contiguous()
    else:
        K_ch_scale = torch.empty(1, H_kv, 1, 0, dtype=torch.float16, device=device)
        K_ch_zp = torch.empty_like(K_ch_scale)

    # sort_idx (UNPADDED D_kept_h per head): pad-right with 0 sentinel to D_kept_max.
    D_kept_h = [n2h + n4h + n8h for n2h, n4h, n8h in zip(n2_h, n4_h, n8_h)]
    D_kept_max = max(D_kept_h) if D_kept_h else 0
    sort_parts = []
    for h in range(H_kv):
        idx_sort_h = per_head[h][5]  # [D_kept_h], unpadded
        if D_kept_max > 0:
            pad = D_kept_max - idx_sort_h.numel()
            if pad > 0:
                idx_sort_h = torch.cat(
                    [idx_sort_h,
                     torch.zeros(pad, dtype=torch.long, device=device)],
                    dim=0,
                )
            sort_parts.append(idx_sort_h)
        else:
            sort_parts.append(torch.empty(0, dtype=torch.long, device=device))
    if D_kept_max > 0:
        K_ch_sort_idx_per_head = torch.stack(sort_parts, dim=0)
    else:
        K_ch_sort_idx_per_head = torch.empty(H_kv, 0, dtype=torch.long, device=device)

    # perm_padded_idx (padded to D_padded_max per head, thanks to Pass 2
    # targets the per-head tensor is already at shared length).
    perm_parts = [per_head[h][6] for h in range(H_kv)]
    for h, p in enumerate(perm_parts):
        assert p.numel() == D_padded_max, (
            f"head {h}: perm len {p.numel()} != D_padded_max {D_padded_max}"
        )
    if D_padded_max > 0:
        K_ch_perm_padded_idx_per_head = torch.stack(perm_parts, dim=0)
    else:
        K_ch_perm_padded_idx_per_head = torch.empty(H_kv, 0, dtype=torch.long, device=device)

    return (
        K_2bit, K_4bit, K_8bit,
        K_ch_scale, K_ch_zp,
        K_ch_sort_idx_per_head,
        K_ch_perm_padded_idx_per_head,
        seg_bounds_per_head,
        (max_N_ch_2, max_N_ch_4, max_N_ch_8),
    )


def unpack_k_mixed(
    packed: "PackedKVLayer",
    D: int,
) -> Tensor:
    """Reverse of pack_k_mixed / pack_k_mixed_per_head. Used for correctness
    testing (Gate 1) and Phase-2 smoke.

    Handles both layouts:
      - Legacy single-head layout: shared ``K_ch_sort_idx [D_kept]`` and
        ``K_ch_seg_bounds: tuple``; dequant + single ``index_copy_`` scatter.
      - Per-head layout (Phase 2): ``K_ch_sort_idx_per_head [H_kv, D_kept_max]``
        and ``K_ch_seg_bounds_per_head [H_kv, 3]``. Under shared-max packing,
        every head's K_*bit / scale / zp is already shared-layout, but each
        head has its own sorted channel order, so scatter-to-D happens per
        head using each head's ``K_ch_sort_idx_per_head[h]`` (sliced to the
        head's real ``D_kept_h = n2 + n4 + n8``).

    Args:
        packed: PackedKVLayer.
        D: original channel count (e.g. 128).

    Returns:
        K_hat: [1, H_kv, T_eff, D] FP16.
    """
    if packed.K_ch_scale is None:
        raise ValueError("unpack_k_mixed: K_ch_scale missing")
    H_kv = packed.K_ch_scale.shape[1]
    # TriZone: K T-dim length is ``T_eff_k`` when populated (T_v + n_v16),
    # else fall back to ``T_eff`` for legacy layers that pre-date TriZone.
    T_eff = packed.T_eff_k if packed.T_eff_k > 0 else packed.T_eff
    device = packed.K_ch_scale.device

    per_head_mode = packed.K_ch_seg_bounds_per_head is not None
    if per_head_mode:
        assert packed.K_ch_sort_idx_per_head is not None, (
            "per-head layout requires K_ch_sort_idx_per_head"
        )

    # Shared-layout shared-max bounds.
    sb_max = packed.K_ch_seg_bounds
    N_ch_2_max, N_ch_4_max, N_ch_8_max = sb_max
    N_ch_2_pad_max = _seg_padded_len(N_ch_2_max, 4)
    N_ch_4_pad_max = _seg_padded_len(N_ch_4_max, 2)

    # Dequantise full shared-layout [1, H_kv, T_eff, D_padded_max] first. Pad
    # channels have scale=zp=0 → (codes - zp) * scale = 0 automatically. This
    # is the fast path (single PyTorch op), no per-head loop needed for
    # dequant; per-head differentiation happens only in the scatter step.
    parts: list = []
    if packed.K_2bit is not None and N_ch_2_pad_max > 0:
        codes2 = transfer_2bit_to_8bit_batchwise(packed.K_2bit)  # [1, H, T, N_ch_2_pad_max]
        sc2 = packed.K_ch_scale[..., :N_ch_2_pad_max]
        zp2 = packed.K_ch_zp[..., :N_ch_2_pad_max]
        parts.append((codes2.float() - zp2.float()) * sc2.float())
    if packed.K_4bit is not None and N_ch_4_pad_max > 0:
        codes4 = transfer_4bit_to_8bit_batchwise(packed.K_4bit)
        off = N_ch_2_pad_max
        sc4 = packed.K_ch_scale[..., off:off + N_ch_4_pad_max]
        zp4 = packed.K_ch_zp[..., off:off + N_ch_4_pad_max]
        parts.append((codes4.float() - zp4.float()) * sc4.float())
    if packed.K_8bit is not None and N_ch_8_max > 0:
        off = N_ch_2_pad_max + N_ch_4_pad_max
        sc8 = packed.K_ch_scale[..., off:off + N_ch_8_max]
        zp8 = packed.K_ch_zp[..., off:off + N_ch_8_max]
        parts.append((packed.K_8bit.float() - zp8.float()) * sc8.float())

    if not parts:
        return torch.zeros(1, H_kv, T_eff, D,
                           dtype=torch.float16, device=device)

    K_hat_sorted = torch.cat(parts, dim=3).half()  # [1, H_kv, T_eff, D_padded_max]

    K_hat = torch.zeros(1, H_kv, T_eff, D, dtype=torch.float16, device=device)

    if per_head_mode:
        # Per-head scatter: each head's real channels occupy
        # [0 : n2_h), [N_ch_2_pad_max : N_ch_2_pad_max + n4_h),
        # [N_ch_2_pad_max + N_ch_4_pad_max : N_ch_2_pad_max + N_ch_4_pad_max + n8_h)
        # in the sorted-layout tensor, and the scatter target channels are
        # K_ch_sort_idx_per_head[h][:D_kept_h] in the original D-space.
        sb_ph = packed.K_ch_seg_bounds_per_head  # [H_kv, 3] int32
        sort_ph = packed.K_ch_sort_idx_per_head  # [H_kv, D_kept_max] long
        for h in range(H_kv):
            n2 = int(sb_ph[h, 0].item())
            n4 = int(sb_ph[h, 1].item())
            n8 = int(sb_ph[h, 2].item())
            D_kept_h = n2 + n4 + n8
            if D_kept_h == 0:
                continue

            # Collect this head's real-channel dequantised values in the
            # sorted order: [n2 from seg2 | n4 from seg4 | n8 from seg8].
            head_parts = []
            if n2 > 0:
                head_parts.append(K_hat_sorted[:, h:h + 1, :, :n2])
            if n4 > 0:
                head_parts.append(
                    K_hat_sorted[:, h:h + 1, :,
                                 N_ch_2_pad_max:N_ch_2_pad_max + n4]
                )
            if n8 > 0:
                head_parts.append(
                    K_hat_sorted[:, h:h + 1, :,
                                 N_ch_2_pad_max + N_ch_4_pad_max:
                                 N_ch_2_pad_max + N_ch_4_pad_max + n8]
                )
            sorted_h = torch.cat(head_parts, dim=3)  # [1, 1, T, D_kept_h]
            sort_idx_h = sort_ph[h, :D_kept_h]       # [D_kept_h]
            # index_copy_ along channel dim at head-h slice:
            K_hat[:, h:h + 1, :, :].index_copy_(3, sort_idx_h, sorted_h)
    else:
        # Legacy: shared K_ch_sort_idx [D_kept], dequantised output is already
        # in sorted-then-padded layout [D_padded_max]. Drop the pad tail and
        # scatter by the shared sort_idx.
        D_kept = N_ch_2_max + N_ch_4_max + N_ch_8_max
        # Rebuild sorted-unpadded [1, H, T, D_kept] from the padded layout.
        drop_parts = []
        if N_ch_2_max > 0:
            drop_parts.append(K_hat_sorted[..., :N_ch_2_max])
        if N_ch_4_max > 0:
            drop_parts.append(K_hat_sorted[..., N_ch_2_pad_max:N_ch_2_pad_max + N_ch_4_max])
        if N_ch_8_max > 0:
            off = N_ch_2_pad_max + N_ch_4_pad_max
            drop_parts.append(K_hat_sorted[..., off:off + N_ch_8_max])
        sorted_unpad = torch.cat(drop_parts, dim=3) if drop_parts else K_hat_sorted[..., :0]
        if D_kept > 0:
            K_hat.index_copy_(3, packed.K_ch_sort_idx[:D_kept], sorted_unpad)

    return K_hat


# ── V packing (mixed-precision segments, per-token) ─────────────────


def pack_v_segments(
    V: Tensor,
    v_bits: Tensor,
    *,
    segment_counts: Optional[Tuple[int, int, int]] = None,
) -> Tuple[
    Optional[Tensor],  # V_2bit
    Optional[Tensor],  # V_4bit
    Optional[Tensor],  # V_8bit
    Tensor,            # V_scale
    Tensor,            # V_zp
    Tensor,            # sort_idx
    Tuple[int, int, int],  # seg_bounds (N_2, N_4, N_8)
]:
    """Sort tokens by assigned bit-width and pack each V segment."""
    T_eff = V.shape[2]
    D = V.shape[3]
    device = V.device

    sort_idx = torch.argsort(v_bits, stable=True)
    V_sorted = V[:, :, sort_idx, :]
    v_bits_sorted = v_bits[sort_idx]

    if segment_counts is None:
        N_2 = int((v_bits_sorted == 2).sum().item())
        N_4 = int((v_bits_sorted == 4).sum().item())
        N_8 = int((v_bits_sorted == 8).sum().item())
    else:
        N_2, N_4, N_8 = map(int, segment_counts)
    assert N_2 + N_4 + N_8 == T_eff, (
        f"Segment sizes {N_2}+{N_4}+{N_8} != T_eff={T_eff}"
    )

    scale_parts: list[Tensor] = []
    zp_parts: list[Tensor] = []

    V_2bit: Optional[Tensor] = None
    if N_2 > 0:
        seg = V_sorted[:, :, :N_2, :]
        codes, sc, zp = quantize_and_pack(seg, 2, dim=3)
        V_2bit = transfer_8bit_to_2bit_batchwise(codes)
        scale_parts.append(sc)
        zp_parts.append(zp)

    V_4bit: Optional[Tensor] = None
    if N_4 > 0:
        seg = V_sorted[:, :, N_2:N_2 + N_4, :]
        codes, sc, zp = quantize_and_pack(seg, 4, dim=3)
        V_4bit = transfer_8bit_to_4bit_batchwise(codes)
        scale_parts.append(sc)
        zp_parts.append(zp)

    V_8bit: Optional[Tensor] = None
    if N_8 > 0:
        seg = V_sorted[:, :, N_2 + N_4:, :]
        codes, sc, zp = quantize_and_pack(seg, 8, dim=3)
        V_8bit = codes
        scale_parts.append(sc)
        zp_parts.append(zp)

    V_scale = torch.cat(scale_parts, dim=2)
    V_zp = torch.cat(zp_parts, dim=2)

    return V_2bit, V_4bit, V_8bit, V_scale, V_zp, sort_idx, (N_2, N_4, N_8)


# ── Single-layer packing (for layer-streaming prefill) ─────────────


def _empty_packed_layer(H_kv: int, D: int, device, dtype) -> "PackedKVLayer":
    empty_v = torch.empty(1, H_kv, 0, 1, dtype=dtype, device=device)
    empty_k_sv = torch.empty(1, H_kv, 1, 0, dtype=dtype, device=device)
    return PackedKVLayer(
        K_2bit=None, K_4bit=None, K_8bit=None,
        K_ch_scale=empty_k_sv, K_ch_zp=empty_k_sv.clone(),
        K_ch_sort_idx=torch.empty(0, dtype=torch.long, device=device),
        K_ch_seg_bounds=(0, 0, 0),
        K_ch_perm_padded_idx=torch.empty(0, dtype=torch.long, device=device),
        V_2bit=None, V_4bit=None, V_8bit=None,
        V_scale=empty_v, V_zp=empty_v.clone(),
        sort_idx=torch.empty(0, dtype=torch.long, device=device),
        seg_bounds=(0, 0, 0),
        T_eff=0,
    )


def build_packed_layer(
    K: Tensor,
    V: Tensor,
    v_bits: Tensor,
    k_bits: Tensor,
    *,
    gqa_factor: Optional[int] = None,
    v_segment_counts: Optional[Tuple[int, int, int]] = None,
    defer_k_for_per_head_stack: bool = False,
) -> Tuple[PackedKVLayer, int, Optional[Tensor]]:
    """Pack one layer's KV cache with per-token V bits and per-channel K bits.

    TriZone (方案 1) layout: v=16 tokens have their K merged into the packed
    K stream (per-channel quant) while their V is returned separately as FP16
    ``v16_V`` so the decode path can bmm it unquantised.

    K_for_pack T-dim layout contract (consumers depend on this):
      - ``[0, T_v)`` = compressed tokens, sorted by ``pack_v_segments.sort_idx``
        so column ``t`` of ``K_for_pack`` aligns with column ``t`` of the V
        segments ``V_2bit/V_4bit/V_8bit``. Decode: ``w[:, :T_v]`` feeds the V
        Triton kernel.
      - ``[T_v, T_eff_k)`` = v=16 tokens in ``v16_ids`` ascending order (the
        natural ``nonzero`` ordering of the original T axis). ``v16_V`` is
        ``V[:, :, v16_ids, :]`` in the same order. Decode: ``w[:, T_v:T_eff_k]``
        feeds ``bmm(w, v16_V)``.
    Neither segment's internal order may be shuffled downstream.

    Args:
        K:      [1, H_kv, T, D] FP16
        V:      [1, H_kv, T, D] FP16
        v_bits: [T] LongTensor, values in {0, 2, 4, 8, 16}
        k_bits: either
                  - [D] LongTensor → legacy shared-across-heads layout, or
                  - [H_kv, D] LongTensor → per-head layout (Phase 2+).
                Values in {0, 2, 4, 8, 16} per channel.
        gqa_factor: H_q // H_kv (e.g. 4 for Llama-3.1-8B, 1 for MHA models
                    like Llama-2-13B). REQUIRED when ``k_bits`` is 2D
                    (per-head); unused when 1D (shared). Caller must derive
                    from ``model.config.num_attention_heads //
                    num_key_value_heads`` — there is no safe default, since a
                    wrong value produces an ``index vs self`` size mismatch
                    in the decode K kernel's per-Q gather.

    Returns:
        (PackedKVLayer, T_eff_k, v16_V) where:
          - ``packed.T_eff`` is the compressed-token count ``T_v``,
          - ``packed.T_eff_k = T_v + n_v16`` is the K sequence length,
          - ``packed.n_v16`` is the v=16 token count,
          - ``v16_V`` is ``[1, H_kv, n_v16, D]`` FP16 (None when n_v16 == 0).
    """
    H_kv = K.shape[1]
    D = K.shape[3]
    device = K.device

    # 1) Evict tokens with v_bits == 0
    keep_mask = (v_bits > 0)
    keep_ids = keep_mask.nonzero(as_tuple=True)[0].sort().values
    T_eff_keep = keep_ids.numel()

    if T_eff_keep == 0:
        return _empty_packed_layer(H_kv, D, device, torch.float16), 0, None

    v_bits_kept = v_bits[keep_ids]  # no clamp — v=16 is a first-class bin now

    # 2) Split compressed (v ∈ {2,4,8}) from v=16 along the kept token axis.
    #    nonzero returns indices in ascending order (PyTorch guarantee), so
    #    ``comp_ids`` / ``v16_ids`` inherit the original input-T order.
    is_comp = (v_bits_kept != 16)
    is_v16 = (v_bits_kept == 16)
    comp_local = is_comp.nonzero(as_tuple=True)[0]
    v16_local = is_v16.nonzero(as_tuple=True)[0]
    comp_ids = keep_ids[comp_local]
    v16_ids = keep_ids[v16_local]
    T_v = int(comp_ids.numel())
    n_v16 = int(v16_ids.numel())
    T_eff_k = T_v + n_v16

    # 3) Pack V segments over compressed tokens only. v=16 V is returned
    #    separately as FP16 so decode can bmm it without quantisation.
    if T_v > 0:
        V_comp = V[:, :, comp_ids, :]
        v_bits_comp = v_bits_kept[comp_local]          # values ∈ {2, 4, 8}
        K_comp = K[:, :, comp_ids, :]
        V_2bit, V_4bit, V_8bit, V_scale, V_zp, sort_idx, seg_bounds = (
            pack_v_segments(
                V_comp,
                v_bits_comp,
                segment_counts=v_segment_counts,
            )
        )
        K_comp_sorted = K_comp[:, :, sort_idx, :].contiguous()
    else:
        # All kept tokens are v=16. V segments are empty but the scale/zp
        # tensors are still shaped ``[1, H_kv, 0, 1]`` for downstream slice
        # consistency.
        V_2bit = V_4bit = V_8bit = None
        V_scale = torch.empty(1, H_kv, 0, 1, dtype=torch.float16, device=device)
        V_zp = torch.empty(1, H_kv, 0, 1, dtype=torch.float16, device=device)
        sort_idx = torch.empty(0, dtype=torch.long, device=device)
        seg_bounds = (0, 0, 0)
        K_comp_sorted = torch.empty(1, H_kv, 0, D, dtype=K.dtype, device=device)

    # v=16 V stays FP16 in input-T ascending order (v16_ids order).
    v16_V = V[:, :, v16_ids, :].contiguous() if n_v16 > 0 else None

    # 4) K_for_pack = [K_comp_sorted (T_v) ; K_v16 (n_v16)] along T.
    if n_v16 > 0:
        K_v16 = K[:, :, v16_ids, :]
        K_for_pack = torch.cat([K_comp_sorted, K_v16], dim=2).contiguous()
    else:
        K_for_pack = K_comp_sorted

    if defer_k_for_per_head_stack:
        if k_bits.dim() != 1 or H_kv != 1:
            raise ValueError(
                "deferred K packing expects one head and 1-D k_bits"
            )
        empty_k_meta = torch.empty(
            1, H_kv, 1, 0, dtype=torch.float16, device=device
        )
        packed = PackedKVLayer(
            K_2bit=None,
            K_4bit=None,
            K_8bit=None,
            K_ch_scale=empty_k_meta,
            K_ch_zp=empty_k_meta.clone(),
            K_ch_sort_idx=torch.empty(
                0, dtype=torch.long, device=device
            ),
            K_ch_seg_bounds=(0, 0, 0),
            K_ch_perm_padded_idx=torch.empty(
                0, dtype=torch.long, device=device
            ),
            V_2bit=V_2bit,
            V_4bit=V_4bit,
            V_8bit=V_8bit,
            V_scale=V_scale,
            V_zp=V_zp,
            sort_idx=sort_idx,
            seg_bounds=seg_bounds,
            T_eff=T_v,
            T_eff_k=T_eff_k,
            n_v16=n_v16,
            deferred_k_for_stack=K_for_pack,
        )
        return packed, T_eff_k, v16_V

    # 5) Pack K (per-channel quant). The 1D/2D branches are unchanged — the
    #    only difference is that T dim is now T_eff_k instead of T_eff.
    if k_bits.dim() == 2:
        assert k_bits.shape == (H_kv, D), (
            f"k_bits 2D must be [H_kv={H_kv}, D={D}], got {tuple(k_bits.shape)}"
        )
        assert gqa_factor is not None, (
            "build_packed_layer: gqa_factor is required when k_bits is 2D "
            "(per-head). Pass gqa_factor = model.config.num_attention_heads "
            "// num_key_value_heads — there is no safe default."
        )
        (K_2bit, K_4bit, K_8bit,
         K_ch_scale, K_ch_zp,
         K_ch_sort_idx_per_head,
         K_ch_perm_padded_idx_per_head,
         K_ch_seg_bounds_per_head,
         K_ch_seg_bounds_max) = pack_k_mixed_per_head(K_for_pack, k_bits)

        H_q = H_kv * gqa_factor
        K_ch_perm_padded_idx_for_Q = (
            K_ch_perm_padded_idx_per_head.repeat_interleave(gqa_factor, dim=0)
        )
        assert K_ch_perm_padded_idx_for_Q.shape == (
            H_q, K_ch_perm_padded_idx_per_head.shape[1]
        )

        K_ch_sort_idx_head0 = K_ch_sort_idx_per_head[0]
        K_ch_perm_padded_idx_head0 = K_ch_perm_padded_idx_per_head[0]

        packed = PackedKVLayer(
            K_2bit=K_2bit, K_4bit=K_4bit, K_8bit=K_8bit,
            K_ch_scale=K_ch_scale, K_ch_zp=K_ch_zp,
            K_ch_sort_idx=K_ch_sort_idx_head0,
            K_ch_seg_bounds=K_ch_seg_bounds_max,
            K_ch_perm_padded_idx=K_ch_perm_padded_idx_head0,
            V_2bit=V_2bit, V_4bit=V_4bit, V_8bit=V_8bit,
            V_scale=V_scale, V_zp=V_zp,
            sort_idx=sort_idx,
            seg_bounds=seg_bounds,
            T_eff=T_v,
            K_ch_sort_idx_per_head=K_ch_sort_idx_per_head,
            K_ch_perm_padded_idx_per_head=K_ch_perm_padded_idx_per_head,
            K_ch_seg_bounds_per_head=K_ch_seg_bounds_per_head,
            K_ch_perm_padded_idx_for_Q=K_ch_perm_padded_idx_for_Q,
            T_eff_k=T_eff_k,
            n_v16=n_v16,
        )
    else:
        (K_2bit, K_4bit, K_8bit,
         K_ch_scale, K_ch_zp, K_ch_sort_idx, K_ch_perm_padded_idx,
         K_ch_seg_bounds) = pack_k_mixed(K_for_pack, k_bits)

        packed = PackedKVLayer(
            K_2bit=K_2bit, K_4bit=K_4bit, K_8bit=K_8bit,
            K_ch_scale=K_ch_scale, K_ch_zp=K_ch_zp,
            K_ch_sort_idx=K_ch_sort_idx, K_ch_seg_bounds=K_ch_seg_bounds,
            K_ch_perm_padded_idx=K_ch_perm_padded_idx,
            V_2bit=V_2bit, V_4bit=V_4bit, V_8bit=V_8bit,
            V_scale=V_scale, V_zp=V_zp,
            sort_idx=sort_idx,
            seg_bounds=seg_bounds,
            T_eff=T_v,
            T_eff_k=T_eff_k,
            n_v16=n_v16,
        )
    return packed, T_eff_k, v16_V


# ── Per-head joint-knapsack packing helper ──────────────────────────


def _pad_and_stack_per_head_layers(
    layers: List[PackedKVLayer],
    per_head_v16_V: List[Optional[Tensor]],
    H_q: int,
    gqa_factor: int,
    head_dim: int,
    device,
    dtype,
    *,
    k_bits_per_head: Optional[Tensor] = None,
    v_bits_per_head: Optional[Tensor] = None,
    retain_allocation_metadata: bool = False,
) -> Tuple[PackedKVLayer, Optional[Tensor]]:
    """Stack H_kv per-head packed layers into a single padded PackedKVLayer
    under the TriZone (方案 1) stripe layout.

    Stripe layout (K T-dim):
      ``[0, max_T_v)``                       — Zone A (compressed), per head
                                                 fills ``[0, t_v_h)`` and
                                                 ``[t_v_h, max_T_v)`` is pad.
      ``[max_T_v, max_T_v + max_n_v16)``     — Zone B (v=16 stripe), per head
                                                 fills ``[max_T_v, max_T_v + n_v16_h)``
                                                 and ``[max_T_v + n_v16_h,
                                                 max_T_eff_k)`` is pad.

    ``softmax_mask`` covers both pad regions with -inf; V kernel only ever
    consumes the Zone-A prefix (``w[:, :max_T_v]``), so Zone B weights feed
    ``bmm(w[:, max_T_v:max_T_eff_k], new_v_only)`` on the decode side.

    Args:
        layers:         [H_kv] per-head PackedKVLayer from build_packed_layer
                        called on [1, 1, T_h, D] per-head input.
        per_head_v16_V: [H_kv] FP16 v=16 V tensor per head; each either None
                        or ``[1, 1, n_v16_h, D]``. Must be in the same T
                        order as the K stripe ([T_v, T_eff_k) of each inner
                        layer's packed K).
        H_q, gqa_factor, head_dim, device, dtype: shape / device context.
        k_bits_per_head: [H_kv, D] LongTensor routing the Phase 2 per-head
                        K rebuild path; None = legacy shared 1D layout.

    Returns:
        (packed_layer, new_v_only)
          packed_layer.T_eff              = max_h(T_v_h)              [max compressed count]
          packed_layer.T_eff_k            = max_T_v + max_n_v16        [K T-dim for QK dot]
          packed_layer.n_v16              = max_h(n_v16_h)
          packed_layer.T_eff_per_head     [H_kv] int32                [compressed per head]
          packed_layer.T_eff_k_per_head   [H_kv] int32                [K len per head]
          packed_layer.n_v16_per_head     [H_kv] int32
          packed_layer.softmax_mask       [H_q, max_T_eff_k] FP32 additive (4-seg)
          packed_layer.fp16_gap_mask      None                        (deprecated)
          packed_layer.n_fp16_per_head    None                        (deprecated)
          new_v_only: [1, H_kv, max_n_v16, D] FP16, or None when max_n_v16 == 0.
    """
    H_kv = len(layers)
    assert H_q == H_kv * gqa_factor, f"H_q={H_q}, H_kv={H_kv}, gqa={gqa_factor}"

    # ---- Per-head T_v / n_v16 / T_eff_k and V seg_bounds ----
    t_v_h = [int(pl.T_eff) for pl in layers]
    n_v16_h = [int(pl.n_v16) for pl in layers]
    # Inner layers may come from pre-TriZone code paths where T_eff_k is 0.
    # In that case the inner layer has no v=16 zone, so T_eff_k == T_v.
    t_eff_k_h = [
        (int(pl.T_eff_k) if pl.T_eff_k > 0 else int(pl.T_eff)) for pl in layers
    ]
    seg_h = [tuple(pl.seg_bounds) for pl in layers]
    max_T_v = max(t_v_h) if t_v_h else 0
    max_n_v16 = max(n_v16_h) if n_v16_h else 0
    max_T_eff_k = max_T_v + max_n_v16
    max_N_2 = max((sb[0] for sb in seg_h), default=0)
    max_N_4 = max((sb[1] for sb in seg_h), default=0)
    max_N_8 = max((sb[2] for sb in seg_h), default=0)

    seg_bounds_per_head = torch.tensor(
        [[sb[0], sb[1], sb[2]] for sb in seg_h],
        dtype=torch.int32, device=device,
    )  # [H_kv, 3]
    T_eff_per_head = torch.tensor(t_v_h, dtype=torch.int32, device=device)      # [H_kv]
    T_eff_k_per_head = torch.tensor(t_eff_k_h, dtype=torch.int32, device=device)
    n_v16_per_head = torch.tensor(n_v16_h, dtype=torch.int32, device=device)

    # ---- K channel metadata: same two-mode dispatch as before. ----
    per_head_k_mode = k_bits_per_head is not None
    ref_pl = layers[0]
    for pl in layers:
        if pl.T_eff_k > 0 or pl.T_eff > 0:
            ref_pl = pl
            break
    if not per_head_k_mode:
        K_ch_sort_idx = ref_pl.K_ch_sort_idx
        K_ch_seg_bounds = ref_pl.K_ch_seg_bounds
        K_ch_perm_padded_idx = ref_pl.K_ch_perm_padded_idx
        N_ch_2, N_ch_4, N_ch_8 = K_ch_seg_bounds
        N_ch_2p = _seg_padded_len(N_ch_2, 4)
        N_ch_4p = _seg_padded_len(N_ch_4, 2)
        D_kept_padded = N_ch_2p + N_ch_4p + N_ch_8

    # ---- V segment stacking (compressed tokens only, no stripe) ----
    def _stack_v_segment(n_max: int, sub_d: int) -> Optional[Tensor]:
        if n_max == 0:
            return None
        parts = []
        for pl, (n_2h, n_4h, n_8h) in zip(layers, seg_h):
            if sub_d == head_dim // 4:
                cur_n = n_2h; cur = pl.V_2bit
            elif sub_d == head_dim // 2:
                cur_n = n_4h; cur = pl.V_4bit
            else:
                cur_n = n_8h; cur = pl.V_8bit
            if cur is None or cur_n == 0:
                seg = torch.zeros(1, 1, n_max, sub_d, dtype=torch.uint8, device=device)
            elif cur_n < n_max:
                pad = torch.zeros(1, 1, n_max - cur_n, sub_d, dtype=cur.dtype, device=device)
                seg = torch.cat([cur, pad], dim=2)
            else:
                seg = cur
            parts.append(seg)
        return torch.cat(parts, dim=1).contiguous()  # [1, H_kv, n_max, sub_d]

    V_2bit = _stack_v_segment(max_N_2, head_dim // 4)
    V_4bit = _stack_v_segment(max_N_4, head_dim // 2)
    V_8bit = _stack_v_segment(max_N_8, head_dim)

    # ---- V_scale / V_zp: pad per-head to max_T_v (compressed only) ----
    def _stack_v_scale_like(field_name: str) -> Tensor:
        parts = []
        for pl, n_h in zip(layers, t_v_h):
            t = getattr(pl, field_name)  # [1, 1, T_v_h, 1] or [1, 1, 0, 1]
            if n_h == 0:
                seg = torch.zeros(1, 1, max_T_v, 1, dtype=torch.float16, device=device)
            elif n_h < max_T_v:
                pad = torch.zeros(1, 1, max_T_v - n_h, 1, dtype=t.dtype, device=device)
                seg = torch.cat([t, pad], dim=2)
            else:
                seg = t
            parts.append(seg)
        return torch.cat(parts, dim=1).contiguous()  # [1, H_kv, max_T_v, 1]

    if max_T_v > 0:
        V_scale = _stack_v_scale_like("V_scale")
        V_zp = _stack_v_scale_like("V_zp")
    else:
        V_scale = torch.empty(1, H_kv, 0, 1, dtype=torch.float16, device=device)
        V_zp = torch.empty_like(V_scale)

    # ---- K byte-segment stripe stacker (legacy 1-D K path) ----
    def _stack_k_segment_stripe(n_ch_sub: int, head_ref: str) -> Optional[Tensor]:
        if n_ch_sub == 0 or max_T_eff_k == 0:
            return None
        parts = []
        for pl, tv_h, nv16_h in zip(layers, t_v_h, n_v16_h):
            cur = getattr(pl, head_ref)  # [1, 1, T_eff_k_h, n_ch_sub] or None
            seg = torch.zeros(1, 1, max_T_eff_k, n_ch_sub, dtype=torch.uint8, device=device)
            if cur is None:
                parts.append(seg); continue
            # Inner K T-dim layout: [comp (tv_h); v16 (nv16_h)].
            if tv_h > 0:
                seg[:, :, :tv_h, :] = cur[:, :, :tv_h, :]
            if nv16_h > 0:
                seg[:, :, max_T_v:max_T_v + nv16_h, :] = cur[:, :, tv_h:tv_h + nv16_h, :]
            parts.append(seg)
        return torch.cat(parts, dim=1).contiguous()  # [1, H_kv, max_T_eff_k, n_ch_sub]

    if per_head_k_mode:
        # ---- Per-head K rebuild on stripe canvas --------------------------
        # Unpack each inner layer's full K (T_eff_k rows), split into
        # compressed + v=16 segments, place into stripe canvas, repack in one
        # shot via pack_k_mixed_per_head.
        unp_parts: list = []
        for pl, tv_h, nv16_h, tk_h in zip(layers, t_v_h, n_v16_h, t_eff_k_h):
            canvas = torch.zeros(
                1, 1, max_T_eff_k, head_dim,
                dtype=torch.float16, device=device,
            )
            if tk_h > 0:
                if pl.deferred_k_for_stack is not None:
                    K_h_full = pl.deferred_k_for_stack
                else:
                    K_h_full = unpack_k_mixed(
                        pl, D=head_dim
                    )  # [1, 1, tk_h, D]
                if tv_h > 0:
                    canvas[:, :, :tv_h, :] = K_h_full[:, :, :tv_h, :]
                if nv16_h > 0:
                    canvas[:, :, max_T_v:max_T_v + nv16_h, :] = (
                        K_h_full[:, :, tv_h:tv_h + nv16_h, :]
                    )
            unp_parts.append(canvas)
        K_stacked = torch.cat(unp_parts, dim=1).contiguous()  # [1, H_kv, max_T_eff_k, D]

        (K_2bit, K_4bit, K_8bit,
         K_ch_scale, K_ch_zp,
         K_ch_sort_idx_per_head,
         K_ch_perm_padded_idx_per_head,
         K_ch_seg_bounds_per_head_dev,
         K_ch_seg_bounds_max) = pack_k_mixed_per_head(
            K_stacked, k_bits_per_head.to(device=device),
        )
        K_ch_sort_idx = K_ch_sort_idx_per_head[0]
        K_ch_perm_padded_idx = K_ch_perm_padded_idx_per_head[0]
        K_ch_seg_bounds = K_ch_seg_bounds_max
        K_ch_perm_padded_idx_for_Q = (
            K_ch_perm_padded_idx_per_head.repeat_interleave(gqa_factor, dim=0)
        )
    else:
        K_2bit = _stack_k_segment_stripe(N_ch_2p // 4 if N_ch_2 > 0 else 0, "K_2bit")
        K_4bit = _stack_k_segment_stripe(N_ch_4p // 2 if N_ch_4 > 0 else 0, "K_4bit")
        K_8bit = _stack_k_segment_stripe(N_ch_8 if N_ch_8 > 0 else 0, "K_8bit")

        def _stack_k_ch_like(field_name: str) -> Tensor:
            parts = []
            for pl in layers:
                t = getattr(pl, field_name)  # [1, 1, 1, D_kept_padded] or [1, 1, 1, 0]
                if t.shape[3] != D_kept_padded:
                    t = torch.zeros(1, 1, 1, D_kept_padded, dtype=torch.float16, device=device)
                parts.append(t)
            return torch.cat(parts, dim=1).contiguous()

        K_ch_scale = _stack_k_ch_like("K_ch_scale")
        K_ch_zp = _stack_k_ch_like("K_ch_zp")
        K_ch_sort_idx_per_head = None
        K_ch_perm_padded_idx_per_head = None
        K_ch_seg_bounds_per_head_dev = None
        K_ch_perm_padded_idx_for_Q = None

    sort_idx = ref_pl.sort_idx

    # ---- 4-segment softmax_mask [H_q, max_T_eff_k] ----
    #   seg 1  [0, t_v_h)                            = 0      (Zone A valid)
    #   seg 2  [t_v_h, max_T_v)                      = -inf   (Zone A pad)
    #   seg 3  [max_T_v, max_T_v + n_v16_h)          = 0      (Zone B valid)
    #   seg 4  [max_T_v + n_v16_h, max_T_eff_k)      = -inf   (Zone B pad)
    if max_T_eff_k > 0:
        mask_kv = torch.zeros(H_kv, max_T_eff_k, dtype=torch.float32, device=device)
        for h in range(H_kv):
            tv_h = t_v_h[h]
            nv16 = n_v16_h[h]
            if tv_h < max_T_v:
                mask_kv[h, tv_h:max_T_v] = float("-inf")
            pad_start = max_T_v + nv16
            if pad_start < max_T_eff_k:
                mask_kv[h, pad_start:max_T_eff_k] = float("-inf")
        softmax_mask = mask_kv.repeat_interleave(gqa_factor, dim=0).contiguous()
    else:
        softmax_mask = torch.empty(H_q, 0, dtype=torch.float32, device=device)

    # ---- new_v_only: pad per-head to max_n_v16, concat along H_kv. ----
    if max_n_v16 > 0:
        parts = []
        for h in range(H_kv):
            v16 = per_head_v16_V[h]
            cur_n = n_v16_h[h]
            if v16 is None or cur_n == 0:
                seg = torch.zeros(1, 1, max_n_v16, head_dim, dtype=torch.float16, device=device)
            elif cur_n < max_n_v16:
                pad = torch.zeros(1, 1, max_n_v16 - cur_n, head_dim, dtype=v16.dtype, device=device)
                seg = torch.cat([v16, pad], dim=2)
            else:
                seg = v16
            parts.append(seg)
        new_v_only = torch.cat(parts, dim=1).contiguous()  # [1, H_kv, max_n_v16, D]
    else:
        new_v_only = None

    packed = PackedKVLayer(
        K_2bit=K_2bit, K_4bit=K_4bit, K_8bit=K_8bit,
        K_ch_scale=K_ch_scale, K_ch_zp=K_ch_zp,
        K_ch_sort_idx=K_ch_sort_idx, K_ch_seg_bounds=K_ch_seg_bounds,
        K_ch_perm_padded_idx=K_ch_perm_padded_idx,
        V_2bit=V_2bit, V_4bit=V_4bit, V_8bit=V_8bit,
        V_scale=V_scale, V_zp=V_zp,
        sort_idx=sort_idx,
        seg_bounds=(max_N_2, max_N_4, max_N_8),
        T_eff=max_T_v,
        seg_bounds_per_head=seg_bounds_per_head,
        T_eff_per_head=T_eff_per_head,
        softmax_mask=softmax_mask,
        # DEPRECATED: v=16 padding now folded into softmax_mask (方案 1 TriZone).
        n_fp16_per_head=None,
        fp16_gap_mask=None,
        K_ch_sort_idx_per_head=K_ch_sort_idx_per_head,
        K_ch_perm_padded_idx_per_head=K_ch_perm_padded_idx_per_head,
        K_ch_seg_bounds_per_head=K_ch_seg_bounds_per_head_dev,
        K_ch_perm_padded_idx_for_Q=K_ch_perm_padded_idx_for_Q,
        T_eff_k=max_T_eff_k,
        n_v16=max_n_v16,
        T_eff_k_per_head=T_eff_k_per_head,
        n_v16_per_head=n_v16_per_head,
        allocation_v_bits=(
            v_bits_per_head.to(device=device, dtype=torch.int16)
            if retain_allocation_metadata and v_bits_per_head is not None
            else None
        ),
        allocation_k_bits=(
            k_bits_per_head.to(device=device, dtype=torch.int16)
            if retain_allocation_metadata and k_bits_per_head is not None
            else None
        ),
    )
    return packed, new_v_only


# ── Full cache builder ──────────────────────────────────────────────


def build_packed_cache(
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
    v_bits: Tensor,
    k_bits: Tensor,
    *,
    original_seq_len: int,
    gqa_factor: Optional[int] = None,
) -> TriZoneCache:
    """Build a TriZoneCache from legacy (K, V) tuples.

    Args:
        past_key_values: tuple of (K, V) per layer; each K,V is [B, H_kv, T, D].
        v_bits: [T_eff] LongTensor with per-token V bit assignment.
        k_bits: one of
                  - [D]         LongTensor, shared across layers and heads;
                  - [L, D]      LongTensor, per-layer shared across heads;
                  - [H_kv, D]   LongTensor, per-head shared across layers
                                (Phase 2 per-head packing), or
                  - [L, H_kv, D] LongTensor, per-layer per-head.
                Shape dispatch:
                  - ndim == 1: shared layout
                  - ndim == 2 and shape[0] == num_layers: per-layer 1D
                  - ndim == 2 and shape[0] == H_kv: per-head (routed to
                    ``pack_k_mixed_per_head`` inside build_packed_layer)
                  - ndim == 3: per-layer per-head
                Ambiguity note: if num_layers happens to equal H_kv, the
                caller must pass 3D to disambiguate.
        original_seq_len: true pre-eviction prefill length (input_ids.shape[1]).
            Used as RoPE ``cache_position`` origin during decode. Callers must
            pass this explicitly — deriving it from ``packed_layers[0].T_eff``
            is wrong whenever any token has ``v_bits == 0`` (eviction).
        gqa_factor: H_q // H_kv. REQUIRED when ``k_bits`` is per-head (2D with
            shape[0]==H_kv, or 3D). Unused when ``k_bits`` is shared-layout 1D
            or 2D-per-layer. See ``build_packed_layer`` for details.

    Returns:
        TriZoneCache with packed layers, per-layer ``new_v_only`` (FP16 v=16
        V) and empty ``new_both_k / new_both_v`` lists.
    """
    num_layers = len(past_key_values)
    first_K = past_key_values[0][0]
    H_kv = int(first_K.shape[1])

    if k_bits.dim() == 1:
        per_layer_k = False
        per_head_k = False
    elif k_bits.dim() == 2:
        if k_bits.shape[0] == num_layers and k_bits.shape[0] != H_kv:
            per_layer_k = True
            per_head_k = False
        elif k_bits.shape[0] == H_kv:
            per_layer_k = False
            per_head_k = True
        elif k_bits.shape[0] == num_layers:
            # num_layers == H_kv collision — default to per-layer for
            # backward compat. Callers wanting per-head in this case must
            # pass 3D.
            per_layer_k = True
            per_head_k = False
        else:
            raise ValueError(
                f"k_bits 2D shape {tuple(k_bits.shape)} matches neither "
                f"num_layers={num_layers} nor H_kv={H_kv}"
            )
    elif k_bits.dim() == 3:
        assert k_bits.shape[:2] == (num_layers, H_kv), (
            f"k_bits 3D must be [L={num_layers}, H_kv={H_kv}, D], "
            f"got {tuple(k_bits.shape)}"
        )
        per_layer_k = True
        per_head_k = True
    else:
        raise ValueError(f"k_bits must be 1D/2D/3D, got shape {tuple(k_bits.shape)}")

    packed_layers: list[PackedKVLayer] = []
    new_v_only_list: list[Optional[Tensor]] = []
    for layer_idx in range(num_layers):
        K, V = past_key_values[layer_idx]
        if per_layer_k and per_head_k:
            k_bits_i = k_bits[layer_idx]       # [H_kv, D]
        elif per_layer_k:
            k_bits_i = k_bits[layer_idx]       # [D]
        elif per_head_k:
            k_bits_i = k_bits                  # [H_kv, D] (shared across layers)
        else:
            k_bits_i = k_bits                  # [D]
        packed_layer, _, v16_V = build_packed_layer(
            K, V, v_bits, k_bits=k_bits_i, gqa_factor=gqa_factor,
        )
        packed_layers.append(packed_layer)
        new_v_only_list.append(v16_V)

    return TriZoneCache(
        packed=tuple(packed_layers),
        new_v_only=new_v_only_list,
        new_both_k=[None] * num_layers,
        new_both_v=[None] * num_layers,
        original_seq_len=int(original_seq_len),
    )


# ── Unpack helpers (testing / verification) ─────────────────────────


def unpack_v_segment(
    V_packed: Tensor,
    V_scale: Tensor,
    V_zp: Tensor,
    n_bits: int,
) -> Tensor:
    """Reverse of one V segment: unpack + dequantise → FP16."""
    if n_bits == 4:
        V_q = transfer_4bit_to_8bit_batchwise(V_packed)
    elif n_bits == 2:
        V_q = transfer_2bit_to_8bit_batchwise(V_packed)
    elif n_bits == 8:
        V_q = V_packed
    else:
        raise ValueError(f"Unsupported n_bits={n_bits}")

    V_hat = (V_q.float() - V_zp.float()) * V_scale.float()
    return V_hat.half()
