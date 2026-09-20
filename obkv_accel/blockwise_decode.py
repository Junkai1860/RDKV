"""Eager packed decode with immutable blockwise compression of decode KV.

The prompt remains in the regular TriZone cache.  Decode KV is split into
fixed-size blocks: a target block stays FP16 until the following block has
supplied its queries, then the target is allocated/packed once and appended
to an immutable list.  Attention performs one packed K/V kernel dispatch per
prompt/decode source and one FP16 dispatch for the pending suffix, followed by
a single global softmax over those sources.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from obkv_accel.fast_decode import (
    _pre_squeeze_packed,
    _rms_norm,
    pre_extract_weights,
)
from obkv_accel.packing import (
    PackedKVLayer,
    TriZoneCache,
    unpack_k_mixed,
)
from obkv_accel.triton_k_kernel import (
    k_mixed_qk_dot,
    k_mixed_qk_dot_into,
)
from obkv_accel.triton_rope import fused_rope
from obkv_accel.triton_v_kernel import (
    _get_v16_dummy,
    v_dequant_weighted_sum_dispatch,
    v_dequant_weighted_sum_dispatch_into,
)
from obkv_accel.trizone_decompress import _unpack_v_perhead


@dataclass(frozen=True)
class BlockwiseRDKVConfig:
    block_size: int
    block_budget_tokens: int
    k_budget_ratio: float
    obs_window: int
    pool_kernel_size: int
    pool_padding: str
    v_bit_options: torch.Tensor
    k_bit_options: torch.Tensor
    epsilon_v: Dict[int, float]
    epsilon_k: Dict[int, float]
    correctness_check: bool = False

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("decode block size must be positive")
        if self.block_budget_tokens <= 0:
            raise ValueError("decode block budget must be positive")
        if self.block_budget_tokens > self.block_size:
            raise ValueError(
                "decode block budget cannot exceed decode block size "
                f"({self.block_budget_tokens} > {self.block_size})"
            )
        if not 0.0 < self.k_budget_ratio < 1.0:
            raise ValueError("k_budget_ratio must be strictly between 0 and 1")


@dataclass
class CompressedDecodeBlock:
    block_id: int
    token_start: int
    token_end: int
    packed: Tuple[PackedKVLayer, ...]
    squeezed: List[PackedKVLayer]
    new_v_only: List[Optional[torch.Tensor]]
    logical_bits_per_layer: List[int]
    fp16_equivalent_tokens_per_layer: List[float]
    physical_bytes: int
    v_bit_histogram: Dict[str, int]
    k_bit_histogram: Dict[str, int]


def plan_block_compressions(
    generated_tokens: int, block_size: int
) -> List[Dict[str, Any]]:
    """Pure reference scheduler used by correctness tests and runtime checks."""
    if generated_tokens < 0 or block_size <= 0:
        raise ValueError("generated_tokens must be non-negative and block_size positive")
    plan: List[Dict[str, Any]] = []
    start = 0
    block_id = 0
    # Every complete successor block scores its complete predecessor while
    # generation is still in progress.
    while generated_tokens - start >= 2 * block_size:
        plan.append(
            {
                "block_id": block_id,
                "token_start": start,
                "token_end": start + block_size - 1,
                "query_source_block": block_id + 1,
                "is_final_flush": False,
                "query_mode": "next_block",
            }
        )
        start += block_size
        block_id += 1
    # At EOS/max length, drain a complete predecessor with any available
    # successor queries, then self-score the final (possibly partial) block.
    while generated_tokens - start > block_size:
        plan.append(
            {
                "block_id": block_id,
                "token_start": start,
                "token_end": start + block_size - 1,
                "query_source_block": block_id + 1,
                "is_final_flush": True,
                "query_mode": "next_block",
            }
        )
        start += block_size
        block_id += 1
    if start < generated_tokens:
        plan.append(
            {
                "block_id": block_id,
                "token_start": start,
                "token_end": generated_tokens - 1,
                "query_source_block": block_id,
                "is_final_flush": True,
                "query_mode": "self_causal",
            }
        )
    return plan


def _sync_stamp(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _tensor_storage_bytes(value: Any) -> int:
    """Count unique tensor storages reachable from a packed-cache object."""
    seen: set[Tuple[str, int]] = set()
    total = 0

    def visit(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, torch.Tensor):
            key = (str(obj.device), int(obj.untyped_storage().data_ptr()))
            if key not in seen:
                seen.add(key)
                total += int(obj.untyped_storage().nbytes())
            return
        if isinstance(obj, dict):
            for item in obj.values():
                visit(item)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
            return
        fields = getattr(obj, "__dataclass_fields__", None)
        if fields:
            for name in fields:
                visit(getattr(obj, name))

    visit(value)
    return total


def cache_physical_bytes(cache: TriZoneCache) -> int:
    return _tensor_storage_bytes((cache.packed, cache.new_v_only))


def _bit_histogram(bits: torch.Tensor) -> Dict[str, int]:
    bits = bits.detach()
    return {
        str(bit): int((bits == bit).sum().item())
        for bit in (0, 2, 4, 8, 16)
    }


def _merge_histogram(
    target: Dict[str, int], source: Dict[str, int]
) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _seg_padded_len(n: int, align: int) -> int:
    if n == 0:
        return 0
    return ((int(n) + align - 1) // align) * align


def _fixed_slot_enabled() -> bool:
    return os.environ.get("OBKV_BLOCKWISE_FIXED_SLOT", "0") == "1"


def _slot_bank_enabled() -> bool:
    return os.environ.get("OBKV_BLOCKWISE_SLOT_BANK", "0") == "1"


def _fixed_slot_targets(
    head_dim: int,
    *,
    block_size: int,
    block_budget_tokens: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    """Conservative fixed layout for a blockwise decode budget.

    At the default 50/50 K/V split, one FP16-equivalent budget token can
    retain at most eight 2-bit V positions. Reserve that worst-case capacity
    independently for each mixed-precision segment. This preserves the
    established 128/1 shapes and scales them to 256/2 without truncation.
    """
    max_v_positions = min(
        int(block_size), 8 * int(block_budget_tokens)
    )
    return (
        (max_v_positions, max_v_positions, max_v_positions),
        (head_dim, head_dim, head_dim),
        max_v_positions,
    )


def _make_empty_fixed_slot_layer(
    *,
    h_kv: int,
    h_q: int,
    gqa_factor: int,
    head_dim: int,
    block_size: int,
    block_budget_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[PackedKVLayer, torch.Tensor]:
    """Create one inactive blockwise slot with graph-stable tensor shapes.

    The slot uses the same conservative layout as real fixed slots, but its
    old-cache softmax mask is all ``-inf`` and every scale/zp byte tensor is
    zero.  Attention may therefore include all preallocated slots in the
    source list without changing numerics before a slot is populated.
    """
    target_v_bounds, target_k_bounds, target_n_v16 = _fixed_slot_targets(
        head_dim,
        block_size=block_size,
        block_budget_tokens=block_budget_tokens,
    )
    target_t_v = sum(target_v_bounds)
    target_t_k = target_t_v + target_n_v16
    n2, n4, n8 = target_k_bounds
    n2p = _seg_padded_len(n2, 4)
    n4p = _seg_padded_len(n4, 2)
    d_padded = n2p + n4p + n8

    seg_bounds_per_head = torch.tensor(
        [target_v_bounds], dtype=torch.int32, device=device
    ).repeat(h_kv, 1)
    t_eff_per_head = torch.full(
        (h_kv,), target_t_v, dtype=torch.int32, device=device
    )
    k_bounds_per_head = torch.tensor(
        [target_k_bounds], dtype=torch.int32, device=device
    ).repeat(h_kv, 1)
    t_eff_k_per_head = torch.full(
        (h_kv,), target_t_k, dtype=torch.int32, device=device
    )
    n_v16_per_head = torch.full(
        (h_kv,), target_n_v16, dtype=torch.int32, device=device
    )
    perm_per_head = torch.zeros(h_kv, d_padded, dtype=torch.long, device=device)
    perm_for_q = torch.zeros(h_q, d_padded, dtype=torch.long, device=device)
    sort_idx_ph = torch.arange(target_t_v, dtype=torch.long, device=device)
    k_sort_idx_ph = torch.zeros(
        h_kv, sum(target_k_bounds), dtype=torch.long, device=device
    )
    softmax_mask = torch.full(
        (h_q, target_t_k), float("-inf"), dtype=torch.float32, device=device
    )

    layer = PackedKVLayer(
        K_2bit=torch.zeros(1, h_kv, target_t_k, n2p // 4, dtype=torch.uint8, device=device),
        K_4bit=torch.zeros(1, h_kv, target_t_k, n4p // 2, dtype=torch.uint8, device=device),
        K_8bit=torch.zeros(1, h_kv, target_t_k, n8, dtype=torch.uint8, device=device),
        K_ch_scale=torch.zeros(1, h_kv, 1, d_padded, dtype=dtype, device=device),
        K_ch_zp=torch.zeros(1, h_kv, 1, d_padded, dtype=dtype, device=device),
        K_ch_sort_idx=k_sort_idx_ph[0],
        K_ch_seg_bounds=target_k_bounds,
        K_ch_perm_padded_idx=perm_per_head[0],
        V_2bit=torch.zeros(1, h_kv, target_v_bounds[0], head_dim // 4, dtype=torch.uint8, device=device),
        V_4bit=torch.zeros(1, h_kv, target_v_bounds[1], head_dim // 2, dtype=torch.uint8, device=device),
        V_8bit=torch.zeros(1, h_kv, target_v_bounds[2], head_dim, dtype=torch.uint8, device=device),
        V_scale=torch.zeros(1, h_kv, target_t_v, 1, dtype=dtype, device=device),
        V_zp=torch.zeros(1, h_kv, target_t_v, 1, dtype=dtype, device=device),
        sort_idx=sort_idx_ph,
        seg_bounds=target_v_bounds,
        T_eff=target_t_v,
        seg_bounds_per_head=seg_bounds_per_head,
        T_eff_per_head=t_eff_per_head,
        softmax_mask=softmax_mask,
        n_fp16_per_head=None,
        fp16_gap_mask=None,
        K_ch_sort_idx_per_head=k_sort_idx_ph,
        K_ch_perm_padded_idx_per_head=perm_per_head,
        K_ch_seg_bounds_per_head=k_bounds_per_head,
        K_ch_perm_padded_idx_for_Q=perm_for_q,
        T_eff_k=target_t_k,
        n_v16=target_n_v16,
        T_eff_k_per_head=t_eff_k_per_head,
        n_v16_per_head=n_v16_per_head,
        allocation_v_bits=None,
        allocation_k_bits=None,
    )
    new_v_slot = torch.zeros(
        1, h_kv, target_n_v16, head_dim, dtype=torch.float16, device=device
    )
    return layer, new_v_slot


def _copy_tensor_field(dst: Optional[torch.Tensor], src: Optional[torch.Tensor]) -> None:
    if dst is None or src is None:
        return
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError(f"fixed slot shape mismatch: dst={tuple(dst.shape)} src={tuple(src.shape)}")
    dst.copy_(src)


def _copy_fixed_slot_block_into_bank(
    *,
    dst_layers: Tuple[PackedKVLayer, ...],
    dst_v16: List[torch.Tensor],
    src_layers: Tuple[PackedKVLayer, ...],
    src_v16: List[Optional[torch.Tensor]],
) -> None:
    """Populate an already-allocated fixed slot in-place."""
    for dst, src, dst_v, src_v in zip(dst_layers, src_layers, dst_v16, src_v16):
        for name in (
            "K_2bit", "K_4bit", "K_8bit", "K_ch_scale", "K_ch_zp",
            "K_ch_sort_idx", "K_ch_perm_padded_idx", "V_2bit", "V_4bit",
            "V_8bit", "V_scale", "V_zp", "sort_idx",
            "seg_bounds_per_head", "T_eff_per_head", "softmax_mask",
            "K_ch_sort_idx_per_head", "K_ch_perm_padded_idx_per_head",
            "K_ch_seg_bounds_per_head", "K_ch_perm_padded_idx_for_Q",
            "T_eff_k_per_head", "n_v16_per_head",
        ):
            _copy_tensor_field(getattr(dst, name), getattr(src, name))
        _copy_tensor_field(dst_v, src_v)


def _copy_t_prefix(
    src: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    t_dim: int = 2,
) -> torch.Tensor:
    dst = torch.zeros(*shape, dtype=dtype, device=device)
    if src is None:
        return dst
    slices = [slice(None)] * len(shape)
    slices[t_dim] = slice(0, min(int(src.shape[t_dim]), int(shape[t_dim])))
    src_slices = [slice(None)] * src.dim()
    src_slices[t_dim] = slices[t_dim]
    dst[tuple(slices)] = src[tuple(src_slices)]
    return dst


def _pad_block_layer_to_fixed_slot(
    packed: PackedKVLayer,
    new_v_only: Optional[torch.Tensor],
    *,
    h_q: int,
    gqa_factor: int,
    head_dim: int,
    block_size: int,
    block_budget_tokens: int,
) -> Tuple[PackedKVLayer, torch.Tensor]:
    device = packed.K_ch_scale.device
    dtype = packed.K_ch_scale.dtype
    h_kv = int(packed.K_ch_scale.shape[1])
    target_v_bounds, target_k_bounds, target_n_v16 = _fixed_slot_targets(
        head_dim,
        block_size=block_size,
        block_budget_tokens=block_budget_tokens,
    )
    target_t_v = sum(target_v_bounds)
    target_t_k = target_t_v + target_n_v16

    src_t_v = int(packed.T_eff)
    src_n_v16 = int(packed.n_v16)
    src_t_k = int(packed.T_eff_k if packed.T_eff_k > 0 else packed.T_eff)
    if any(
        int(actual) > int(target)
        for actual, target in zip(packed.seg_bounds, target_v_bounds)
    ):
        raise ValueError(
            "fixed V slot capacity is too small: "
            f"actual={packed.seg_bounds} target={target_v_bounds}"
        )
    if src_n_v16 > target_n_v16:
        raise ValueError(
            "fixed V16 slot capacity is too small: "
            f"actual={src_n_v16} target={target_n_v16}"
        )

    # V tensors: keep each per-head compact V segment as-is, pad segment maxes.
    v2_t, v4_t, v8_t = target_v_bounds
    V_2bit = _copy_t_prefix(
        packed.V_2bit,
        shape=(1, h_kv, v2_t, head_dim // 4),
        dtype=torch.uint8,
        device=device,
    )
    V_4bit = _copy_t_prefix(
        packed.V_4bit,
        shape=(1, h_kv, v4_t, head_dim // 2),
        dtype=torch.uint8,
        device=device,
    )
    V_8bit = _copy_t_prefix(
        packed.V_8bit,
        shape=(1, h_kv, v8_t, head_dim),
        dtype=torch.uint8,
        device=device,
    )
    V_scale = _copy_t_prefix(
        packed.V_scale,
        shape=(1, h_kv, target_t_v, 1),
        dtype=dtype,
        device=device,
    )
    V_zp = _copy_t_prefix(
        packed.V_zp,
        shape=(1, h_kv, target_t_v, 1),
        dtype=dtype,
        device=device,
    )

    # K byte tensors: remap [compact V rows | v16 rows] into the fixed
    # [target_t_v compact rows | target_n_v16 stripe] canvas.
    src_n2, src_n4, src_n8 = packed.K_ch_seg_bounds
    src_n2p = _seg_padded_len(src_n2, 4)
    src_n4p = _seg_padded_len(src_n4, 2)
    src_sub2 = src_n2p // 4
    src_sub4 = src_n4p // 2
    tgt_n2, tgt_n4, tgt_n8 = target_k_bounds
    tgt_n2p = _seg_padded_len(tgt_n2, 4)
    tgt_n4p = _seg_padded_len(tgt_n4, 2)
    tgt_sub2 = tgt_n2p // 4
    tgt_sub4 = tgt_n4p // 2

    def _remap_k(src: Optional[torch.Tensor], target_sub: int) -> torch.Tensor:
        dst = torch.zeros(
            1, h_kv, target_t_k, target_sub, dtype=torch.uint8, device=device
        )
        if src is None:
            return dst
        copy_sub = min(int(src.shape[3]), target_sub)
        copy_tv = min(src_t_v, int(src.shape[2]), target_t_v)
        if copy_tv > 0 and copy_sub > 0:
            dst[:, :, :copy_tv, :copy_sub] = src[:, :, :copy_tv, :copy_sub]
        copy_v16 = min(src_n_v16, max(0, int(src.shape[2]) - src_t_v), target_n_v16)
        if copy_v16 > 0 and copy_sub > 0:
            dst[:, :, target_t_v:target_t_v + copy_v16, :copy_sub] = (
                src[:, :, src_t_v:src_t_v + copy_v16, :copy_sub]
            )
        return dst

    K_2bit = _remap_k(packed.K_2bit, tgt_sub2)
    K_4bit = _remap_k(packed.K_4bit, tgt_sub4)
    K_8bit = _remap_k(packed.K_8bit, tgt_n8)

    # K scale/zp and Q permutation are laid out as [2-pad | 4-pad | 8].
    target_d_padded = tgt_n2p + tgt_n4p + tgt_n8
    K_ch_scale = torch.zeros(1, h_kv, 1, target_d_padded, dtype=dtype, device=device)
    K_ch_zp = torch.zeros_like(K_ch_scale)

    def _copy_k_meta(field: torch.Tensor, dst: torch.Tensor) -> None:
        src_off = 0
        dst_off = 0
        width = min(src_n2p, tgt_n2p)
        if width > 0:
            dst[..., dst_off:dst_off + width] = field[..., src_off:src_off + width]
        src_off += src_n2p
        dst_off += tgt_n2p
        width = min(src_n4p, tgt_n4p)
        if width > 0:
            dst[..., dst_off:dst_off + width] = field[..., src_off:src_off + width]
        src_off += src_n4p
        dst_off += tgt_n4p
        width = min(src_n8, tgt_n8)
        if width > 0:
            dst[..., dst_off:dst_off + width] = field[..., src_off:src_off + width]

    _copy_k_meta(packed.K_ch_scale, K_ch_scale)
    _copy_k_meta(packed.K_ch_zp, K_ch_zp)

    def _pad_perm(src: Optional[torch.Tensor], rows: int) -> torch.Tensor:
        dst = torch.zeros(rows, target_d_padded, dtype=torch.long, device=device)
        if src is None:
            return dst
        src_off = 0
        dst_off = 0
        width = min(src_n2p, tgt_n2p)
        if width > 0:
            dst[:, dst_off:dst_off + width] = src[:, src_off:src_off + width]
        src_off += src_n2p
        dst_off += tgt_n2p
        width = min(src_n4p, tgt_n4p)
        if width > 0:
            dst[:, dst_off:dst_off + width] = src[:, src_off:src_off + width]
        src_off += src_n4p
        dst_off += tgt_n4p
        width = min(src_n8, tgt_n8)
        if width > 0:
            dst[:, dst_off:dst_off + width] = src[:, src_off:src_off + width]
        return dst

    K_ch_perm_padded_idx_per_head = _pad_perm(
        packed.K_ch_perm_padded_idx_per_head, h_kv
    )
    K_ch_perm_padded_idx_for_Q = _pad_perm(
        packed.K_ch_perm_padded_idx_for_Q, h_q
    )
    K_ch_perm_padded_idx = K_ch_perm_padded_idx_per_head[0]

    target_d_kept = sum(target_k_bounds)
    K_ch_sort_idx_per_head = torch.zeros(
        h_kv, target_d_kept, dtype=torch.long, device=device
    )
    if packed.K_ch_sort_idx_per_head is not None:
        width = min(int(packed.K_ch_sort_idx_per_head.shape[1]), target_d_kept)
        if width > 0:
            K_ch_sort_idx_per_head[:, :width] = packed.K_ch_sort_idx_per_head[:, :width]
    K_ch_sort_idx = K_ch_sort_idx_per_head[0]
    sort_idx = torch.zeros(target_t_v, dtype=torch.long, device=device)
    if packed.sort_idx is not None:
        width = min(int(packed.sort_idx.shape[0]), target_t_v)
        if width > 0:
            sort_idx[:width] = packed.sort_idx[:width]

    seg_bounds_per_head = packed.seg_bounds_per_head
    if seg_bounds_per_head is None:
        seg_bounds_per_head = torch.zeros(h_kv, 3, dtype=torch.int32, device=device)
    K_ch_seg_bounds_per_head = packed.K_ch_seg_bounds_per_head
    if K_ch_seg_bounds_per_head is None:
        K_ch_seg_bounds_per_head = torch.zeros(h_kv, 3, dtype=torch.int32, device=device)

    T_eff_per_head = packed.T_eff_per_head
    if T_eff_per_head is None:
        T_eff_per_head = torch.full((h_kv,), src_t_v, dtype=torch.int32, device=device)
    n_v16_per_head = packed.n_v16_per_head
    if n_v16_per_head is None:
        n_v16_per_head = torch.zeros(h_kv, dtype=torch.int32, device=device)
    T_eff_k_per_head = packed.T_eff_k_per_head
    if T_eff_k_per_head is None:
        T_eff_k_per_head = T_eff_per_head + n_v16_per_head

    if os.environ.get("OBKV_BLOCKWISE_LEGACY_PATH", "0") == "1":
        mask_kv = torch.full(
            (h_kv, target_t_k),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        for h in range(h_kv):
            tv = int(T_eff_per_head[h].item())
            nv = int(n_v16_per_head[h].item())
            if tv > 0:
                mask_kv[h, :min(tv, target_t_v)] = 0.0
            if nv > 0:
                mask_kv[
                    h,
                    target_t_v:target_t_v + min(nv, target_n_v16),
                ] = 0.0
    else:
        positions = torch.arange(
            target_t_k, device=device
        ).view(1, target_t_k)
        valid_v = (
            positions
            < T_eff_per_head.clamp(max=target_t_v).view(h_kv, 1)
        )
        valid_v16 = (
            (positions >= target_t_v)
            & (
                positions
                < (
                    target_t_v
                    + n_v16_per_head.clamp(
                        max=target_n_v16
                    ).view(h_kv, 1)
                )
            )
        )
        valid_kv = valid_v | valid_v16
        mask_kv = torch.where(
            valid_kv,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full(
                (), float("-inf"), dtype=torch.float32, device=device
            ),
        )
    softmax_mask = mask_kv.repeat_interleave(gqa_factor, dim=0).contiguous()

    new_v_slot = torch.zeros(
        1, h_kv, target_n_v16, head_dim, dtype=torch.float16, device=device
    )
    if new_v_only is not None:
        copy_n = min(int(new_v_only.shape[2]), target_n_v16)
        if copy_n > 0:
            new_v_slot[:, :, :copy_n, :] = new_v_only[:, :, :copy_n, :]

    padded = PackedKVLayer(
        K_2bit=K_2bit,
        K_4bit=K_4bit,
        K_8bit=K_8bit,
        K_ch_scale=K_ch_scale,
        K_ch_zp=K_ch_zp,
        K_ch_sort_idx=K_ch_sort_idx,
        K_ch_seg_bounds=target_k_bounds,
        K_ch_perm_padded_idx=K_ch_perm_padded_idx,
        V_2bit=V_2bit,
        V_4bit=V_4bit,
        V_8bit=V_8bit,
        V_scale=V_scale,
        V_zp=V_zp,
        sort_idx=sort_idx,
        seg_bounds=target_v_bounds,
        T_eff=target_t_v,
        seg_bounds_per_head=seg_bounds_per_head,
        T_eff_per_head=T_eff_per_head,
        softmax_mask=softmax_mask,
        n_fp16_per_head=None,
        fp16_gap_mask=None,
        K_ch_sort_idx_per_head=K_ch_sort_idx_per_head,
        K_ch_perm_padded_idx_per_head=K_ch_perm_padded_idx_per_head,
        K_ch_seg_bounds_per_head=K_ch_seg_bounds_per_head,
        K_ch_perm_padded_idx_for_Q=K_ch_perm_padded_idx_for_Q,
        T_eff_k=target_t_k,
        n_v16=target_n_v16,
        T_eff_k_per_head=T_eff_k_per_head,
        n_v16_per_head=n_v16_per_head,
        allocation_v_bits=packed.allocation_v_bits,
        allocation_k_bits=packed.allocation_k_bits,
    )
    return padded, new_v_slot


def _logical_layer_bits(
    packed: PackedKVLayer, head_dim: int
) -> int:
    return int(_logical_layer_bits_tensor(packed, head_dim).item())


def _logical_layer_bits_tensor(
    packed: PackedKVLayer, head_dim: int
) -> torch.Tensor:
    """Return logical bits without forcing a device-to-host synchronization."""
    v_bits = packed.allocation_v_bits
    k_bits = packed.allocation_k_bits
    if v_bits is None or k_bits is None:
        raise RuntimeError("packed block is missing allocator decisions")
    kept_per_head = (v_bits > 0).sum(dim=1).to(torch.int64)
    v_total = v_bits.to(torch.int64).sum() * int(head_dim)
    k_total = (
        k_bits.to(torch.int64).sum(dim=1) * kept_per_head
    ).sum()
    return v_total + k_total


def _validate_packed_layer_reference(
    packed: PackedKVLayer,
    squeezed: PackedKVLayer,
    new_v_only: Optional[torch.Tensor],
    query: torch.Tensor,
    *,
    num_kv_groups: int,
    head_dim: int,
) -> None:
    """Compare packed Triton K/V dispatches with explicit dequantization."""
    t_eff_k = (
        squeezed.T_eff_k if squeezed.T_eff_k > 0 else squeezed.T_eff
    )
    if t_eff_k == 0:
        return
    q = query.float()
    actual_k = k_mixed_qk_dot(q, squeezed, t_eff_k)
    unpacked_k = unpack_k_mixed(packed, D=head_dim)[0].float()
    expected_k = torch.einsum(
        "hd,htd->ht",
        q,
        unpacked_k.repeat_interleave(num_kv_groups, dim=0),
    ) / math.sqrt(head_dim)
    torch.testing.assert_close(
        actual_k,
        expected_k,
        rtol=2e-2,
        atol=2e-2,
        msg="packed K block diverges from explicit dequantization",
    )

    v_zone_a = _unpack_v_perhead(packed, head_dim=head_dim)[0].float()
    if new_v_only is not None:
        unpacked_v = torch.cat(
            [v_zone_a, new_v_only[0].float()], dim=1
        )
    else:
        unpacked_v = v_zone_a
    logits = torch.zeros(
        q.shape[0], t_eff_k, dtype=torch.float32, device=q.device
    )
    if squeezed.softmax_mask is not None:
        logits.add_(squeezed.softmax_mask)
    weights = F.softmax(logits, dim=-1)
    dummy_v, zero_counts = _get_v16_dummy(q.device, unpacked_v.shape[0])
    actual_v = v_dequant_weighted_sum_dispatch(
        weights,
        squeezed,
        V16=(
            new_v_only[0].contiguous()
            if new_v_only is not None
            else dummy_v
        ),
        n_v16_per_head=(
            squeezed.n_v16_per_head
            if squeezed.n_v16_per_head is not None
            else zero_counts
        ),
        max_T_v=squeezed.T_eff,
    )
    expected_v = (
        weights.view(
            unpacked_v.shape[0], num_kv_groups, t_eff_k
        )
        @ unpacked_v
    ).reshape(q.shape[0], head_dim)
    torch.testing.assert_close(
        actual_v,
        expected_v,
        rtol=3e-2,
        atol=3e-2,
        msg="packed V block diverges from explicit dequantization",
    )


def _score_block(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    num_kv_groups: int,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return V token scores and ThinK K-channel scores for one layer."""
    # queries [H_q,Q,D], keys [1,H_kv,T,D]
    h_kv = int(keys.shape[1])
    q_len = int(queries.shape[1])
    t_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    q_grouped = queries.float().view(
        h_kv, num_kv_groups, q_len, head_dim
    )
    k = keys[0].float()
    logits = torch.einsum("hgqd,htd->hgqt", q_grouped, k)
    logits.mul_(1.0 / math.sqrt(head_dim))
    if causal:
        if q_len != t_len:
            raise ValueError(
                "self-scored final flush requires equal query/key lengths, "
                f"got Q={q_len}, T={t_len}"
            )
        invalid = torch.arange(t_len, device=keys.device).view(1, t_len)
        invalid = invalid > torch.arange(q_len, device=keys.device).view(
            q_len, 1
        )
        logits.masked_fill_(invalid.view(1, 1, q_len, t_len), float("-inf"))
    attention = F.softmax(logits, dim=-1)
    token_scores = attention.sum(dim=(1, 2)).float()  # [H_kv,T]
    q_stat = q_grouped.square().mean(dim=(1, 2))
    k_stat = k.square().mean(dim=1)
    channel_scores = (q_stat * k_stat).float()  # [H_kv,D]
    return token_scores, channel_scores


def _channel_score(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    num_kv_groups: int,
) -> torch.Tensor:
    h_kv = int(keys.shape[1])
    q_len = int(queries.shape[1])
    head_dim = int(keys.shape[3])
    q_grouped = queries.float().view(
        h_kv, num_kv_groups, q_len, head_dim
    )
    return (
        q_grouped.square().mean(dim=(1, 2))
        * keys[0].float().square().mean(dim=1)
    ).float()


def _compress_block(
    *,
    target_kv: List[torch.Tensor],
    query_block: List[torch.Tensor],
    token_scores_by_layer: List[torch.Tensor],
    block_id: int,
    token_start: int,
    token_end: int,
    query_source_block: int,
    is_final_flush: bool,
    causal: bool,
    config: BlockwiseRDKVConfig,
    num_kv_groups: int,
    model_dtype: torch.dtype,
) -> Tuple[CompressedDecodeBlock, Dict[str, Any]]:
    # Imported lazily to avoid an import cycle: obkv_fast imports the decoder.
    from obkv_fast import _perhead_select_and_pack

    phase_timing_enabled = (
        os.environ.get("OBKV_BLOCKWISE_PHASE_TIMING", "0") == "1"
    )
    legacy_path_enabled = (
        os.environ.get("OBKV_BLOCKWISE_LEGACY_PATH", "0") == "1"
    )
    event_start = _sync_stamp(target_kv[0].device)
    phase: Dict[str, float] = {}
    packed_layers: List[PackedKVLayer] = []
    v16_layers: List[Optional[torch.Tensor]] = []
    logical_bit_tensors: List[torch.Tensor] = []
    v_allocation_tensors: List[torch.Tensor] = []
    k_allocation_tensors: List[torch.Tensor] = []
    logical_bits: List[int] = []
    equivalents: List[float] = []
    v_hist: Dict[str, int] = {}
    k_hist: Dict[str, int] = {}

    head_dim = int(target_kv[0].shape[-1] // 2)
    h_kv = int(target_kv[0].shape[1])
    logical_fp16_denominator = 2 * h_kv * head_dim * 16

    for layer_idx, (kv, queries, token_scores) in enumerate(
        zip(target_kv, query_block, token_scores_by_layer)
    ):
        layer_device = kv.device
        keys = kv[..., :head_dim].to(dtype=model_dtype)
        values = kv[..., head_dim:].to(dtype=model_dtype)

        if phase_timing_enabled:
            score_t0 = _sync_stamp(layer_device)
        channel_scores = _channel_score(
            queries,
            keys,
            num_kv_groups=num_kv_groups,
        )
        if phase_timing_enabled:
            score_t1 = _sync_stamp(layer_device)
            phase["score_ms"] = phase.get("score_ms", 0.0) + (
                score_t1 - score_t0
            ) * 1000.0

        layer_timing: Optional[Dict[str, float]] = (
            {} if phase_timing_enabled else None
        )
        packed, new_v_only, _ = _perhead_select_and_pack(
            keys,
            values,
            token_scores,
            channel_scores,
            gqa_factor=num_kv_groups,
            eviction_mode="joint",
            token_budget=config.block_budget_tokens,
            k_budget_ratio=config.k_budget_ratio,
            obs_window=config.obs_window,
            pool_kernel_size=config.pool_kernel_size,
            pool_padding=config.pool_padding,
            pool_type="avg",
            v_bit_options=config.v_bit_options,
            k_bit_options=config.k_bit_options,
            eps_V=config.epsilon_v,
            eps_K=config.epsilon_k,
            timing=layer_timing,
            retain_allocation_metadata=True,
        )
        if layer_timing is not None:
            for key, value in layer_timing.items():
                phase[key] = phase.get(key, 0.0) + float(value)

        if legacy_path_enabled:
            layer_bits = _logical_layer_bits(packed, head_dim)
            logical_bits.append(layer_bits)
            equivalents.append(layer_bits / logical_fp16_denominator)
            _merge_histogram(
                v_hist, _bit_histogram(packed.allocation_v_bits)
            )
            _merge_histogram(
                k_hist, _bit_histogram(packed.allocation_k_bits)
            )
        else:
            logical_bit_tensors.append(
                _logical_layer_bits_tensor(packed, head_dim)
            )
            if (
                packed.allocation_v_bits is None
                or packed.allocation_k_bits is None
            ):
                raise RuntimeError(
                    "packed block is missing allocator decisions"
                )
            v_allocation_tensors.append(packed.allocation_v_bits)
            k_allocation_tensors.append(packed.allocation_k_bits)
        packed_layers.append(packed)
        v16_layers.append(new_v_only)

        if config.correctness_check:
            tensors = [
                value
                for value in (
                    packed.K_ch_scale,
                    packed.K_ch_zp,
                    packed.V_scale,
                    packed.V_zp,
                    new_v_only,
                )
                if value is not None and value.is_floating_point()
            ]
            if any(not torch.isfinite(value).all() for value in tensors):
                raise FloatingPointError(
                    f"non-finite packed metadata in layer {layer_idx}"
                )

    if _fixed_slot_enabled():
        padded_layers: List[PackedKVLayer] = []
        padded_v16: List[torch.Tensor] = []
        for packed_layer, v16 in zip(packed_layers, v16_layers):
            padded_layer, padded_v = _pad_block_layer_to_fixed_slot(
                packed_layer,
                v16,
                h_q=h_kv * num_kv_groups,
                gqa_factor=num_kv_groups,
                head_dim=head_dim,
                block_size=config.block_size,
                block_budget_tokens=config.block_budget_tokens,
            )
            padded_layers.append(padded_layer)
            padded_v16.append(padded_v)
        packed_layers = padded_layers
        v16_layers = padded_v16

    if phase_timing_enabled:
        metadata_t0 = _sync_stamp(target_kv[0].device)
    packed_tuple = tuple(packed_layers)
    squeezed = _pre_squeeze_packed(packed_tuple)
    if config.correctness_check:
        for layer_idx, (packed, packed_squeezed, v16, queries) in enumerate(
            zip(packed_layers, squeezed, v16_layers, query_block)
        ):
            try:
                _validate_packed_layer_reference(
                    packed,
                    packed_squeezed,
                    v16,
                    queries[:, -1, :],
                    num_kv_groups=num_kv_groups,
                    head_dim=head_dim,
                )
            except Exception as error:
                raise AssertionError(
                    f"packed reference check failed at layer {layer_idx}"
                ) from error

    physical_bytes = _tensor_storage_bytes((packed_tuple, v16_layers))
    if legacy_path_enabled:
        event_end = _sync_stamp(target_kv[0].device)
        if phase_timing_enabled:
            phase["metadata_update_ms"] = (
                event_end - metadata_t0
            ) * 1000.0
    else:
        logical_bits_device = torch.stack(logical_bit_tensors)
        bit_values = (0, 2, 4, 8, 16)
        all_v_bits = torch.cat(
            [bits.reshape(-1) for bits in v_allocation_tensors]
        )
        all_k_bits = torch.cat(
            [bits.reshape(-1) for bits in k_allocation_tensors]
        )
        v_hist_device = torch.stack(
            [(all_v_bits == bit).sum() for bit in bit_values]
        )
        k_hist_device = torch.stack(
            [(all_k_bits == bit).sum() for bit in bit_values]
        )

        event_end = _sync_stamp(target_kv[0].device)
        if phase_timing_enabled:
            phase["metadata_update_ms"] = (
                event_end - metadata_t0
            ) * 1000.0

        logical_bits = [
            int(value)
            for value in logical_bits_device.detach().cpu().tolist()
        ]
        equivalents = [
            value / logical_fp16_denominator for value in logical_bits
        ]
        v_hist = {
            str(bit): int(count)
            for bit, count in zip(
                bit_values, v_hist_device.detach().cpu().tolist()
            )
        }
        k_hist = {
            str(bit): int(count)
            for bit, count in zip(
                bit_values, k_hist_device.detach().cpu().tolist()
            )
        }

    block = CompressedDecodeBlock(
        block_id=block_id,
        token_start=token_start,
        token_end=token_end,
        packed=packed_tuple,
        squeezed=squeezed,
        new_v_only=v16_layers,
        logical_bits_per_layer=logical_bits,
        fp16_equivalent_tokens_per_layer=equivalents,
        physical_bytes=physical_bytes,
        v_bit_histogram=v_hist,
        k_bit_histogram=k_hist,
    )
    allocation_ms = phase.get("v_allocation_ms", 0.0) + phase.get(
        "k_allocation_ms", 0.0
    )
    event = {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "query_source_block": query_source_block,
        "is_final_flush": is_final_flush,
        "query_mode": "self_causal" if causal else "next_block",
        "target_budget_fp16_tokens": config.block_budget_tokens,
        "logical_allocated_bits": int(sum(logical_bits)),
        "logical_allocated_bits_per_layer": logical_bits,
        "fp16_equivalent_tokens_per_layer": equivalents,
        "mean_fp16_equivalent_tokens_per_layer": (
            sum(equivalents) / max(1, len(equivalents))
        ),
        "physical_bytes": physical_bytes,
        "v_bit_histogram": v_hist,
        "k_bit_histogram": k_hist,
        "slot_shape_signature": [
            {
                "layer": layer_idx,
                "T_eff": int(packed.T_eff),
                "T_eff_k": int(packed.T_eff_k),
                "seg_bounds": list(map(int, packed.seg_bounds)),
                "k_ch_seg_bounds": list(map(int, packed.K_ch_seg_bounds)),
                "has_v2": packed.V_2bit is not None,
                "has_v4": packed.V_4bit is not None,
                "has_v8": packed.V_8bit is not None,
                "has_k2": packed.K_2bit is not None,
                "has_k4": packed.K_4bit is not None,
                "has_k8": packed.K_8bit is not None,
                "v2_shape": (
                    list(packed.V_2bit.shape)
                    if packed.V_2bit is not None else None
                ),
                "v4_shape": (
                    list(packed.V_4bit.shape)
                    if packed.V_4bit is not None else None
                ),
                "v8_shape": (
                    list(packed.V_8bit.shape)
                    if packed.V_8bit is not None else None
                ),
                "k2_shape": (
                    list(packed.K_2bit.shape)
                    if packed.K_2bit is not None else None
                ),
                "k4_shape": (
                    list(packed.K_4bit.shape)
                    if packed.K_4bit is not None else None
                ),
                "k8_shape": (
                    list(packed.K_8bit.shape)
                    if packed.K_8bit is not None else None
                ),
                "new_v_only_shape": (
                    list(v16_layers[layer_idx].shape)
                    if v16_layers[layer_idx] is not None else None
                ),
            }
            for layer_idx, packed in enumerate(packed_layers)
        ],
        "phase_timing_enabled": phase_timing_enabled,
        "legacy_path_enabled": legacy_path_enabled,
        "score_time_ms": phase.get("score_ms", 0.0),
        "pooling_time_ms": phase.get("pooling_ms", 0.0),
        "v_allocation_time_ms": phase.get("v_allocation_ms", 0.0),
        "k_allocation_time_ms": phase.get("k_allocation_ms", 0.0),
        "allocation_time_ms": allocation_ms,
        "quantization_time_ms": phase.get("quantization_ms", 0.0),
        "pack_time_ms": phase.get("packing_ms", 0.0),
        "metadata_update_time_ms": phase.get("metadata_update_ms", 0.0),
        "total_event_time_ms": (event_end - event_start) * 1000.0,
    }
    print(f"[BLOCKWISE_EVENT] {json.dumps(event, sort_keys=True)}", flush=True)
    return block, event


def greedy_decode_blockwise(
    *,
    model,
    dual_cache: TriZoneCache,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: int,
    config: BlockwiseRDKVConfig,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy decode with packed immutable decode-history blocks.

    CUDA Graph replay requires the fixed-slot bank so that every captured
    tensor has a stable shape and address while blocks transition from
    inactive (all-masked) to active in place.
    """
    config.validate()
    if primary_device.type != "cuda":
        raise RuntimeError("blockwise packed decode requires CUDA")

    eos_set = {int(token) for token in eos_token_ids if token is not None}
    generated_ids: List[int] = []
    weights = pre_extract_weights(model)
    if not weights.model_single_device:
        raise NotImplementedError(
            "blockwise RDKV currently requires --device-map single"
        )

    model_dtype = next(model.parameters()).dtype
    h_q = weights.num_heads
    h_kv = weights.num_kv_heads
    groups = weights.num_kv_groups
    head_dim = weights.head_dim
    inv_sqrt_d = 1.0 / math.sqrt(head_dim)
    prompt_squeezed = _pre_squeeze_packed(dual_cache.packed)
    prompt_bytes = cache_physical_bytes(dual_cache)
    use_slot_bank = _fixed_slot_enabled() and _slot_bank_enabled()
    if use_cuda_graph and not use_slot_bank:
        raise ValueError(
            "blockwise CUDA Graph requires "
            "OBKV_BLOCKWISE_FIXED_SLOT=1 and OBKV_BLOCKWISE_SLOT_BANK=1"
        )
    max_slot_blocks = (max_new_tokens + config.block_size - 1) // config.block_size
    slot_bank_layers: List[Tuple[PackedKVLayer, ...]] = []
    slot_bank_squeezed: List[List[PackedKVLayer]] = []
    slot_bank_v16: List[List[torch.Tensor]] = []
    slot_bank_physical_bytes = 0
    capacity = 2 * config.block_size
    runtime_key = (
        max_new_tokens,
        config.block_size,
        config.block_budget_tokens,
        max_slot_blocks,
        weights.num_layers,
        h_q,
        h_kv,
        head_dim,
        str(model_dtype),
        str(primary_device),
        use_cuda_graph,
        tuple(
            (
                packed.T_eff,
                packed.T_eff_k,
                int(packed.softmax_mask.shape[1])
                if packed.softmax_mask is not None
                else 0,
            )
            for packed in prompt_squeezed
        ),
    )
    runtime = (
        getattr(dual_cache, "_blockwise_runtime_cache", None)
        if use_slot_bank
        else None
    )
    if runtime is not None and runtime.get("key") != runtime_key:
        runtime = None
        try:
            del dual_cache._blockwise_runtime_cache
        except AttributeError:
            pass

    if runtime is not None:
        slot_bank_layers = runtime["slot_bank_layers"]
        slot_bank_squeezed = runtime["slot_bank_squeezed"]
        slot_bank_v16 = runtime["slot_bank_v16"]
        slot_bank_physical_bytes = runtime["slot_bank_physical_bytes"]
        pending_kv = runtime["pending_kv"]
        pending_q = runtime["pending_q"]
        self_scores = runtime["self_scores"]
        next_scores = runtime["next_scores"]
        # Quantized payloads may remain stale between benchmark repeats, but
        # inactive slots are made semantically empty by resetting their masks.
        # A slot activation overwrites every payload/metadata tensor in place.
        for slot_layers in slot_bank_layers:
            for packed in slot_layers:
                if packed.softmax_mask is not None:
                    packed.softmax_mask.fill_(float("-inf"))
        for self_score, next_score in zip(self_scores, next_scores):
            self_score.zero_()
            next_score.zero_()
    elif use_slot_bank:
        for _slot in range(max_slot_blocks):
            layers: List[PackedKVLayer] = []
            v16_layers: List[torch.Tensor] = []
            for layer in weights.layers:
                empty_layer, empty_v16 = _make_empty_fixed_slot_layer(
                    h_kv=h_kv,
                    h_q=h_q,
                    gqa_factor=groups,
                    head_dim=head_dim,
                    block_size=config.block_size,
                    block_budget_tokens=config.block_budget_tokens,
                    dtype=model_dtype,
                    device=layer.device,
                )
                layers.append(empty_layer)
                v16_layers.append(empty_v16)
            layer_tuple = tuple(layers)
            slot_bank_layers.append(layer_tuple)
            slot_bank_squeezed.append(_pre_squeeze_packed(layer_tuple))
            slot_bank_v16.append(v16_layers)
        slot_bank_physical_bytes = _tensor_storage_bytes((slot_bank_layers, slot_bank_v16))
        pending_kv = [
            torch.empty(
                1,
                h_kv,
                capacity,
                2 * head_dim,
                dtype=torch.float32 if use_cuda_graph else model_dtype,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        pending_q = [
            torch.empty(
                h_q,
                capacity,
                head_dim,
                dtype=model_dtype,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        # Exact V-side attention mass captured from the normal global softmax.
        # self_scores covers causal queries from the same pending block;
        # next_scores covers successor-block queries attending to the preceding
        # block.
        self_scores = [
            torch.zeros(
                h_kv,
                capacity,
                dtype=torch.float32,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        next_scores = [
            torch.zeros(
                h_kv,
                config.block_size,
                dtype=torch.float32,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        runtime = {
            "key": runtime_key,
            "slot_bank_layers": slot_bank_layers,
            "slot_bank_squeezed": slot_bank_squeezed,
            "slot_bank_v16": slot_bank_v16,
            "slot_bank_physical_bytes": slot_bank_physical_bytes,
            "pending_kv": pending_kv,
            "pending_q": pending_q,
            "self_scores": self_scores,
            "next_scores": next_scores,
        }
        dual_cache._blockwise_runtime_cache = runtime
    else:
        pending_kv = [
            torch.empty(
                1,
                h_kv,
                capacity,
                2 * head_dim,
                dtype=model_dtype,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        pending_q = [
            torch.empty(
                h_q,
                capacity,
                head_dim,
                dtype=model_dtype,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        self_scores = [
            torch.zeros(
                h_kv,
                capacity,
                dtype=torch.float32,
                device=layer.device,
            )
            for layer in weights.layers
        ]
        next_scores = [
            torch.zeros(
                h_kv,
                config.block_size,
                dtype=torch.float32,
                device=layer.device,
            )
            for layer in weights.layers
        ]
    pending_len = 0
    pending_start = 0
    compressed: List[CompressedDecodeBlock] = []
    block_events: List[Dict[str, Any]] = []
    memory_trace: List[Dict[str, Any]] = []
    token_cuda_events: List[torch.cuda.Event] = []
    dummy_by_device: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    def dummy_for(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        value = dummy_by_device.get(device)
        if value is None:
            value = _get_v16_dummy(device, h_kv)
            dummy_by_device[device] = value
        return value

    # CUDA Graph state is cached on the prompt cache together with the fixed
    # slot bank.  Captured graphs therefore keep both tensor shapes and storage
    # addresses stable across block activations and benchmark repeats.
    graph_state: Optional[Dict[str, Any]] = None
    if use_cuda_graph:
        assert runtime is not None
        graph_state = runtime.get("cuda_graph")
        if graph_state is None:
            h_buf = torch.empty(
                1,
                1,
                weights.hidden_size,
                dtype=model_dtype,
                device=primary_device,
            )
            residual_buf = torch.empty_like(h_buf)
            cos_half_buf = torch.empty(
                head_dim // 2, dtype=torch.float32, device=primary_device
            )
            sin_half_buf = torch.empty_like(cos_half_buf)
            q_scaled_buf = torch.empty(
                h_q, head_dim, dtype=torch.float32, device=primary_device
            )
            q_store_buf = torch.empty(
                h_q, head_dim, dtype=model_dtype, device=primary_device
            )
            kv_store_buf = torch.empty(
                h_kv, 2 * head_dim, dtype=model_dtype, device=primary_device
            )
            pending_score_buf = torch.empty(
                h_q, capacity, dtype=torch.float32, device=primary_device
            )
            pending_mask = torch.full(
                (weights.num_layers, 1, capacity),
                float("-inf"),
                dtype=torch.float32,
                device=primary_device,
            )
            self_mask = torch.zeros(
                1, capacity, dtype=torch.float32, device=primary_device
            )
            next_active = torch.zeros(
                1, dtype=torch.float32, device=primary_device
            )
            source_specs: List[
                List[Tuple[PackedKVLayer, torch.Tensor, torch.Tensor, int]]
            ] = []
            total_old_per_layer: List[int] = []
            active_total_old_per_layer: List[List[int]] = []
            captured_active_slot_counts = list(range(max_slot_blocks))
            for layer_idx, layer in enumerate(weights.layers):
                layer_sources: List[
                    Tuple[PackedKVLayer, torch.Tensor, torch.Tensor, int]
                ] = []
                raw_sources: List[
                    Tuple[PackedKVLayer, Optional[torch.Tensor]]
                ] = [
                    (
                        prompt_squeezed[layer_idx],
                        dual_cache.new_v_only[layer_idx],
                    )
                ]
                raw_sources.extend(
                    (
                        slot_bank_squeezed[slot_idx][layer_idx],
                        slot_bank_v16[slot_idx][layer_idx],
                    )
                    for slot_idx in range(len(slot_bank_squeezed))
                )
                for packed, v16 in raw_sources:
                    source_len = (
                        packed.T_eff_k
                        if packed.T_eff_k > 0
                        else packed.T_eff
                    )
                    if (
                        packed.softmax_mask is not None
                        and packed.softmax_mask.numel()
                        and packed.softmax_mask.shape[1] != source_len
                    ):
                        raise AssertionError(
                            "blockwise graph mask width mismatch: "
                            f"layer={layer_idx} mask={packed.softmax_mask.shape[1]} "
                            f"source={source_len}"
                        )
                    dummy_v, zero_counts = dummy_for(layer.device)
                    v16_arg = (
                        v16[0].contiguous() if v16 is not None else dummy_v
                    )
                    counts = (
                        packed.n_v16_per_head
                        if packed.n_v16_per_head is not None
                        else zero_counts
                    )
                    layer_sources.append(
                        (packed, v16_arg, counts, source_len)
                    )
                source_specs.append(layer_sources)
                total_old_per_layer.append(
                    sum(source[3] for source in layer_sources)
                )
                active_total_old_per_layer.append(
                    [
                        sum(
                            source[3]
                            for source in layer_sources[:1 + active_slots]
                        )
                        for active_slots in captured_active_slot_counts
                    ]
                )

            old_score_bufs = [
                torch.empty(
                    h_q,
                    total_old,
                    dtype=torch.float32,
                    device=primary_device,
                )
                for total_old in total_old_per_layer
            ]
            all_score_bufs = [
                torch.empty(
                    h_q,
                    total_old + capacity,
                    dtype=torch.float32,
                    device=primary_device,
                )
                for total_old in total_old_per_layer
            ]
            out_accum_buf = torch.empty(
                h_q, head_dim, dtype=torch.float32, device=primary_device
            )
            out_source_buf = torch.empty_like(out_accum_buf)
            out_pending_buf = torch.empty_like(out_accum_buf)
            next_token_buf = torch.empty(
                1, 1, dtype=torch.long, device=primary_device
            )
            graphs_a: Dict[int, List[torch.cuda.CUDAGraph]] = {
                active_slots: []
                for active_slots in captured_active_slot_counts
            }
            graphs_b: Dict[int, List[torch.cuda.CUDAGraph]] = {
                active_slots: []
                for active_slots in captured_active_slot_counts
            }

            def graph_a_body(layer_idx: int, active_slots: int) -> None:
                layer = weights.layers[layer_idx]
                residual_buf.copy_(h_buf)
                layer_hidden = _rms_norm(
                    h_buf, layer.input_ln_weight, layer.input_ln_eps
                )
                if layer.qkv_weight is not None:
                    qkv = F.linear(
                        layer_hidden, layer.qkv_weight, layer.qkv_bias
                    )
                    q, k, v = qkv.split(layer.qkv_split_sizes, dim=-1)
                else:
                    q = F.linear(
                        layer_hidden,
                        layer.q_proj_weight,
                        layer.q_proj_bias,
                    )
                    k = F.linear(
                        layer_hidden,
                        layer.k_proj_weight,
                        layer.k_proj_bias,
                    )
                    v = F.linear(
                        layer_hidden,
                        layer.v_proj_weight,
                        layer.v_proj_bias,
                    )
                q_2d = q.view(h_q, head_dim)
                k_2d = k.view(h_kv, head_dim)
                v_2d = v.view(h_kv, head_dim)
                if layer.q_norm_weight is not None:
                    q_2d = _rms_norm(
                        q_2d, layer.q_norm_weight, layer.qk_norm_eps
                    )
                if layer.k_norm_weight is not None:
                    k_2d = _rms_norm(
                        k_2d, layer.k_norm_weight, layer.qk_norm_eps
                    )
                fused_rope(q_2d, k_2d, cos_half_buf, sin_half_buf)
                q_float = q_2d.float()
                offset = 0
                for packed, _, _, source_len in source_specs[layer_idx][
                    :1 + active_slots
                ]:
                    if source_len:
                        score_view = old_score_bufs[layer_idx][
                            :, offset:offset + source_len
                        ]
                        k_mixed_qk_dot_into(
                            q_float, packed, source_len, score_view
                        )
                        if packed.softmax_mask is not None:
                            score_view.add_(packed.softmax_mask)
                    offset += source_len
                q_scaled_buf.copy_(q_float * inv_sqrt_d)
                q_store_buf.copy_(q_2d)
                kv_store_buf[:, :head_dim].copy_(k_2d)
                kv_store_buf[:, head_dim:].copy_(v_2d)

            def graph_b_body(layer_idx: int, active_slots: int) -> None:
                layer = weights.layers[layer_idx]
                total_old = active_total_old_per_layer[layer_idx][
                    active_slots
                ]
                torch.bmm(
                    q_scaled_buf.view(h_kv, groups, head_dim),
                    pending_kv[layer_idx][
                        0, :, :, :head_dim
                    ].transpose(-1, -2),
                    out=pending_score_buf.view(h_kv, groups, capacity),
                )
                pending_score_buf.add_(pending_mask[layer_idx])
                all_scores = all_score_bufs[layer_idx][
                    :, :total_old + capacity
                ]
                if total_old:
                    all_scores[:, :total_old].copy_(
                        old_score_bufs[layer_idx][:, :total_old]
                    )
                all_scores[:, total_old:].copy_(pending_score_buf)
                all_weights = F.softmax(all_scores, dim=-1)

                pending_weights = all_weights[:, total_old:]
                pending_mass = pending_weights.view(
                    h_kv, groups, capacity
                ).sum(dim=1)
                self_scores[layer_idx].add_(
                    pending_mass * self_mask
                )
                next_scores[layer_idx].add_(
                    pending_mass[:, :config.block_size] * next_active
                )

                out_accum_buf.zero_()
                offset = 0
                for packed, v16_arg, counts, source_len in source_specs[
                    layer_idx
                ][:1 + active_slots]:
                    if source_len:
                        v_dequant_weighted_sum_dispatch_into(
                            all_weights[:, offset:offset + source_len],
                            packed,
                            out_source_buf,
                            V16=v16_arg,
                            n_v16_per_head=counts,
                            max_T_v=packed.T_eff,
                        )
                        out_accum_buf.add_(out_source_buf)
                    offset += source_len
                torch.bmm(
                    pending_weights.view(h_kv, groups, capacity),
                    pending_kv[layer_idx][0, :, :, head_dim:].float(),
                    out=out_pending_buf.view(h_kv, groups, head_dim),
                )
                out_accum_buf.add_(out_pending_buf)
                attention_output = out_accum_buf.to(model_dtype).view(
                    1, 1, weights.attn_output_dim
                )
                attention_output = F.linear(
                    attention_output, layer.o_proj_weight
                )
                layer_hidden = residual_buf + attention_output
                residual = layer_hidden.clone()
                layer_hidden = _rms_norm(
                    layer_hidden,
                    layer.post_ln_weight,
                    layer.post_ln_eps,
                )
                if layer.gate_up_weight is not None:
                    gate_up = F.linear(
                        layer_hidden, layer.gate_up_weight
                    )
                    gate, up = gate_up.split(
                        layer.gate_up_split_sizes, dim=-1
                    )
                else:
                    gate = F.linear(
                        layer_hidden, layer.gate_proj_weight
                    )
                    up = F.linear(layer_hidden, layer.up_proj_weight)
                layer_hidden = F.linear(
                    F.silu(gate) * up, layer.down_proj_weight
                )
                h_buf.copy_(residual + layer_hidden)

            # Pre-touch every exact packed-kernel signature and all cuBLAS
            # paths before graph capture.  The pending suffix is masked to one
            # zero-initialized token so warmup/capture always has a finite
            # softmax denominator.
            for buf in pending_kv:
                buf.zero_()
            pending_mask.fill_(float("-inf"))
            pending_mask[:, :, 0].zero_()
            self_mask.zero_()
            self_mask[:, 0].fill_(1.0)
            next_active.zero_()
            h_buf.zero_()
            cos_half_buf.zero_()
            sin_half_buf.zero_()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for active_slots in captured_active_slot_counts:
                    for layer_idx in range(weights.num_layers):
                        graph_a_body(layer_idx, active_slots)
                        graph_b_body(layer_idx, active_slots)
                for _ in range(2):
                    for layer_idx in range(weights.num_layers):
                        graph_a_body(layer_idx, 0)
                        graph_b_body(layer_idx, 0)
                    normed = _rms_norm(
                        h_buf,
                        weights.final_norm_weight,
                        weights.final_norm_eps,
                    )
                    logits = F.linear(normed, weights.lm_head_weight)
                    next_token_buf.copy_(
                        logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    )
            torch.cuda.current_stream().wait_stream(capture_stream)

            for active_slots in captured_active_slot_counts:
                for layer_idx in range(weights.num_layers):
                    graph_a = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph_a, stream=capture_stream):
                        graph_a_body(layer_idx, active_slots)
                    graphs_a[active_slots].append(graph_a)
                    graph_b = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph_b, stream=capture_stream):
                        graph_b_body(layer_idx, active_slots)
                    graphs_b[active_slots].append(graph_b)
            graph_c = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_c, stream=capture_stream):
                normed = _rms_norm(
                    h_buf,
                    weights.final_norm_weight,
                    weights.final_norm_eps,
                )
                logits = F.linear(normed, weights.lm_head_weight)
                next_token_buf.copy_(
                    logits[:, -1, :].argmax(dim=-1, keepdim=True)
                )
            torch.cuda.current_stream().wait_stream(capture_stream)
            for self_score, next_score in zip(self_scores, next_scores):
                self_score.zero_()
                next_score.zero_()
            graph_state = {
                "h_buf": h_buf,
                "cos_half_buf": cos_half_buf,
                "sin_half_buf": sin_half_buf,
                "q_store_buf": q_store_buf,
                "kv_store_buf": kv_store_buf,
                "pending_mask": pending_mask,
                "self_mask": self_mask,
                "next_active": next_active,
                "next_token_buf": next_token_buf,
                "graphs_a": graphs_a,
                "graphs_b": graphs_b,
                "graph_c": graph_c,
                "source_lengths": total_old_per_layer,
                "captured_active_slot_counts": captured_active_slot_counts,
            }
            runtime["cuda_graph"] = graph_state

    def compressed_equivalents() -> float:
        return sum(
            sum(block.fp16_equivalent_tokens_per_layer)
            / max(1, len(block.fp16_equivalent_tokens_per_layer))
            for block in compressed
        )

    decode_peak_equiv = 0.0
    decode_peak_physical = 0

    def record_memory(stage: str) -> None:
        nonlocal decode_peak_equiv, decode_peak_physical
        comp_eq = compressed_equivalents()
        logical_eq = comp_eq + pending_len
        pending_bytes = (
            pending_len
            * len(pending_kv)
            * h_kv
            * 2
            * head_dim
            * pending_kv[0].element_size()
        )
        physical = sum(block.physical_bytes for block in compressed) + pending_bytes
        if use_slot_bank:
            physical = slot_bank_physical_bytes + pending_bytes
        decode_peak_equiv = max(decode_peak_equiv, logical_eq)
        decode_peak_physical = max(decode_peak_physical, physical)
        memory_trace.append(
            {
                "stage": stage,
                "generated_tokens": len(generated_ids),
                "compressed_blocks": len(compressed),
                "pending_fp16_tokens": pending_len,
                "compressed_history_fp16_equivalent_tokens": comp_eq,
                "decode_cache_fp16_equivalent_tokens": logical_eq,
                "decode_cache_physical_bytes": physical,
            }
        )

    def validate_order() -> None:
        cursor = 0
        for block in compressed:
            if block.token_start != cursor or block.token_end < block.token_start:
                raise AssertionError(
                    "decode block order corruption: "
                    f"expected start {cursor}, got "
                    f"[{block.token_start},{block.token_end}]"
                )
            cursor = block.token_end + 1
        if cursor != pending_start:
            raise AssertionError(
                f"pending suffix starts at {pending_start}, expected {cursor}"
            )
        if pending_start + pending_len != len(generated_ids):
            raise AssertionError(
                "decode token coverage mismatch: compressed+pending does not "
                f"cover [0,{len(generated_ids)})"
            )

    def compress_prefix(
        target_len: int,
        *,
        query_offset: int,
        query_len: int,
        is_final_flush: bool,
        causal: bool,
    ) -> None:
        nonlocal pending_len, pending_start
        if target_len <= 0 or query_len <= 0:
            return
        target = [
            buf[:, :, :target_len, :].clone()
            for buf in pending_kv
        ]
        queries = [
            buf[:, query_offset:query_offset + query_len, :].clone()
            for buf in pending_q
        ]
        token_scores = [
            (
                self_score[:, :target_len]
                if causal
                else next_score[:, :target_len]
            ).clone()
            for self_score, next_score in zip(self_scores, next_scores)
        ]
        block_id = len(compressed)
        query_source = block_id if causal else block_id + 1
        block, event = _compress_block(
            target_kv=target,
            query_block=queries,
            token_scores_by_layer=token_scores,
            block_id=block_id,
            token_start=pending_start,
            token_end=pending_start + target_len - 1,
            query_source_block=query_source,
            is_final_flush=is_final_flush,
            causal=causal,
            config=config,
            num_kv_groups=groups,
            model_dtype=model_dtype,
        )
        compressed.append(block)
        if use_slot_bank:
            if block_id >= len(slot_bank_layers):
                raise RuntimeError(
                    f"blockwise slot bank overflow: block_id={block_id} "
                    f"slots={len(slot_bank_layers)}"
                )
            _copy_fixed_slot_block_into_bank(
                dst_layers=slot_bank_layers[block_id],
                dst_v16=slot_bank_v16[block_id],
                src_layers=block.packed,
                src_v16=block.new_v_only,
            )
        block_events.append(event)
        remain = pending_len - target_len
        for kv_buf, q_buf, self_score, next_score in zip(
            pending_kv, pending_q, self_scores, next_scores
        ):
            if remain:
                kv_buf[:, :, :remain, :].copy_(
                    kv_buf[:, :, target_len:pending_len, :].clone()
                )
                q_buf[:, :remain, :].copy_(
                    q_buf[:, target_len:pending_len, :].clone()
                )
                self_score[:, :remain].copy_(
                    self_score[:, target_len:pending_len].clone()
                )
            self_score[:, remain:].zero_()
            next_score.zero_()
        pending_start += target_len
        pending_len = remain
        validate_order()
        record_memory(
            "final_flush" if is_final_flush else f"compressed_block_{block_id}"
        )

    if next_token_logits.dim() == 1:
        next_token_logits = next_token_logits.unsqueeze(0)
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
    position_ids = torch.zeros(
        (1, 1), dtype=torch.long, device=primary_device
    )
    record_memory("decode_start")
    decode_start_wall = _sync_stamp(primary_device)
    decode_start_event = torch.cuda.Event(enable_timing=True)
    decode_start_event.record()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in eos_set:
                break
            generated_ids.append(token_id)
            position_ids.fill_(decode_position_start + step)
            hidden = weights.embed_tokens(next_token.to(weights.embed_dev))
            cos, sin = weights.rotary_emb(hidden, position_ids)
            cos_half = cos.view(head_dim)[: head_dim // 2]
            sin_half = sin.view(head_dim)[: head_dim // 2]

            write_pos = pending_len
            if write_pos >= capacity:
                raise RuntimeError(
                    f"pending decode cache overflow ({write_pos} >= {capacity})"
                )

            if use_cuda_graph:
                assert graph_state is not None
                active_slots = len(compressed)
                if active_slots not in graph_state[
                    "captured_active_slot_counts"
                ]:
                    raise RuntimeError(
                        "missing blockwise CUDA Graph variant for "
                        f"active_slots={active_slots}; captured="
                        f"{graph_state['captured_active_slot_counts']}"
                    )
                current_len = write_pos + 1
                query_block_start = (
                    write_pos // config.block_size
                ) * config.block_size
                graph_state["h_buf"].copy_(hidden)
                graph_state["cos_half_buf"].copy_(cos_half)
                graph_state["sin_half_buf"].copy_(sin_half)
                graph_state["pending_mask"].fill_(float("-inf"))
                graph_state["pending_mask"][:, :, :current_len].zero_()
                graph_state["self_mask"].zero_()
                graph_state["self_mask"][
                    :, query_block_start:current_len
                ].fill_(1.0)
                graph_state["next_active"].fill_(
                    1.0
                    if query_block_start >= config.block_size
                    else 0.0
                )
                for layer_idx in range(weights.num_layers):
                    graph_state["graphs_a"][active_slots][
                        layer_idx
                    ].replay()
                    pending_q[layer_idx][:, write_pos, :].copy_(
                        graph_state["q_store_buf"]
                    )
                    pending_kv[layer_idx][0, :, write_pos, :].copy_(
                        graph_state["kv_store_buf"]
                    )
                    graph_state["graphs_b"][active_slots][
                        layer_idx
                    ].replay()
                graph_state["graph_c"].replay()
                next_token = graph_state["next_token_buf"]
                pending_len += 1
                token_event = torch.cuda.Event(enable_timing=True)
                token_event.record()
                token_cuda_events.append(token_event)
                record_memory(f"token_{len(generated_ids)}")
                if pending_len == capacity:
                    compress_prefix(
                        config.block_size,
                        query_offset=config.block_size,
                        query_len=config.block_size,
                        is_final_flush=False,
                        causal=False,
                    )
                continue

            for layer_idx, layer in enumerate(weights.layers):
                residual = hidden
                hidden = _rms_norm(
                    hidden, layer.input_ln_weight, layer.input_ln_eps
                )
                if layer.qkv_weight is not None:
                    qkv = F.linear(hidden, layer.qkv_weight, layer.qkv_bias)
                    q, k, v = qkv.split(layer.qkv_split_sizes, dim=-1)
                else:
                    q = F.linear(
                        hidden, layer.q_proj_weight, layer.q_proj_bias
                    )
                    k = F.linear(
                        hidden, layer.k_proj_weight, layer.k_proj_bias
                    )
                    v = F.linear(
                        hidden, layer.v_proj_weight, layer.v_proj_bias
                    )
                q_2d = q.view(h_q, head_dim)
                k_2d = k.view(h_kv, head_dim)
                v_2d = v.view(h_kv, head_dim)
                if layer.q_norm_weight is not None:
                    q_2d = _rms_norm(
                        q_2d, layer.q_norm_weight, layer.qk_norm_eps
                    )
                if layer.k_norm_weight is not None:
                    k_2d = _rms_norm(
                        k_2d, layer.k_norm_weight, layer.qk_norm_eps
                    )
                fused_rope(q_2d, k_2d, cos_half, sin_half)
                pending_q[layer_idx][:, write_pos, :].copy_(q_2d)
                pending_kv[layer_idx][0, :, write_pos, :head_dim].copy_(k_2d)
                pending_kv[layer_idx][0, :, write_pos, head_dim:].copy_(v_2d)

                packed_sources: List[
                    Tuple[PackedKVLayer, Optional[torch.Tensor]]
                ] = [
                    (prompt_squeezed[layer_idx], dual_cache.new_v_only[layer_idx])
                ]
                if use_slot_bank:
                    packed_sources.extend(
                        (
                            slot_bank_squeezed[slot_idx][layer_idx],
                            slot_bank_v16[slot_idx][layer_idx],
                        )
                        for slot_idx in range(len(slot_bank_squeezed))
                    )
                else:
                    packed_sources.extend(
                        (
                            block.squeezed[layer_idx],
                            block.new_v_only[layer_idx],
                        )
                        for block in compressed
                    )
                score_parts: List[torch.Tensor] = []
                source_lengths: List[int] = []
                q_float = q_2d.float()
                for packed, _ in packed_sources:
                    source_len = (
                        packed.T_eff_k
                        if packed.T_eff_k > 0
                        else packed.T_eff
                    )
                    source_lengths.append(source_len)
                    if source_len:
                        score = k_mixed_qk_dot(
                            q_float, packed, source_len
                        )
                        if packed.softmax_mask is not None:
                            score.add_(packed.softmax_mask)
                        score_parts.append(score)
                    else:
                        score_parts.append(
                            torch.empty(
                                h_q,
                                0,
                                dtype=torch.float32,
                                device=layer.device,
                            )
                        )

                current_len = write_pos + 1
                k_pending = pending_kv[layer_idx][
                    0, :, :current_len, :head_dim
                ]
                q_grouped = q_2d.view(h_kv, groups, head_dim)
                pending_scores = torch.bmm(
                    q_grouped, k_pending.transpose(-1, -2)
                ).reshape(h_q, current_len).float()
                pending_scores.mul_(inv_sqrt_d)
                all_scores = torch.cat(
                    [*score_parts, pending_scores], dim=-1
                )
                if config.correctness_check and not torch.isfinite(
                    all_scores
                ).any(dim=-1).all():
                    raise FloatingPointError(
                        f"all attention positions masked at layer {layer_idx}"
                    )
                all_weights = F.softmax(all_scores, dim=-1)

                offset = 0
                out = torch.zeros(
                    h_q,
                    head_dim,
                    dtype=torch.float32,
                    device=layer.device,
                )
                for (packed, v16), source_len in zip(
                    packed_sources, source_lengths
                ):
                    if source_len:
                        dummy_v, zero_counts = dummy_for(layer.device)
                        v16_arg = (
                            v16[0].contiguous()
                            if v16 is not None
                            else dummy_v
                        )
                        counts = (
                            packed.n_v16_per_head
                            if packed.n_v16_per_head is not None
                            else zero_counts
                        )
                        out.add_(
                            v_dequant_weighted_sum_dispatch(
                                all_weights[:, offset:offset + source_len],
                                packed,
                                V16=v16_arg,
                                n_v16_per_head=counts,
                                max_T_v=packed.T_eff,
                            )
                        )
                    offset += source_len
                pending_weights = all_weights[:, offset:]
                query_block_start = (
                    write_pos // config.block_size
                ) * config.block_size
                self_width = current_len - query_block_start
                self_scores[layer_idx][
                    :, query_block_start:current_len
                ].add_(
                    pending_weights[
                        :, query_block_start:current_len
                    ].view(h_kv, groups, self_width).sum(dim=1)
                )
                if query_block_start >= config.block_size:
                    next_scores[layer_idx].add_(
                        pending_weights[:, :config.block_size]
                        .view(h_kv, groups, config.block_size)
                        .sum(dim=1)
                    )
                v_pending = pending_kv[layer_idx][
                    0, :, :current_len, head_dim:
                ].float()
                w_pending = pending_weights.view(
                    h_kv, groups, current_len
                )
                out.add_(
                    torch.bmm(w_pending, v_pending).reshape(h_q, head_dim)
                )
                if config.correctness_check and not torch.isfinite(out).all():
                    raise FloatingPointError(
                        f"non-finite attention output at layer {layer_idx}"
                    )

                attention_output = out.to(model_dtype).view(
                    1, 1, weights.attn_output_dim
                )
                attention_output = F.linear(
                    attention_output, layer.o_proj_weight
                )
                hidden = residual + attention_output
                residual = hidden
                hidden = _rms_norm(
                    hidden, layer.post_ln_weight, layer.post_ln_eps
                )
                if layer.gate_up_weight is not None:
                    gate_up = F.linear(hidden, layer.gate_up_weight)
                    gate, up = gate_up.split(
                        layer.gate_up_split_sizes, dim=-1
                    )
                else:
                    gate = F.linear(hidden, layer.gate_proj_weight)
                    up = F.linear(hidden, layer.up_proj_weight)
                hidden = F.linear(
                    F.silu(gate) * up, layer.down_proj_weight
                )
                hidden = residual + hidden

            hidden = _rms_norm(
                hidden, weights.final_norm_weight, weights.final_norm_eps
            )
            logits = F.linear(hidden, weights.lm_head_weight)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            pending_len += 1
            token_event = torch.cuda.Event(enable_timing=True)
            token_event.record()
            token_cuda_events.append(token_event)
            record_memory(f"token_{len(generated_ids)}")

            if pending_len == capacity:
                compress_prefix(
                    config.block_size,
                    query_offset=config.block_size,
                    query_len=config.block_size,
                    is_final_flush=False,
                    causal=False,
                )

    decode_pre_flush_wall_end = _sync_stamp(primary_device)
    # If generation stopped with a complete target plus an incomplete
    # successor, use all available successor queries to drain the target.
    while pending_len > config.block_size:
        successor_len = min(
            config.block_size, pending_len - config.block_size
        )
        compress_prefix(
            config.block_size,
            query_offset=config.block_size,
            query_len=successor_len,
            is_final_flush=True,
            causal=False,
        )
    final_flush_start = _sync_stamp(primary_device)
    if pending_len:
        compress_prefix(
            pending_len,
            query_offset=0,
            query_len=pending_len,
            is_final_flush=True,
            causal=True,
        )
    final_flush_end = _sync_stamp(primary_device)
    record_memory("decode_complete")
    validate_order()

    torch.cuda.synchronize(primary_device)
    token_timestamps_ms = [
        decode_start_event.elapsed_time(event)
        for event in token_cuda_events
    ]
    n_gen = len(generated_ids)
    decode_ms = (decode_pre_flush_wall_end - decode_start_wall) * 1000.0
    final_flush_ms = (final_flush_end - final_flush_start) * 1000.0
    final_equiv = compressed_equivalents()
    report: Dict[str, Any] = {
        "decode_blockwise_rdkv": True,
        "decode_cuda_graph": use_cuda_graph,
        "decode_cuda_graph_active_slot_variants": (
            graph_state["captured_active_slot_counts"]
            if graph_state is not None
            else []
        ),
        "decode_block_size": config.block_size,
        "decode_block_budget_tokens": config.block_budget_tokens,
        "generated_tokens": n_gen,
        "block_events": block_events,
        "memory_trace": memory_trace,
        "per_token_timestamps_ms": token_timestamps_ms,
        "decode_wall_time_ms_excluding_final_flush": decode_ms,
        "avg_tpot_ms": decode_ms / max(1, n_gen),
        "final_flush_time_ms": final_flush_ms,
        "completion_latency_ms": decode_ms + final_flush_ms,
        "prompt_compressed_kv_physical_bytes": prompt_bytes,
        "decode_slot_bank_enabled": use_slot_bank,
        "decode_slot_bank_slots": len(slot_bank_layers),
        "decode_slot_bank_physical_bytes": slot_bank_physical_bytes,
        "decode_cache_peak_fp16_equivalent_tokens": decode_peak_equiv,
        "decode_cache_final_fp16_equivalent_tokens": final_equiv,
        "decode_cache_peak_physical_bytes": decode_peak_physical,
        "decode_cache_final_physical_bytes": sum(
            block.physical_bytes for block in compressed
        ) if not use_slot_bank else slot_bank_physical_bytes,
        "num_compressed_decode_blocks": len(compressed),
        "coverage": [
            [block.token_start, block.token_end] for block in compressed
        ],
    }
    expected_plan = plan_block_compressions(n_gen, config.block_size)
    actual_plan = [
        {
            key: event[key]
            for key in (
                "block_id",
                "token_start",
                "token_end",
                "query_source_block",
                "is_final_flush",
                "query_mode",
            )
        }
        for event in block_events
    ]
    if actual_plan != expected_plan:
        raise AssertionError(
            "blockwise compression schedule diverged from reference: "
            f"actual={actual_plan}, expected={expected_plan}"
        )
    return generated_ids, report
