"""Decompress a TriZoneCache into HF-compatible legacy past_key_values.

Additive module for the cpu_offload 70B decode path. Running
``greedy_decode_fast`` on a model with weights partially on CPU fails in
``pre_extract_weights`` (concatenating qkv/gate_up per layer doubles the
weight footprint). For accuracy eval we instead dequantise the packed
cache back to fp16 ``(K, V)`` tuples and hand them to
``greedy_decode_pt`` which goes through ``model.forward`` — that path
honours accelerate's AlignDevicesHook and handles the per-layer
weight swap transparently.

Nothing here modifies existing behaviour: the module is only imported
when ``OBKV_DECOMPRESS_DECODE=1``.

Per-layer cache layout after unpack:
  K / V: [1, H_kv, T_eff_k, D] fp16
  T_eff_k = T_v (Zone A, v_bits ∈ {2,4,8}) + n_v16 (Zone B, v_bits = 16)
  Within Zone A:    sort_idx order (grouped by bit-width).
  Within Zone B:    v16_ids ascending (natural original-T order).

Attention math is preserved across layers because K has RoPE baked in
at its *original* position (not cache index) and V is looked up via the
same index that K is — so every cached slot has a consistent (K, V)
pair. HF's ``DynamicCache.from_legacy_cache`` accepts varying per-layer
lengths; position encoding for new tokens comes from the explicit
``position_ids`` passed by ``greedy_decode_pt``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .packing import (
    PackedKVLayer,
    TriZoneCache,
    unpack_k_mixed,
    unpack_v_segment,
)


def _unpack_v_zone_a(pl: PackedKVLayer) -> Tensor:
    """Dequantise the v=2/4/8 V segments and concatenate them along T.

    Output is in ``sort_idx`` order (the same order the segments were
    packed in ``pack_v_segments``). Shape: ``[1, H_kv, T_v, D]`` fp16.
    """
    N_2, N_4, N_8 = pl.seg_bounds
    T_v = int(pl.T_eff)
    assert N_2 + N_4 + N_8 == T_v, (
        f"V seg_bounds {(N_2, N_4, N_8)} must sum to T_v={T_v}"
    )
    H_kv = pl.V_scale.shape[1]
    D = pl.V_scale.shape[3] if pl.V_scale.dim() == 4 else 0
    # V_scale/V_zp are [1, H_kv, T_v, 1] — the segment-wise ``D`` comes
    # from the packed segment tensor itself.
    device = pl.V_scale.device

    if T_v == 0:
        # Infer D from K scale fallback; caller will still pass head_dim.
        return torch.empty(1, H_kv, 0, 1, dtype=torch.float16, device=device)

    parts: List[Tensor] = []
    offset = 0
    if N_2 > 0 and pl.V_2bit is not None:
        s = pl.V_scale[:, :, offset:offset + N_2, :]
        z = pl.V_zp[:, :, offset:offset + N_2, :]
        parts.append(unpack_v_segment(pl.V_2bit, s, z, n_bits=2))
        offset += N_2
    if N_4 > 0 and pl.V_4bit is not None:
        s = pl.V_scale[:, :, offset:offset + N_4, :]
        z = pl.V_zp[:, :, offset:offset + N_4, :]
        parts.append(unpack_v_segment(pl.V_4bit, s, z, n_bits=4))
        offset += N_4
    if N_8 > 0 and pl.V_8bit is not None:
        s = pl.V_scale[:, :, offset:offset + N_8, :]
        z = pl.V_zp[:, :, offset:offset + N_8, :]
        parts.append(unpack_v_segment(pl.V_8bit, s, z, n_bits=8))

    if not parts:
        return torch.empty(1, H_kv, 0, 1, dtype=torch.float16, device=device)
    return torch.cat(parts, dim=2)


def _is_perhead_layout(pl: PackedKVLayer) -> bool:
    """True when ``pl`` came out of ``_pad_and_stack_per_head_layers``."""
    return pl.seg_bounds_per_head is not None


def _unpack_v_perhead(pl: PackedKVLayer, *, head_dim: int) -> Tensor:
    """Per-head Zone-A V dequantisation under the stripe layout.

    Returns ``[1, H_kv, max_T_v, head_dim]`` fp16 with per-head
    ``[0, T_v_h)`` filled with the dequantised compressed tokens and
    ``[T_v_h, max_T_v)`` zero-padded — matching the V kernel's ``w[:, :T_v_h]``
    consumption rule on the prefill side.
    """
    max_T_v = int(pl.T_eff)
    H_kv = pl.V_scale.shape[1]
    device = pl.V_scale.device
    V_out = torch.zeros(
        1, H_kv, max_T_v, head_dim,
        dtype=torch.float16, device=device,
    )
    if max_T_v == 0:
        return V_out

    sb_ph = pl.seg_bounds_per_head.cpu()
    tv_ph = pl.T_eff_per_head.cpu()
    for h in range(H_kv):
        n2, n4, n8 = (int(x) for x in sb_ph[h].tolist())
        t_v_h = int(tv_ph[h])
        assert n2 + n4 + n8 == t_v_h, (
            f"head {h}: seg_bounds_per_head ({n2},{n4},{n8}) "
            f"!= T_eff_per_head {t_v_h}"
        )
        if t_v_h == 0:
            continue
        parts: List[Tensor] = []
        off = 0
        if n2 > 0 and pl.V_2bit is not None:
            s = pl.V_scale[:, h:h + 1, off:off + n2, :]
            z = pl.V_zp[:, h:h + 1, off:off + n2, :]
            parts.append(unpack_v_segment(
                pl.V_2bit[:, h:h + 1, :n2, :], s, z, n_bits=2,
            ))
            off += n2
        if n4 > 0 and pl.V_4bit is not None:
            s = pl.V_scale[:, h:h + 1, off:off + n4, :]
            z = pl.V_zp[:, h:h + 1, off:off + n4, :]
            parts.append(unpack_v_segment(
                pl.V_4bit[:, h:h + 1, :n4, :], s, z, n_bits=4,
            ))
            off += n4
        if n8 > 0 and pl.V_8bit is not None:
            s = pl.V_scale[:, h:h + 1, off:off + n8, :]
            z = pl.V_zp[:, h:h + 1, off:off + n8, :]
            parts.append(unpack_v_segment(
                pl.V_8bit[:, h:h + 1, :n8, :], s, z, n_bits=8,
            ))
        if parts:
            V_out[:, h:h + 1, :t_v_h, :] = torch.cat(parts, dim=2)
    return V_out


def _assemble_v_stripe_perhead(
    V_zone_a: Tensor,
    V_zone_b: Optional[Tensor],
    *,
    max_T_v: int,
    max_n_v16: int,
    head_dim: int,
    H_kv: int,
    device: torch.device,
) -> Tensor:
    """Concatenate Zone A and Zone B into the K-stripe layout
    ``[1, H_kv, max_T_v + max_n_v16, head_dim]`` fp16.

    ``V_zone_b`` (``dual_cache.new_v_only[i]``) is already padded to
    ``[1, H_kv, max_n_v16, D]`` per-head via ``_pad_and_stack_per_head_layers``
    so a plain ``torch.cat`` is correct.
    """
    if max_T_v == 0 and max_n_v16 == 0:
        return torch.zeros(
            1, H_kv, 0, head_dim, dtype=torch.float16, device=device,
        )
    if max_n_v16 == 0:
        return V_zone_a.to(device)
    if max_T_v == 0:
        assert V_zone_b is not None, (
            "max_T_v == 0 but max_n_v16 > 0 with V_zone_b=None — "
            "inconsistent PackedKVLayer state"
        )
        return V_zone_b.to(device)
    assert V_zone_b is not None, (
        f"max_n_v16={max_n_v16} > 0 but new_v_only is None — "
        "inconsistent PackedKVLayer state"
    )
    return torch.cat(
        [V_zone_a.to(device), V_zone_b.to(device)], dim=2,
    )


def unpack_trizone_to_past_kv(
    dual_cache: TriZoneCache,
    *,
    head_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    return_masks: bool = False,
    H_q: Optional[int] = None,
    max_new_tokens: int = 0,
    mask_dtype: torch.dtype = torch.float16,
):
    """Reverse a ``TriZoneCache`` into HF legacy ``past_key_values``.

    The returned tuple can be passed directly to ``greedy_decode_pt``
    (which wraps it in ``DynamicCache.from_legacy_cache`` via
    ``prepare_past_key_values_for_model``).

    Args
    ----
    dual_cache : TriZoneCache
        Output of ``_streaming_prefill_and_pack``. Its ``packed`` tuple
        and ``new_v_only`` list are consumed read-only.
    head_dim : int
        Original per-head channel count ``D`` (e.g. 128). Needed by
        ``unpack_k_mixed`` because the scale/zp tensors are padded to
        ``D_kept_padded_max``.
    device : Optional[torch.device]
        If given, move each layer's output to this device. Default =
        keep on whichever device the packed tensors live.
    dtype : torch.dtype
        Cast result to this dtype. Default fp16 (matches HF prefill).
    return_masks : bool
        When True, return ``(past_kv, masks_list)``. ``masks_list[i]`` is
        a 4D ``[1, H_q, 1, T_eff_k_i + max_new_tokens]`` per-head additive
        attention mask for the per-head packed layout, or ``None`` for the
        shared layout (caller falls back to plain ``greedy_decode_pt``).
    H_q : Optional[int]
        Number of query heads. For the per-head layout we infer it from
        ``pl.softmax_mask.shape[0]``; if ``H_q`` is also passed, the two
        must agree (asserted) — guards against a wrong call-site value.
    max_new_tokens : int
        Right-pads the per-head mask along the key axis with zeros so
        SDPA's auto-slice ``[..., :key.shape[-2]]`` always finds enough
        columns during decode (each new token grows the cache by 1).
    mask_dtype : torch.dtype
        Dtype to cast the per-head mask to. **Default fp16, intentional**:
        passing fp32 ``-inf`` to PyTorch SDPA's safe-softmax kernel makes
        the max-subtraction step compute ``-inf - (-inf) = NaN`` and the
        whole row collapses to NaN logits. fp16 ``-inf`` routes to a
        different SDPA backend that handles all-masked rows without NaN
        and was empirically verified on Llama-3.1-8B.

    Returns
    -------
    Tuple of ``(K_layer_i, V_layer_i)`` for ``i in [0, num_layers)``,
    each tensor ``[1, H_kv, T_eff_k_i, head_dim]``. When ``return_masks``
    is True, returns ``(past_kv, masks)`` where ``masks`` is a list of
    per-layer 4D masks (or None entries for shared-layout layers).
    """
    past_kv: List[Tuple[Tensor, Tensor]] = []
    masks_list: List[Optional[Tensor]] = []
    for i, pl in enumerate(dual_cache.packed):
        K_full = unpack_k_mixed(pl, D=head_dim)  # [1, H_kv, T_eff_k, D]

        per_head = _is_perhead_layout(pl)
        if per_head:
            V_zone_a = _unpack_v_perhead(pl, head_dim=head_dim)
            V_zone_b = (
                dual_cache.new_v_only[i]
                if i < len(dual_cache.new_v_only) else None
            )
            H_kv = pl.V_scale.shape[1]
            max_T_v = int(pl.T_eff)
            max_n_v16 = int(pl.n_v16)
            v_device = (
                device if device is not None else pl.V_scale.device
            )
            V_full = _assemble_v_stripe_perhead(
                V_zone_a, V_zone_b,
                max_T_v=max_T_v, max_n_v16=max_n_v16,
                head_dim=head_dim, H_kv=H_kv, device=v_device,
            )
        else:
            V_zone_a = _unpack_v_zone_a(pl)  # [1, H_kv, T_v, D] (or [..., 0, 1])
            V_zone_b = (
                dual_cache.new_v_only[i]
                if i < len(dual_cache.new_v_only) else None
            )
            if V_zone_b is not None and V_zone_b.shape[2] > 0:
                if V_zone_a.shape[2] == 0:
                    V_full = V_zone_b
                else:
                    V_full = torch.cat(
                        [V_zone_a.to(V_zone_b.device), V_zone_b], dim=2,
                    )
            else:
                V_full = V_zone_a
            # If V zone A was the ``[..., 0, 1]`` placeholder (no compressed
            # tokens) and there's no zone B, fall back to an empty tensor
            # with the correct D.
            if V_full.shape[2] == 0 and V_full.shape[3] != head_dim:
                V_full = torch.empty(
                    1, K_full.shape[1], 0, head_dim,
                    dtype=V_full.dtype, device=V_full.device,
                )

        assert K_full.shape[2] == V_full.shape[2], (
            f"layer {i}: K T-dim {K_full.shape[2]} != V T-dim {V_full.shape[2]} "
            f"(T_v={pl.T_eff}, n_v16={pl.n_v16}, T_eff_k={pl.T_eff_k})"
        )
        assert K_full.shape[3] == head_dim and V_full.shape[3] == head_dim, (
            f"layer {i}: K/V last dim must equal head_dim={head_dim}, "
            f"got K={K_full.shape}, V={V_full.shape}"
        )

        if device is not None:
            K_full = K_full.to(device)
            V_full = V_full.to(device)
        K_full = K_full.to(dtype)
        V_full = V_full.to(dtype)

        past_kv.append((K_full.contiguous(), V_full.contiguous()))

        if return_masks:
            if per_head and pl.softmax_mask is not None:
                mask_2d = pl.softmax_mask  # [H_q, max_T_eff_k]
                mask_H_q = int(mask_2d.shape[0])
                if H_q is not None:
                    assert H_q == mask_H_q, (
                        f"layer {i}: caller H_q={H_q} != softmax_mask "
                        f"H_q={mask_H_q} — wrong head count at call-site"
                    )
                T_eff_k = int(mask_2d.shape[1])
                target_device = device if device is not None else mask_2d.device
                mask_2d = mask_2d.to(device=target_device, dtype=mask_dtype)
                mask_4d = mask_2d.view(1, mask_H_q, 1, T_eff_k)
                if max_new_tokens > 0:
                    pad = torch.zeros(
                        1, mask_H_q, 1, max_new_tokens,
                        dtype=mask_dtype, device=target_device,
                    )
                    mask_4d = torch.cat([mask_4d, pad], dim=-1)
                masks_list.append(mask_4d.contiguous())
            else:
                masks_list.append(None)

    if return_masks:
        return tuple(past_kv), masks_list
    return tuple(past_kv)
