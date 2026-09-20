"""
obkv_accel/decode_hook.py

S4: Decode attention assembly + monkey-patch.

Implements a dual-zone decode system:
  - Old (packed) cache: K in uniform 4-bit, V in mixed 2/4/8-bit segments.
    Attention computed via Triton kernels (or PyTorch reference fallback).
  - New (FP16) cache: standard matmul attention for tokens generated
    since prefill.

Monkey-patches ``LlamaAttention.forward()`` at decode time so that the
rest of the model (RMSNorm, MLP, residual connections) stays untouched.

Compatible with HuggingFace transformers >= 4.40 (new Cache-based API).
"""

from __future__ import annotations

import math
import os
import types
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from obkv_accel.packing import PackedKVLayer, DualZoneCache

# ---------------------------------------------------------------------------
# Kernel imports with fallback to pure-PyTorch references
# ---------------------------------------------------------------------------
_FORCE_REF = os.environ.get("OBKV_FORCE_REF", "0") == "1"

if _FORCE_REF:
    _HAS_TRITON_K = False
    _HAS_TRITON_V = False
else:
    try:
        from obkv_accel.triton_k_kernel import k_dequant_qk_dot
        _HAS_TRITON_K = True
    except (ImportError, RuntimeError):
        _HAS_TRITON_K = False

    try:
        from obkv_accel.triton_v_kernel import v_dequant_weighted_sum_dispatch
        _HAS_TRITON_V = True
    except (ImportError, RuntimeError):
        _HAS_TRITON_V = False

if not _HAS_TRITON_K:
    from obkv_accel.triton_k_kernel import k_dequant_qk_dot_ref as k_dequant_qk_dot

# If Triton V kernels are unavailable, build a dispatch wrapper around the
# pure-PyTorch reference so that the calling convention stays the same.
if not _HAS_TRITON_V:
    from obkv_accel.triton_v_kernel import v_dequant_weighted_sum_ref as _v_ref

    def v_dequant_weighted_sum_dispatch(
        attn_weights_old: torch.Tensor,
        packed_layer: PackedKVLayer,
    ) -> torch.Tensor:
        """Fallback dispatch that delegates to the PyTorch reference."""
        return _v_ref(
            attn_weights_old,
            packed_layer.V_2bit,
            packed_layer.V_4bit,
            packed_layer.V_8bit,
            packed_layer.V_scale,
            packed_layer.V_zp,
            packed_layer.seg_bounds,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT_D = math.sqrt(128.0)


# =====================================================================
# PackedCacheWrapper -- makes DualZoneCache look like an HF Cache
# =====================================================================
class PackedCacheWrapper:
    """Minimal duck-typed wrapper around :class:`DualZoneCache` that
    satisfies the interface expected by ``LlamaModel.forward()`` in
    HuggingFace transformers >= 4.40.

    The main model code queries the cache for:
      * ``get_seq_length(layer_idx)`` -- to build ``cache_position``
      * ``get_mask_sizes(cache_position, layer_idx)`` -- to build the
        causal attention mask with the correct KV length
      * ``__len__`` / ``__getitem__`` -- backward-compat indexing
      * ``is_sliding`` (property) -- to check for sliding-window layers
      * ``update(...)`` -- called inside ``LlamaAttention.forward()``,
        but our monkey-patched forward intercepts this and manages
        the cache itself, so we provide a no-op stub.

    We intentionally do **not** inherit from ``transformers.Cache``
    because the ABC may add new abstract methods across versions.
    Duck-typing is more robust here.
    """

    def __init__(self, dual_cache: DualZoneCache):
        self.dual_cache = dual_cache
        # Provide a ``layers`` attribute so that ``len(self)`` and
        # ``is_sliding`` work without crashing.
        # We store one sentinel per model layer.
        self._num_layers = len(dual_cache.packed)

    # --- Core length queries ------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total number of cached tokens visible to ``layer_idx``.

        TriZone: ``T_old`` is the K sequence length (compressed + v=16
        tokens), ``T_new`` is the decode-appended count in ``new_both_k``.
        """
        pl = self.dual_cache.packed[layer_idx]
        T_old = pl.T_eff_k if pl.T_eff_k > 0 else pl.T_eff
        new_k = self.dual_cache.new_both_k[layer_idx]
        T_new = new_k.shape[2] if new_k is not None else 0
        return T_old + T_new

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    # --- Mask generation helpers --------------------------------------

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for mask creation.

        ``kv_length`` = total KV sequence length the query will attend to
        (cached tokens + current query tokens).
        ``kv_offset`` is always 0 for non-sliding caches.
        """
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0

    # --- Container protocol -------------------------------------------

    def __len__(self) -> int:
        return self._num_layers

    def __getitem__(self, layer_idx: int):
        """Backward-compat: ``past_key_values[i][j].shape[2]`` still
        works for code that indexes into the cache the old way.

        Returns a tuple of two dummy tensors whose ``.shape[2]``
        equals the cached sequence length for that layer.
        """
        seq_len = self.get_seq_length(layer_idx)
        # We only need shape[2] to be correct; other dims are placeholders.
        dummy = torch.empty(1, 1, seq_len, 1)
        return (dummy, dummy)

    # --- Sliding window -----------------------------------------------

    @property
    def is_sliding(self) -> List[bool]:
        return [False] * self._num_layers

    @property
    def is_initialized(self) -> bool:
        return True

    # --- Cache update (no-op; our patched forward manages the cache) ---

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op update. The monkey-patched forward manages ``dual_cache``
        directly, so the default HF code path that calls
        ``past_key_values.update(...)`` should never trigger for our
        patched layers. If it does (e.g. an un-patched layer), we still
        need to return something valid.
        """
        return key_states, value_states


# =====================================================================
# Patched forward factory
# =====================================================================


def make_packed_decode_forward(
    layer_idx: int,
    dual_cache: DualZoneCache,
):
    """Return a replacement ``forward`` method for one layer's
    ``self_attn`` (``LlamaAttention``).

    The returned function is designed for **single-token decode** only
    (``q_len == 1``).  It:

    1. Projects Q / K / V via the module's existing linear layers.
    2. Applies RoPE to Q and the current-step K.
    3. Computes QK scores for the old (packed) cache via the Triton K
       kernel and for the new (FP16) cache + current token via standard
       ``torch.matmul``.
    4. Concatenates scores, applies global FP32 softmax, then splits
       the resulting weights.
    5. Computes old-cache V contribution via the Triton V kernel and
       the new-cache V contribution via standard matmul.
    6. Sums, casts back to model dtype, and applies the O projection.
    7. Appends the current K / V to ``dual_cache.new_k / new_v``.

    Parameters
    ----------
    layer_idx : int
        Index of the decoder layer being patched.
    dual_cache : DualZoneCache
        Shared mutable cache object.

    Returns
    -------
    forward : callable
        A bound method (via ``types.MethodType``) is **not** created
        here; instead we return a plain function whose first positional
        argument is ``self`` (the attention module instance).  The
        caller should assign it with
        ``layer.self_attn.forward = types.MethodType(fn, layer.self_attn)``.
    """
    # TriZone (方案 1): this hook path references stale PackedKVLayer fields
    # (``K_packed``/``K_scale``/``K_zp``) that were removed before the
    # per-head K Phase 2 refactor and it was never updated to consume the
    # new ``TriZoneCache`` three-zone layout. There are no external callers
    # of ``greedy_decode_packed``; use ``greedy_decode_fast`` instead.
    raise NotImplementedError(
        "decode_hook.make_packed_decode_forward is incompatible with "
        "TriZoneCache; use obkv_accel.fast_decode.greedy_decode_fast."
    )

    packed_layer: PackedKVLayer = dual_cache.packed[layer_idx]

    def _patched_forward(
        self_attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single-token decode forward with dual-zone cache."""

        bsz, q_len, _ = hidden_states.size()
        model_dtype = hidden_states.dtype

        # New transformers (>=4.46): attributes on config, not module
        _cfg = self_attn.config
        num_heads = _cfg.num_attention_heads       # H_q = 32
        num_kv_heads = _cfg.num_key_value_heads    # H_kv = 8
        num_kv_groups = self_attn.num_key_value_groups  # 4 (still on module)
        head_dim = self_attn.head_dim              # 128 (still on module)
        hidden_size = num_heads * head_dim         # 4096

        # ---- 1. Q / K / V projection ----
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        # Reshape: [B, 1, H, D] -> [B, H, 1, D]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # ---- 2. RoPE ----
        # Older HF versions compute RoPE inside each attention layer;
        # newer versions may pass precomputed position_embeddings=(cos, sin).
        if position_embeddings is None:
            cos, sin = self_attn.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        # Squeeze batch dim for kernel calls (B=1 during decode)
        # q: [H_q, D]     cur_k: [H_kv, D]     cur_v: [H_kv, D]
        q = query_states.squeeze(0).squeeze(-2)         # [H_q, D]
        cur_k = key_states.squeeze(0).squeeze(-2)       # [H_kv, D]
        cur_v = value_states.squeeze(0).squeeze(-2)     # [H_kv, D]

        # ---- 3. Old-cache QK scores (Triton or reference) ----
        T_old = packed_layer.T_eff
        # K kernel expects: Q [H_q, D], K_packed [H_kv, T, D//2], etc.
        # PackedKVLayer stores shapes [1, H_kv, ...]; squeeze the batch.
        scores_old = k_dequant_qk_dot(
            q,                                          # [H_q, D]
            packed_layer.K_packed.squeeze(0),            # [H_kv, T_eff, D//2]
            packed_layer.K_scale.squeeze(0),             # [H_kv, 1, D]
            packed_layer.K_zp.squeeze(0),                # [H_kv, 1, D]
            T_old,
        )  # -> [H_q, T_old]  FP32

        # ---- 4. New-cache + current-token QK scores (standard matmul) ----
        new_k = dual_cache.new_k[layer_idx]  # [1, H_kv, T_new, D] or None

        if new_k is not None:
            # Cat current K onto new cache for the QK computation
            new_and_cur_k = torch.cat(
                [new_k.squeeze(0), cur_k.unsqueeze(1)], dim=1,
            )  # [H_kv, T_new+1, D]
        else:
            new_and_cur_k = cur_k.unsqueeze(1)  # [H_kv, 1, D]

        T_new_plus_1 = new_and_cur_k.shape[1]

        # GQA expand for new keys: [H_kv, T, D] -> [H_q, T, D]
        new_and_cur_k_expanded = new_and_cur_k.unsqueeze(0)  # [1, H_kv, T, D]
        new_and_cur_k_expanded = repeat_kv(
            new_and_cur_k_expanded, num_kv_groups,
        ).squeeze(0)  # [H_q, T_new+1, D]

        # QK: [H_q, T_new+1]
        scores_new = torch.einsum(
            'hd,htd->ht', q.float(), new_and_cur_k_expanded.float(),
        ) / _SQRT_D  # FP32

        # ---- 5. Global softmax in FP32 ----
        all_scores = torch.cat([scores_old, scores_new], dim=-1)  # [H_q, T_old+T_new+1]
        attn_weights = F.softmax(all_scores, dim=-1, dtype=torch.float32)

        # Split weights
        w_old = attn_weights[:, :T_old]                    # [H_q, T_old]
        w_new = attn_weights[:, T_old:]                    # [H_q, T_new+1]

        # ---- 6. Old V: Triton weighted sum ----
        # v_dequant_weighted_sum_dispatch expects:
        #   attn_weights_old: [H_q, T_eff] FP32
        #   packed_layer with squeezed V data [H_kv, ...]
        # Build a squeezed PackedKVLayer view for the dispatch.
        squeezed_layer = PackedKVLayer(
            K_packed=packed_layer.K_packed.squeeze(0),
            K_scale=packed_layer.K_scale.squeeze(0),
            K_zp=packed_layer.K_zp.squeeze(0),
            V_2bit=(packed_layer.V_2bit.squeeze(0)
                    if packed_layer.V_2bit is not None else None),
            V_4bit=(packed_layer.V_4bit.squeeze(0)
                    if packed_layer.V_4bit is not None else None),
            V_8bit=(packed_layer.V_8bit.squeeze(0)
                    if packed_layer.V_8bit is not None else None),
            V_scale=packed_layer.V_scale.squeeze(0),     # [H_kv, T_eff, 1]
            V_zp=packed_layer.V_zp.squeeze(0),           # [H_kv, T_eff, 1]
            sort_idx=packed_layer.sort_idx,
            seg_bounds=packed_layer.seg_bounds,
            T_eff=packed_layer.T_eff,
        )
        out_old = v_dequant_weighted_sum_dispatch(
            w_old, squeezed_layer,
        )  # [H_q, D]  FP32

        # ---- 7. New V: standard matmul ----
        new_v = dual_cache.new_v[layer_idx]  # [1, H_kv, T_new, D] or None

        if new_v is not None:
            new_and_cur_v = torch.cat(
                [new_v.squeeze(0), cur_v.unsqueeze(1)], dim=1,
            )  # [H_kv, T_new+1, D]
        else:
            new_and_cur_v = cur_v.unsqueeze(1)  # [H_kv, 1, D]

        # GQA expand: [H_kv, T, D] -> [H_q, T, D]
        new_and_cur_v_expanded = new_and_cur_v.unsqueeze(0)  # [1, H_kv, T, D]
        new_and_cur_v_expanded = repeat_kv(
            new_and_cur_v_expanded, num_kv_groups,
        ).squeeze(0)  # [H_q, T_new+1, D]

        # out_new = w_new @ V_new  =>  [H_q, D]
        out_new = torch.einsum(
            'ht,htd->hd', w_new, new_and_cur_v_expanded.float(),
        )  # [H_q, D]  FP32

        # ---- 8. Sum and cast back ----
        attn_output = (out_old + out_new).to(model_dtype)   # [H_q, D]

        # Reshape: [H_q, D] -> [B=1, 1, hidden_size]
        attn_output = attn_output.unsqueeze(0).unsqueeze(0)   # [1, 1, H_q, D]
        # Note: attn_output is [1, 1, H_q, D], but o_proj expects [B, q_len, hidden_size]
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        # ---- 9. O projection ----
        attn_output = self_attn.o_proj(attn_output)

        # ---- 10. Update new_k / new_v in dual_cache ----
        # Store post-RoPE K and raw V for future decode steps.
        # key_states, value_states are [B=1, H_kv, 1, D].
        if dual_cache.new_k[layer_idx] is not None:
            dual_cache.new_k[layer_idx] = torch.cat(
                [dual_cache.new_k[layer_idx],
                 key_states.detach()], dim=2,
            )
        else:
            dual_cache.new_k[layer_idx] = key_states.detach()

        if dual_cache.new_v[layer_idx] is not None:
            dual_cache.new_v[layer_idx] = torch.cat(
                [dual_cache.new_v[layer_idx],
                 value_states.detach()], dim=2,
            )
        else:
            dual_cache.new_v[layer_idx] = value_states.detach()

        # ---- 11. Return ----
        # New transformers (>=4.46) LlamaDecoderLayer unpacks only 2 values:
        #   hidden_states, _ = self.self_attn(...)
        # Older versions expected 3: (attn_output, attn_weights, past_key_value).
        # Return 2 to match the current installed version.
        return attn_output, None

    return _patched_forward


# =====================================================================
# greedy_decode_packed
# =====================================================================


def greedy_decode_packed(
    model,
    dual_cache: DualZoneCache,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: int,
) -> List[int]:
    """Greedy decode using the packed dual-zone cache.

    This function:
      1. Installs monkey-patches on every layer's ``self_attn.forward``.
      2. Runs a standard auto-regressive decode loop, calling
         ``model.forward()`` each step.
      3. Restores original forwards in a ``finally`` block.

    Parameters
    ----------
    model
        A ``LlamaForCausalLM`` instance (HuggingFace).
    dual_cache : DualZoneCache
        Pre-built packed cache from prefill.
    next_token_logits : Tensor
        ``[1, vocab_size]`` or ``[vocab_size]`` logits from the prefill's
        last position.
    max_new_tokens : int
        Maximum number of tokens to generate.
    eos_token_ids : sequence of int
        Stop-token IDs.
    primary_device : torch.device
        Device on which to run decode.
    decode_position_start : int
        The *original* sequence position of the first generated token
        (i.e. ``original_seq_len``, not ``T_eff``).

    Returns
    -------
    generated_ids : list[int]
        Token IDs produced by greedy decoding.
    """
    # TriZone (方案 1): see make_packed_decode_forward above — this entry
    # point has no external callers and is incompatible with the new
    # three-zone cache layout. Use ``greedy_decode_fast`` instead.
    raise NotImplementedError(
        "decode_hook.greedy_decode_packed is incompatible with TriZoneCache; "
        "use obkv_accel.fast_decode.greedy_decode_fast."
    )

    eos_set = {int(tid) for tid in eos_token_ids if tid is not None}
    generated_ids: List[int] = []

    # ---- Monkey-patch attention forwards ----
    layers = model.model.layers
    num_layers = len(layers)
    original_forwards: List[Any] = [None] * num_layers

    for i in range(num_layers):
        original_forwards[i] = layers[i].self_attn.forward
        patched_fn = make_packed_decode_forward(i, dual_cache)
        layers[i].self_attn.forward = types.MethodType(
            patched_fn, layers[i].self_attn,
        )

    cache_wrapper = PackedCacheWrapper(dual_cache)

    # First token from prefill logits
    if next_token_logits.dim() == 1:
        next_token_logits = next_token_logits.unsqueeze(0)
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

    try:
        for i in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in eos_set:
                break
            generated_ids.append(token_id)

            position_ids = torch.tensor(
                [[decode_position_start + i]],
                device=primary_device,
                dtype=torch.long,
            )

            with torch.inference_mode():
                outputs = model(
                    input_ids=next_token.to(primary_device),
                    past_key_values=cache_wrapper,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )

            # The model returns ``past_key_values`` but we manage the
            # cache internally; ignore the returned value.
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    finally:
        # ---- Restore original forwards ----
        for i in range(num_layers):
            layers[i].self_attn.forward = original_forwards[i]

    return generated_ids
