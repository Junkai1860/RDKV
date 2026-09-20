"""obkv_fast.py — OBKV core inference library.

Self-contained pipeline for value_residual_probe scoring + global knapsack
bit allocation + Triton packed decode + reflect pooling.

This is a pure library: no `main()`, no CLI, no argparse.
"""
import json
import math
import os
import random
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama import modeling_llama as hf_llama_modeling
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
try:
    from transformers.cache_utils import Cache as HFCache, DynamicCache
except ImportError:
    HFCache = None
    DynamicCache = None

from knapsack_solver import (
    knapsack_bit_allocation,
    knapsack_bit_allocation_batched,
    DEFAULT_EPSILON_K,
    DEFAULT_EPSILON_V,
    load_epsilon_kv_from_calibration,
    allocate_v_bits_with_fp16_topk,
)
from obkv_accel.packing import (
    build_packed_cache,
    build_packed_layer,
    DualZoneCache,
    TriZoneCache,
    _empty_packed_layer,
    _pad_and_stack_per_head_layers,
)
from obkv_accel.fast_decode import greedy_decode_fast, pre_extract_weights

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"



def patch_llama31_rope_compat() -> None:
    if getattr(LlamaConfig, "_rdkv_llama31_rope_patched", False):
        return

    # Transformers >= 4.5x already validates llama3 RoPE via rope_config_validation
    # and no longer exposes the legacy hooks we patched in older releases.
    if not hasattr(LlamaConfig, "_rope_scaling_validation") or not hasattr(LlamaAttention, "_init_rope"):
        LlamaConfig._rdkv_llama31_rope_patched = True
        return

    original_validation = LlamaConfig._rope_scaling_validation

    def patched_validation(self) -> None:
        rope_scaling = self.rope_scaling
        if rope_scaling is None:
            return
        if not isinstance(rope_scaling, dict):
            raise ValueError(f"`rope_scaling` must be a dictionary, got {rope_scaling}")

        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        factor = rope_scaling.get("factor")
        if rope_type == "llama3":
            required_keys = {
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
                "rope_type",
            }
            missing = sorted(required_keys.difference(rope_scaling))
            if missing:
                raise ValueError(
                    f"`rope_scaling` for llama3 is missing required keys: {', '.join(missing)}"
                )
            if not isinstance(factor, float) or factor <= 1.0:
                raise ValueError(f"`rope_scaling.factor` must be a float > 1, got {factor}")
            return

        if "type" not in rope_scaling and rope_type is not None:
            self.rope_scaling = dict(rope_scaling)
            self.rope_scaling["type"] = rope_type
        original_validation(self)

    class Llama31RotaryEmbedding(hf_llama_modeling.LlamaRotaryEmbedding):
        def __init__(
            self,
            dim,
            *,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            rope_scaling=None,
        ):
            super().__init__(
                dim,
                max_position_embeddings=max_position_embeddings,
                base=base,
                device=device,
            )
            if rope_scaling is None:
                raise ValueError("`rope_scaling` is required for llama3 rotary embedding.")

            factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            old_context_len = rope_scaling["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2 * math.pi / self.inv_freq

            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            self.register_buffer("inv_freq", inv_freq_llama, persistent=False)

    def patched_init_rope(self) -> None:
        rope_scaling = self.config.rope_scaling
        if rope_scaling is None:
            self.rotary_emb = hf_llama_modeling.LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            return

        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        scaling_factor = rope_scaling["factor"]
        if rope_type == "linear":
            self.rotary_emb = hf_llama_modeling.LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif rope_type == "dynamic":
            self.rotary_emb = hf_llama_modeling.LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif rope_type == "llama3":
            self.rotary_emb = Llama31RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=rope_scaling,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {rope_type}")

    LlamaConfig._rope_scaling_validation = patched_validation
    LlamaAttention._init_rope = patched_init_rope
    LlamaConfig._rdkv_llama31_rope_patched = True



def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent



def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def apply_score_pooling(
    scores: torch.Tensor,
    pool_type: str,
    kernel_size: int,
    padding_mode: str = "zero",
) -> torch.Tensor:
    pool_fn = F.avg_pool1d if pool_type == "avg" else F.max_pool1d
    effective_kernel_size = kernel_size
    if padding_mode == "reflect":
        sequence_length = int(scores.shape[-1])
        if sequence_length <= 0:
            raise ValueError("Cannot reflect-pool an empty score sequence.")
        # torch reflection padding requires pad < sequence_length.  A final
        # decode flush may contain only one or two tokens, so use the largest
        # kernel that preserves reflection semantics for that short suffix.
        effective_kernel_size = min(kernel_size, 2 * sequence_length - 1)
    pad = effective_kernel_size // 2

    if scores.dim() == 1:
        pooled_input = scores.unsqueeze(0).unsqueeze(0)
        if padding_mode == "reflect":
            pooled_input = F.pad(pooled_input, (pad, pad), mode="reflect")
            pooled = pool_fn(
                pooled_input, kernel_size=effective_kernel_size, stride=1
            )
        else:
            pooled = pool_fn(pooled_input, kernel_size=kernel_size, stride=1, padding=pad)
        return pooled.squeeze(0).squeeze(0)

    if scores.dim() == 2:
        pooled_input = scores.unsqueeze(1)
        if padding_mode == "reflect":
            pooled_input = F.pad(pooled_input, (pad, pad), mode="reflect")
            pooled = pool_fn(
                pooled_input, kernel_size=effective_kernel_size, stride=1
            )
        else:
            pooled = pool_fn(pooled_input, kernel_size=kernel_size, stride=1, padding=pad)
        return pooled.squeeze(1)

    raise ValueError(f"Expected score tensor rank 1 or 2, got shape={tuple(scores.shape)}.")



def resolve_primary_device(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, str) and value not in {"cpu", "disk", "meta"}:
                return torch.device(value)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, torch.device) and value.type not in {"cpu", "meta"}:
                return value
    return next(model.parameters()).device


def middle_truncate_token_ids(token_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if token_ids.numel() <= max_tokens:
        return token_ids
    half = max_tokens // 2
    if half == 0:
        return token_ids[:max_tokens]
    tail = max_tokens - half
    return torch.cat([token_ids[:half], token_ids[-tail:]], dim=0)




def load_model_and_tokenizer(
    model_path: str,
    mode: str,
    attn_implementation: str = "flash_attention_2",
    device_map_arg: str = "single",
) -> Tuple[torch.nn.Module, object, torch.device]:
    local_files_only = Path(model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    # Only apply Llama-3.1 RoPE compatibility patch for Llama models
    if getattr(config, "model_type", None) == "llama":
        patch_llama31_rope_compat()
    if hasattr(config, "_flash_attn_2_enabled"):
        config._flash_attn_2_enabled = attn_implementation == "flash_attention_2"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = attn_implementation

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kwargs = {
        "config": config,
        "torch_dtype": model_dtype,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        if device_map_arg == "balanced":
            load_kwargs["device_map"] = "balanced"
            # Optional per-GPU cap to force a multi-device split on small
            # models (Gate 3 smoke test). Format: "8GiB,8GiB" or
            # "0:8GiB,1:8GiB". Without this, HF will pack 8B/13B onto
            # cuda:0 and the smoke silently re-tests the single-card path.
            balanced_max_mem_env = os.environ.get("OBKV_BALANCED_MAX_MEMORY")
            if balanced_max_mem_env:
                max_memory: Dict = {}
                for i, part in enumerate(
                    p.strip() for p in balanced_max_mem_env.split(",") if p.strip()
                ):
                    if ":" in part:
                        key_str, val_str = part.split(":", 1)
                        key_str = key_str.strip()
                        if key_str.lower() == "cpu":
                            max_memory["cpu"] = val_str.strip()
                        else:
                            max_memory[int(key_str)] = val_str.strip()
                    else:
                        max_memory[i] = part
                load_kwargs["max_memory"] = max_memory
        elif device_map_arg == "cpu_offload":
            # Single-GPU inference with weights spilled to CPU RAM. accelerate's
            # AlignDevicesHook swaps each module's weights CPU→GPU just before
            # forward and back after. Use for 70B-scale accuracy eval when
            # model doesn't fit single-GPU — do NOT use for latency numbers.
            assert torch.cuda.device_count() >= 1, "cpu_offload requires 1 GPU"
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            # Leave ~25 GB headroom on a 97 GB GH200 for activations (128K
            # prefill q/k/v/MLP intermediates) + TriZone packed cache + Triton
            # scratch. Tune via OBKV_CPU_OFFLOAD_GPU_GIB if needed.
            gpu_gib = int(os.environ.get(
                "OBKV_CPU_OFFLOAD_GPU_GIB",
                str(max(8, gpu_total // (1024 ** 3) - 25)),
            ))
            cpu_gib = int(os.environ.get("OBKV_CPU_OFFLOAD_CPU_GIB", "200"))
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {
                0: f"{gpu_gib}GiB",
                "cpu": f"{cpu_gib}GiB",
            }
        elif device_map_arg == "auto" or (mode == "backward" and torch.cuda.device_count() > 1):
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    primary_device = resolve_primary_device(model)
    if not hasattr(model, "hf_device_map"):
        model = model.to(primary_device)
    # Eviction eval never updates weights, so keep parameter gradients off.
    model.requires_grad_(False)
    if mode == "backward":
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.eval()
    return model, tokenizer, primary_device



def evict_past_key_values(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    keep_ids: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if keep_ids.numel() == past_key_values[0][0].shape[2]:
        return past_key_values
    new_past = []
    for key_states, value_states in past_key_values:
        layer_keep_ids = keep_ids.to(device=key_states.device)
        expanded = layer_keep_ids.view(1, 1, -1, 1).expand(
            key_states.shape[0],
            key_states.shape[1],
            layer_keep_ids.shape[0],
            key_states.shape[-1],
        )
        key_new = key_states.gather(2, expanded)
        value_new = value_states.gather(2, expanded)
        new_past.append((key_new, value_new))
    return tuple(new_past)



def detach_past_key_values(
    past_key_values,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if past_key_values is None:
        return tuple()
    if HFCache is not None and isinstance(past_key_values, HFCache):
        if not hasattr(past_key_values, "to_legacy_cache"):
            raise TypeError(
                f"Unsupported cache object without `to_legacy_cache()`: {type(past_key_values).__name__}"
            )
        past_key_values = past_key_values.to_legacy_cache()
    return tuple((key_states.detach(), value_states.detach()) for key_states, value_states in past_key_values)


def prepare_past_key_values_for_model(past_key_values):
    if not past_key_values:
        return None
    if DynamicCache is not None and isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values



def prefill_last_token(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    outputs = model.model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )
    last_hidden_state = outputs.last_hidden_state[:, -1:, :]
    next_token_logits = model.lm_head(last_hidden_state).float()[:, 0, :]
    past_key_values = detach_past_key_values(outputs.past_key_values)
    return next_token_logits, past_key_values


def greedy_decode_pt(
    model: torch.nn.Module,
    past_key_values,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: Optional[int] = None,
) -> List[int]:
    """Plain PyTorch greedy decode through HF model.forward().

    Used by baselines (Full KV, kvpress) that don't go through the packed
    Triton decode path. OBKV production path uses greedy_decode_fast.
    """
    eos_set = {int(token_id) for token_id in eos_token_ids if token_id is not None}
    generated_ids: List[int] = []
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
    for i in range(max_new_tokens):
        token_id = int(next_token.item())
        if token_id in eos_set:
            break
        generated_ids.append(token_id)
        model_kwargs = {
            "input_ids": next_token.to(primary_device),
            "past_key_values": prepare_past_key_values_for_model(past_key_values),
            "use_cache": True,
            "return_dict": True,
        }
        if decode_position_start is not None:
            model_kwargs["position_ids"] = torch.tensor(
                [[decode_position_start + i]],
                device=next_token.device,
                dtype=torch.long,
            )
        with torch.inference_mode():
            outputs = model(**model_kwargs)
        past_key_values = detach_past_key_values(outputs.past_key_values)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return generated_ids


def greedy_decode_pt_with_perhead_masks(
    model: torch.nn.Module,
    *,
    past_key_values,
    next_token_logits: torch.Tensor,
    per_layer_masks: Sequence[torch.Tensor],
    per_layer_base_lens: Sequence[int],
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    primary_device: torch.device,
    decode_position_start: Optional[int] = None,
) -> List[int]:
    """Variant of greedy_decode_pt that injects per-layer per-head 4D
    attention_mask via forward pre-hooks.

    Required for per-head ragged caches produced by
    ``_pad_and_stack_per_head_layers`` — pad rows in K/V must be masked out
    head-by-head so softmax doesn't mix them in. SDPA / eager backend
    required; FA2 silently drops 4D per-head masks
    (``masking_utils._preprocess_mask_arguments`` short-circuits 4D, but
    flash_attn_func ignores the argument).

    Each ``per_layer_masks[i]`` has shape
    ``[1, H_q, 1, base_len_i + max_new_tokens]`` with -inf on padded key
    positions. The hook slices it to ``[..., :base_len_i + step + 1]`` so
    the new token's K column (auto-appended by DynamicCache) is left
    unmasked.
    """
    eos_set = {int(token_id) for token_id in eos_token_ids if token_id is not None}
    generated_ids: List[int] = []
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
    step_ref = [0]

    def make_hook(idx: int):
        base = int(per_layer_base_lens[idx])
        mask_i = per_layer_masks[idx]

        def hook(module, args, kwargs):
            cur_kv = base + step_ref[0] + 1
            kwargs["attention_mask"] = mask_i[:, :, :, :cur_kv]
            return args, kwargs

        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        for i in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in eos_set:
                break
            generated_ids.append(token_id)
            step_ref[0] = i
            model_kwargs = {
                "input_ids": next_token.to(primary_device),
                "past_key_values": prepare_past_key_values_for_model(past_key_values),
                "use_cache": True,
                "return_dict": True,
            }
            if decode_position_start is not None:
                model_kwargs["position_ids"] = torch.tensor(
                    [[decode_position_start + i]],
                    device=next_token.device,
                    dtype=torch.long,
                )
            with torch.inference_mode():
                outputs = model(**model_kwargs)
            past_key_values = detach_past_key_values(outputs.past_key_values)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()
    return generated_ids



# Per-head streaming diagnostic sinks: appended per layer during prefill, cleared
# at the start of each streaming run. Memory cost is O(num_layers * H_kv * D).
_DIAG_K_BITS_PERHEAD_JOINT: List[torch.Tensor] = []
_DIAG_K_AVG_PERHEAD_JOINT: List[torch.Tensor] = []
_DIAG_K_VALID_LAYER_JOINT: List[bool] = []          # T_eff_total > 0
_DIAG_K_T_EFF_PERHEAD_JOINT: List[torch.Tensor] = []

# Per-(layer, head) visualization sinks. Populated only when RDKV_PERHEAD_VIZ_DIR
# is set (else memory cost is non-trivial: num_layers * H_kv * (T + D) floats).
_DIAG_V_BITS_PERHEAD_JOINT: List[torch.Tensor] = []
_DIAG_V_SCORES_PERHEAD_JOINT: List[torch.Tensor] = []
_DIAG_CHANNEL_SCORES_PERHEAD_JOINT: List[torch.Tensor] = []
_DIAG_K_DYNRANGE_PERHEAD_JOINT: List[torch.Tensor] = []


def _reset_perhead_k_diag_sinks() -> None:
    _DIAG_K_BITS_PERHEAD_JOINT.clear()
    _DIAG_K_AVG_PERHEAD_JOINT.clear()
    _DIAG_K_VALID_LAYER_JOINT.clear()
    _DIAG_K_T_EFF_PERHEAD_JOINT.clear()
    _DIAG_V_BITS_PERHEAD_JOINT.clear()
    _DIAG_V_SCORES_PERHEAD_JOINT.clear()
    _DIAG_CHANNEL_SCORES_PERHEAD_JOINT.clear()
    _DIAG_K_DYNRANGE_PERHEAD_JOINT.clear()


def _allocate_k_bits_per_head(
    channel_scores: torch.Tensor,
    k_avg: Union[float, torch.Tensor],
    *,
    epsilon,
    bit_options,
) -> torch.Tensor:
    """Per-head K bit allocation.

    Args:
        channel_scores: [H_kv, D] float32. Per-head channel importance scores.
        k_avg: either a scalar float (shared across heads, legacy) or a
            1-D tensor of shape [H_kv] (per-head budget targets, set via
            OBKV_K_AVG_MODE=perhead). Scalar path is bit-exact legacy behaviour.
        epsilon, bit_options: knapsack knobs (pass through).

    Returns:
        k_bits: [H_kv, D] long, per-head bit allocation.
    """
    assert channel_scores.dim() == 2, (
        f"_allocate_k_bits_per_head expects [H_kv, D]; got {tuple(channel_scores.shape)}"
    )
    H_kv = int(channel_scores.shape[0])
    if torch.is_tensor(k_avg):
        assert k_avg.dim() == 1 and int(k_avg.shape[0]) == H_kv, (
            f"k_avg tensor must be [H_kv={H_kv}]; got {tuple(k_avg.shape)}"
        )
        # Per-head budget. Pass tensor directly to batched solver.
        k_avg_arg: Union[float, torch.Tensor] = k_avg.detach().float()
    else:
        # Shared scalar budget across all H_kv rows.
        k_avg_arg = float(k_avg)

    # Single batched knapsack call replaces the per-head Python loop.
    return knapsack_bit_allocation_batched(
        channel_scores,
        k_avg_arg,
        epsilon=epsilon,
        bit_options=bit_options,
    )  # [H_kv, D]


def _collapse_k_bits_mode(k_bits_perhead: torch.Tensor) -> torch.Tensor:
    """Collapse per-head k_bits [H_kv, D] to a single shared [D] via mode.

    Used as a Phase-1 bridge to feed downstream pack_k_mixed (still 1D). The
    Phase-2 packing refactor will consume the per-head tensor directly and
    this helper can be removed.

    torch.mode tie-break is implementation-defined but deterministic for fixed
    input; the Phase-1 gate tolerates single-bit-per-channel drift under ties.
    """
    assert k_bits_perhead.dim() == 2
    return k_bits_perhead.mode(dim=0).values.to(torch.long)


def _print_perhead_k_diag(
    all_layer_k_bits: List[torch.Tensor],
    *,
    method: str,
    head_dim: int = 128,
) -> None:
    """Diagnostic print for Phase 1 fast-path dispatch feasibility.

    For each layer, prints per-head (N_2, N_4, N_8) counts, pattern uniqueness,
    exact-padded-tuple uniqueness, and the fraction of heads whose
    (N_2, N_4_pad, N_8) tuple matches any Phase-3 fast-path kernel precondition.
    Also prints aggregate ratios across layers.

    Args:
        all_layer_k_bits: list of per-layer [H_kv, D] long tensors.
        method: label for log lines (e.g. "global", "streaming_joint").
        head_dim: K channel dim D (Llama-3.1-8B: 128).
    """
    D = head_dim

    def _fp_cond(n2, n4_pad, n8):
        uniform4 = (n2 == 0 and n8 == 0 and n4_pad == D)
        mixed24 = (n2 > 0 and n4_pad > 0 and n8 == 0)
        mixed48 = (n2 == 0 and n4_pad > 0 and 0 < n8 <= 16)
        mixed248 = (n2 > 0 and n4_pad > 0 and 0 < n8 <= 16)
        return uniform4 or mixed24 or mixed48 or mixed248

    all_same_pattern_flags: List[bool] = []
    all_exact_same_flags: List[bool] = []
    fp_eligible_ratios: List[float] = []
    unique_patterns_per_layer: List[int] = []

    for l, k_bits_h in enumerate(all_layer_k_bits):
        if k_bits_h is None or k_bits_h.numel() == 0:
            continue
        H_kv = int(k_bits_h.shape[0])
        per_head_counts = []
        for h in range(H_kv):
            b = k_bits_h[h]
            n2 = int((b == 2).sum().item())
            n4 = int((b == 4).sum().item())
            n8 = int((b == 8).sum().item())
            n2_pad = ((n2 + 3) // 4) * 4
            n4_pad = ((n4 + 1) // 2) * 2
            per_head_counts.append((n2, n4, n8, n2_pad, n4_pad))

        patterns = [(c[0] > 0, c[1] > 0, c[2] > 0) for c in per_head_counts]
        unique_pats = len(set(patterns))
        all_same_pattern_l = unique_pats == 1

        padded_tuples = [(c[3], c[4], c[2]) for c in per_head_counts]
        all_exact_l = len(set(padded_tuples)) == 1

        fp_eligible_h = [_fp_cond(c[0], c[4], c[2]) for c in per_head_counts]
        fp_eligible_ratio_l = sum(fp_eligible_h) / max(1, H_kv)

        unique_patterns_per_layer.append(unique_pats)
        all_same_pattern_flags.append(all_same_pattern_l)
        all_exact_same_flags.append(all_exact_l)
        fp_eligible_ratios.append(fp_eligible_ratio_l)

        print(
            f"[PERHEAD_K_DIAG/{method}] layer={l:02d} "
            f"unique_patterns={unique_pats} "
            f"all_same_pattern={all_same_pattern_l} "
            f"all_exact_same={all_exact_l} "
            f"fp_eligible_heads={sum(fp_eligible_h)}/{H_kv}",
            flush=True,
        )

    if all_same_pattern_flags:
        n = len(all_same_pattern_flags)
        print(
            f"[PERHEAD_K_DIAG/{method}] "
            f"all_same_pattern_ratio={sum(all_same_pattern_flags)/n:.3f} "
            f"all_exact_same_ratio={sum(all_exact_same_flags)/n:.3f} "
            f"mean_fp_eligible_head_ratio={sum(fp_eligible_ratios)/n:.3f} "
            f"unique_patterns_per_layer={unique_patterns_per_layer}",
            flush=True,
        )


def _flush_perhead_viz_plots(obs_window: int) -> None:
    """Save per-(layer, head) V/K diagnostic PNGs gated by RDKV_PERHEAD_VIZ_DIR.

    Pairs picked via RDKV_PERHEAD_VIZ_PAIRS (default '0:0,0:1,15:0,15:1,31:0,31:1').
    For each pair (L, H): writes
      ht_colored_by_bits_layer{L}_head{H}.png   (V-side, OBKV V3 analogue)
      bar_channel_detail_layer{L}_head{H}.png   (K-side, OBKV K2 analogue)
    """
    out_dir = os.environ.get("RDKV_PERHEAD_VIZ_DIR", "")
    if not out_dir:
        return
    if not _DIAG_V_BITS_PERHEAD_JOINT:
        print("[viz] no V-side sink data captured; skipping per-head plots", flush=True)
        return

    pairs_env = os.environ.get(
        "RDKV_PERHEAD_VIZ_PAIRS", "0:0,0:1,15:0,15:1,31:0,31:1",
    )
    pairs: List[Tuple[int, int]] = []
    for tok in pairs_env.split(","):
        l_str, h_str = tok.split(":")
        pairs.append((int(l_str), int(h_str)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Bump fonts ~15 pt above default (~10 → 25).
    BIG = 25
    plt.rcParams.update({
        "font.size": BIG,
        "axes.labelsize": BIG,
        "axes.titlesize": BIG,
        "xtick.labelsize": BIG,
        "ytick.labelsize": BIG,
        "legend.fontsize": BIG,
    })

    os.makedirs(out_dir, exist_ok=True)
    BIT_COLOR = {0: "black", 2: "red", 4: "orange", 8: "blue", 16: "green"}
    num_layers = len(_DIAG_V_BITS_PERHEAD_JOINT)
    written = 0

    for L, H in pairs:
        if L >= num_layers:
            print(f"[viz] skip (layer={L}, head={H}): only {num_layers} layers captured")
            continue
        v_scores = _DIAG_V_SCORES_PERHEAD_JOINT[L]            # [H_kv, T]
        v_bits   = _DIAG_V_BITS_PERHEAD_JOINT[L]              # [H_kv, T]
        ch_scores = _DIAG_CHANNEL_SCORES_PERHEAD_JOINT[L]     # [H_kv, D]
        k_bits   = _DIAG_K_BITS_PERHEAD_JOINT[L]              # [H_kv, D]
        H_kv = int(v_scores.shape[0])
        if H >= H_kv:
            print(f"[viz] skip (layer={L}, head={H}): head>=H_kv={H_kv}")
            continue

        # ── V-side: per-token score colored by bit-width ──────────────────
        ht = np.maximum(v_scores[H].numpy().astype(np.float32), 1e-20)
        vb = v_bits[H].numpy()
        T = len(ht)
        positions = np.arange(T)
        fig, ax = plt.subplots(figsize=(max(14, T // 180), 7))
        for b, c in BIT_COLOR.items():
            mask = vb == b
            if mask.any():
                ax.semilogy(positions[mask], ht[mask], ".", color=c,
                            markersize=5, alpha=0.7)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("token score (log scale)")
        fig.tight_layout()
        v_stem = os.path.join(out_dir, f"token_score_colored_by_bits_layer{L}_head{H}")
        fig.savefig(v_stem + ".png", dpi=150)
        fig.savefig(v_stem + ".pdf")
        plt.close(fig)
        written += 2

        # ── K-side: per-channel score colored by bit-width (scatter style) ─
        sigma_c = np.maximum(ch_scores[H].numpy().astype(np.float32), 1e-20)
        kb = k_bits[H].numpy().astype(int)
        D = len(sigma_c)
        x = np.arange(D)
        fig, ax = plt.subplots(figsize=(max(14, D // 6), 7))
        for b, c in BIT_COLOR.items():
            mask = kb == b
            if mask.any():
                ax.semilogy(x[mask], sigma_c[mask], ".", color=c,
                            markersize=14, alpha=0.85)
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("channel score (log scale)")
        fig.tight_layout()
        k_stem = os.path.join(out_dir, f"channel_score_colored_by_bits_layer{L}_head{H}")
        fig.savefig(k_stem + ".png", dpi=150)
        fig.savefig(k_stem + ".pdf")
        plt.close(fig)
        written += 2

    print(f"[viz] wrote {written} per-head plot files (PNG+PDF) to {out_dir}", flush=True)

    # Dump raw per-(layer, head) tensors so future style tweaks can render
    # offline (no GPU prefill rerun). One .pt per (L, H), small (~tens of KB).
    for L, H in pairs:
        if L >= num_layers:
            continue
        H_kv = int(_DIAG_V_BITS_PERHEAD_JOINT[L].shape[0])
        if H >= H_kv:
            continue
        torch.save(
            {
                "layer": L, "head": H, "obs_window": obs_window,
                "v_scores": _DIAG_V_SCORES_PERHEAD_JOINT[L][H].clone(),  # [T]
                "v_bits":   _DIAG_V_BITS_PERHEAD_JOINT[L][H].clone(),    # [T]
                "channel_scores": _DIAG_CHANNEL_SCORES_PERHEAD_JOINT[L][H].clone(),  # [D]
                "k_bits":   _DIAG_K_BITS_PERHEAD_JOINT[L][H].clone(),    # [D]
            },
            os.path.join(out_dir, f"raw_layer{L}_head{H}.pt"),
        )

    # Per-(layer, head) bit distribution for the requested pairs.
    print(f"[viz] per-(layer, head) bit distribution (V then K, candidates 0/2/4/8/16):", flush=True)
    for L, H in pairs:
        if L >= num_layers:
            continue
        H_kv = int(_DIAG_V_BITS_PERHEAD_JOINT[L].shape[0])
        if H >= H_kv:
            continue
        vb = _DIAG_V_BITS_PERHEAD_JOINT[L][H].numpy()
        kb = _DIAG_K_BITS_PERHEAD_JOINT[L][H].numpy()
        v_dist = {b: int((vb == b).sum()) for b in (0, 2, 4, 8, 16)}
        k_dist = {b: int((kb == b).sum()) for b in (0, 2, 4, 8, 16)}
        print(
            f"[viz]   L={L:>2} H={H} | V({len(vb)}t): {v_dist}"
            f"  | K({len(kb)}c): {k_dist}",
            flush=True,
        )


def _score_one_layer(aw, q, o, v, head_dim, v_score_type: str = "attn_linear",
                     k=None, k_score_type: str = "think"):
    """Per-layer (token, channel) score from captured Q/V + attention weights.

    Inputs are 5D with the H_kv axis separated:
      aw: [B, H_kv, q_per_kv, obs_window, T]
      q:  [B, H_kv, q_per_kv, obs_window, head_dim]
      v:  [B, H_kv, T, head_dim]
      k:  [B, H_kv, T, head_dim]                 (required for K-side think scoring)
      o:  [B, H_kv, q_per_kv, obs_window, head_dim]   (kept in signature; unused under default config)

    Returns:
      layer_token:   [B, H_kv, T]            V-side token score, sum_{g,tau} a.
      layer_channel: [B, H_kv, head_dim]     K-side per-channel ThinK score.

    OBKV_SCORE_FP32=1 promotes q/v to fp32 BEFORE the squared / dot ops.
    """
    if v_score_type != "attn_linear":
        raise ValueError(
            f"Unsupported v_score_type={v_score_type!r}; expected 'attn_linear'."
        )
    if k_score_type != "think":
        raise ValueError(
            f"Unsupported k_score_type={k_score_type!r}; expected 'think'."
        )
    if k is None:
        raise ValueError("k_score_type='think' requires the keys tensor (k=...).")
    del o  # unused in the default attn_linear / think config

    if os.environ.get("OBKV_SCORE_FP32", "0") == "1":
        aw = aw.float()
        q = q.float()
        v = v.float()

    # V-side attn_linear: layer_token = sum_{g,tau} a.
    layer_token = aw.sum(dim=(2, 3)).float()

    # K-side ThinK: layer_channel_c = mean_{tau,g}(q_{tau,c}^2) * mean_t(k_{t,c}^2).
    q_norm_per_c = q.float().square().mean(dim=(2, 3))   # [B, H_kv, head_dim]
    k_norm_per_c = k.float().square().mean(dim=2)        # [B, H_kv, head_dim]
    layer_channel = (q_norm_per_c * k_norm_per_c).float()

    return layer_token, layer_channel


def _perhead_select_and_pack_joint(
    K: torch.Tensor,
    V: torch.Tensor,
    layer_scores: torch.Tensor,
    channel_scores: torch.Tensor,
    *,
    gqa_factor: int,
    token_budget: int,
    k_budget_ratio: float,
    obs_window: int,
    pool_kernel_size: int,
    pool_padding: str,
    pool_type: str,
    v_bit_options: Optional[torch.Tensor],
    k_bit_options: Optional[torch.Tensor],
    eps_V: Dict[int, float],
    eps_K: Dict[int, float],
    timing: Optional[Dict[str, float]] = None,
    retain_allocation_metadata: bool = False,
):
    """Joint-knapsack per-head selection + padded packing.

    Each head independently runs a single knapsack over all seq_len tokens
    with bit_options={0,2,4,8,16}. 0=evict, 16=FP16 new zone, {2,4,8}=compressed.
    Heads have varying T_eff_h and N_{2,4,8}^h; tensors are padded to
    max_h(·) so Triton kernels see rectangular inputs, and per-head metadata
    (seg_bounds_per_head, softmax_mask, fp16_gap_mask) zeros out padded
    positions' contributions.

    Args:
        K, V: [1, H_kv, T, D] FP16 layer KV tensors (on device).
        layer_scores: [H_kv, T] per-head token scores (CPU).
        channel_scores: [D] per-layer channel scores (CPU float32).
        token_budget, k_budget_ratio, obs_window, pool_*, bit_options, epsilons:
            knapsack knobs (same semantics as run_obkv).

    Returns (packed_layer, (new_k, new_v), t_eff) where
      - packed_layer: PackedKVLayer with per-head metadata fields populated.
      - (new_k, new_v): padded [1, H_kv, max_n_fp16, D] FP16 new-zone tensors,
        or (None, None) if no head has any V=16 token.
      - t_eff: max_h(T_eff_h), the padded compressed-zone length.
    """
    assert K.dim() == 4 and V.dim() == 4, (
        f"K/V must be [1,H_kv,T,D], got {tuple(K.shape)}/{tuple(V.shape)}"
    )
    assert layer_scores.dim() == 2, (
        f"layer_scores must be [H_kv,T], got {tuple(layer_scores.shape)}"
    )
    H_kv, T = int(layer_scores.shape[0]), int(layer_scores.shape[1])
    head_dim = int(K.shape[3])
    H_q = H_kv * gqa_factor

    def _stage_stamp() -> float:
        if timing is None:
            return 0.0
        if K.device.type == "cuda":
            torch.cuda.synchronize(K.device)
        return time.perf_counter()

    _stage_t0 = _stage_stamp()

    # 1) Per-head pooling (2D input; returns new tensor).
    if pool_kernel_size > 1:
        pooled = apply_score_pooling(
            layer_scores,
            pool_type=pool_type,
            kernel_size=pool_kernel_size,
            padding_mode=pool_padding,
        )
    else:
        pooled = layer_scores.clone()

    pooled_clamped = pooled.float()  # [H_kv, T], keeps device for GPU knapsack
    _stage_after_pool = _stage_stamp()

    # 4) Budget — uniform average-bits target applied to each head's T-token
    #    knapsack. total_budget_bits covers V + K over `token_budget` tokens
    #    at FP16; k_budget_ratio splits them. v_avg = average bits per V
    #    element across the full seq_len (knapsack will pick 0 for low-score
    #    tokens, higher bits for important ones).
    total_budget_bits = 2 * token_budget * head_dim * 16
    v_budget_bits = (1.0 - k_budget_ratio) * total_budget_bits
    v_avg = v_budget_bits / (T * head_dim)

    # 5) Stage 1 — per-head V knapsack via single batched call (replaces
    #    the per-head loop; same result, ~300ms saved per layer at 128K).
    k_budget_bits = k_budget_ratio * total_budget_bits
    v_bits_all = knapsack_bit_allocation_batched(
        pooled_clamped,  # [H_kv, T] float32 on K.device
        v_avg,
        epsilon=eps_V,
        bit_options=v_bit_options,
    )  # [H_kv, T] long in {0,2,4,8,16}
    _stage_after_v_alloc = _stage_stamp()

    per_head_v_bits: List[torch.Tensor] = []
    per_head_kept_ids: List[torch.Tensor] = []
    per_head_kept_v_bits: List[torch.Tensor] = []
    T_eff_total = 0
    for h in range(H_kv):
        v_bits_h = v_bits_all[h]
        keep_mask_h = (v_bits_h > 0)
        kept_ids_h = keep_mask_h.nonzero(as_tuple=True)[0]
        kept_v_bits_h = v_bits_h[kept_ids_h]
        T_eff_total += int(kept_ids_h.numel())

        per_head_v_bits.append(v_bits_h)
        per_head_kept_ids.append(kept_ids_h)
        per_head_kept_v_bits.append(kept_v_bits_h)

    # 6) Stage 2 — Per-head K bit allocation (TriZone / 方案 1). v=16 tokens'
    #    K is now per-channel-quantised into ``packed`` alongside compressed
    #    tokens (8-bit K NMSE ≈ 0.006 is negligible), so the K budget covers
    #    every kept token uniformly. Denominator must be the ACTUAL summed
    #    per-head kept count Σ_h T_eff_h — using token_budget * H_kv inflates
    #    k_avg whenever the V knapsack keeps more low-bit tokens than B.
    k_budget_bits_layer = k_budget_bits * H_kv

    # channel_scores is [H_kv, D] in Phase 1.
    assert channel_scores.dim() == 2, (
        f"_perhead_select_and_pack_joint: channel_scores must be [H_kv, D], "
        f"got {tuple(channel_scores.shape)}"
    )
    H_kv_ch = int(channel_scores.shape[0])

    T_eff_per_head = torch.tensor(
        [int(ids.numel()) for ids in per_head_kept_ids],
        dtype=torch.long,
    )
    # Per-head: clamp(min=1) lets T_eff_h==0 heads land at
    # k_avg_h = k_budget_bits / D >> 16, knapsack saturates to 16; pack_k_mixed
    # then clamps 16->8 internally. That head's packed K is an empty tensor
    # (_empty_packed_layer branch), so k_bits_perhead[h] is cosmetic — decode
    # never consumes it. No explicit fallback needed.
    k_avg_per_head = k_budget_bits / (
        head_dim * T_eff_per_head.clamp(min=1).float()
    )
    k_bits_perhead = _allocate_k_bits_per_head(
        channel_scores.float().to(K.device),
        k_avg_per_head,
        epsilon=eps_K,
        bit_options=k_bit_options,
    )
    _stage_after_k_alloc = _stage_stamp()
    # K-budget log is opt-in: ``OBKV_K_BUDGET_LOG=1`` enables the per-layer
    # min/max print. The ``.item()`` calls force GPU→CPU sync (slow on
    # long-context streaming where they fire 32× per prefill); default
    # off keeps production runs sync-free.
    if os.environ.get("OBKV_K_BUDGET_LOG", "0") == "1":
        k_avg = (
            float(k_avg_per_head.min().item()),
            float(k_avg_per_head.max().item()),
        )
        k_avg_log_str = f"min={k_avg[0]:.3f} max={k_avg[1]:.3f}"
    else:
        k_avg_log_str = None  # not used downstream when log is off
    k_avg_dump = k_avg_per_head.detach()  # GPU tensor; .cpu() deferred

    # Append GPU tensors to diagnostic sinks; the .cpu() happens lazily inside
    # ``_print_perhead_k_diag`` (which is called once per streaming run, after
    # the layer loop, so its syncs don't break per-layer pipelining).
    _DIAG_K_BITS_PERHEAD_JOINT.append(k_bits_perhead.detach())
    _DIAG_K_AVG_PERHEAD_JOINT.append(k_avg_dump)
    _DIAG_K_VALID_LAYER_JOINT.append(bool(T_eff_total > 0))
    _DIAG_K_T_EFF_PERHEAD_JOINT.append(T_eff_per_head.detach())

    if os.environ.get("RDKV_PERHEAD_VIZ_DIR", ""):
        _DIAG_V_BITS_PERHEAD_JOINT.append(
            torch.stack([b.detach().cpu() for b in per_head_v_bits], dim=0)
        )  # [H_kv, T]
        _DIAG_V_SCORES_PERHEAD_JOINT.append(pooled_clamped.detach().cpu())  # [H_kv, T]
        _DIAG_CHANNEL_SCORES_PERHEAD_JOINT.append(channel_scores.detach().cpu())  # [H_kv, D]
        K_view = K[0].float()  # [H_kv, T, D]
        dynrange = (K_view.amax(dim=1) - K_view.amin(dim=1)).cpu()  # [H_kv, D]
        _DIAG_K_DYNRANGE_PERHEAD_JOINT.append(dynrange)

    if os.environ.get("OBKV_K_BUDGET_LOG", "0") == "1":
        print(f"  [K-budget/perhead-joint] H_kv={H_kv} "
              f"T_eff_total={T_eff_total} k_avg={k_avg_log_str}")

    if os.environ.get("OBKV_DIAG_BIT_DIST", "0") == "1":
        v_bits_flat = torch.cat(per_head_v_bits, dim=0)
        v_total = int(v_bits_flat.numel())
        v_dist = {int(b): int((v_bits_flat == b).sum().item()) for b in [0, 2, 4, 8, 16]}
        v_avg_actual = float(v_bits_flat.float().mean().item())
        k_bits_flat = k_bits_perhead.flatten()
        k_total = int(k_bits_flat.numel())
        k_dist = {int(b): int((k_bits_flat == b).sum().item()) for b in [0, 2, 4, 8, 16]}
        k_avg_actual = float(k_bits_flat.float().mean().item())
        v_pct = {b: 100.0 * c / v_total for b, c in v_dist.items()}
        k_pct = {b: 100.0 * c / k_total for b, c in k_dist.items()}
        print(
            f"  [BIT_DIST] V (T={T_eff_total}/{v_total} kept) avg={v_avg_actual:.2f} "
            f"dist%={{0:{v_pct[0]:.1f}, 2:{v_pct[2]:.1f}, 4:{v_pct[4]:.1f}, 8:{v_pct[8]:.1f}, 16:{v_pct[16]:.1f}}} | "
            f"K (D×H={k_total}) avg={k_avg_actual:.2f} "
            f"dist%={{0:{k_pct[0]:.1f}, 2:{k_pct[2]:.1f}, 4:{k_pct[4]:.1f}, 8:{k_pct[8]:.1f}, 16:{k_pct[16]:.1f}}}",
            flush=True,
        )

    # 7) Per-head K channel eviction — zero K[:, h, :, c] where
    # k_bits_perhead[h, c] == 0 (different heads may prune different channels).
    # Always run torch.where (no .item() gate): the gate's ``.any().item()``
    # was a small-work optimisation that forced GPU→CPU sync every layer
    # for negligible savings (mask construction is GPU-side). Unconditional
    # torch.where is no-op when mask is all-zero; the caching allocator
    # reclaims the resulting clone immediately.
    k_evict_mask_ph = (k_bits_perhead == 0).to(K.device)  # [H_kv, D]
    _m4 = k_evict_mask_ph.unsqueeze(0).unsqueeze(2)  # [1, H_kv, 1, D]
    K = torch.where(_m4, torch.zeros_like(K), K)

    # 8) Stage 3 — per-head pack. build_packed_layer handles v=16 split
    #    internally (K folds into packed stream, V returned separately).
    per_head_layers = []
    per_head_v16_V: List[Optional[torch.Tensor]] = []
    legacy_blockwise_path = (
        os.environ.get("OBKV_BLOCKWISE_LEGACY_PATH", "0") == "1"
    )
    if legacy_blockwise_path:
        v_segment_counts = [None] * H_kv
    else:
        v_segment_counts = torch.stack(
            [(v_bits_all == bit).sum(dim=1) for bit in (2, 4, 8)],
            dim=1,
        ).detach().cpu().tolist()
    for h in range(H_kv):
        kept_ids_h = per_head_kept_ids[h].to(K.device)
        kept_v_bits_h = per_head_kept_v_bits[h]

        if kept_ids_h.numel() == 0:
            per_head_layers.append(
                _empty_packed_layer(1, head_dim, K.device, K.dtype)
            )
            per_head_v16_V.append(None)
            continue

        K_h_kept = K[:, h:h + 1, :, :].index_select(2, kept_ids_h).contiguous()
        V_h_kept = V[:, h:h + 1, :, :].index_select(2, kept_ids_h).contiguous()
        # Pass this head's k_bits as a 1D [D] slice; build_packed_layer
        # routes through the legacy pack_k_mixed branch for H_kv=1. The
        # per-head K fields get populated later by
        # _pad_and_stack_per_head_layers from k_bits_per_head.
        k_bits_h = k_bits_perhead[h].to(K.device)
        layer_h, _, v16_V_h = build_packed_layer(
            K_h_kept, V_h_kept, kept_v_bits_h, k_bits_h,
            v_segment_counts=(
                None
                if v_segment_counts[h] is None
                else tuple(map(int, v_segment_counts[h]))
            ),
            defer_k_for_per_head_stack=not legacy_blockwise_path,
        )
        per_head_layers.append(layer_h)
        per_head_v16_V.append(v16_V_h)
    _stage_after_quant = _stage_stamp()

    # 9) Pad + stack with TriZone stripe layout. Returns (packed, new_v_only).
    packed_layer, new_v_only = _pad_and_stack_per_head_layers(
        per_head_layers,
        per_head_v16_V,
        H_q=H_q,
        gqa_factor=gqa_factor,
        head_dim=head_dim,
        device=K.device,
        dtype=K.dtype,
        k_bits_per_head=k_bits_perhead.to(K.device),
        v_bits_per_head=torch.stack(per_head_v_bits, dim=0).to(K.device),
        retain_allocation_metadata=retain_allocation_metadata,
    )
    _stage_after_pack = _stage_stamp()

    if timing is not None:
        timing["pooling_ms"] = timing.get("pooling_ms", 0.0) + (
            _stage_after_pool - _stage_t0
        ) * 1000.0
        timing["v_allocation_ms"] = timing.get("v_allocation_ms", 0.0) + (
            _stage_after_v_alloc - _stage_after_pool
        ) * 1000.0
        timing["k_allocation_ms"] = timing.get("k_allocation_ms", 0.0) + (
            _stage_after_k_alloc - _stage_after_v_alloc
        ) * 1000.0
        timing["quantization_ms"] = timing.get("quantization_ms", 0.0) + (
            _stage_after_quant - _stage_after_k_alloc
        ) * 1000.0
        timing["packing_ms"] = timing.get("packing_ms", 0.0) + (
            _stage_after_pack - _stage_after_quant
        ) * 1000.0

    return packed_layer, new_v_only, packed_layer.T_eff


def _perhead_select_and_pack(
    K: torch.Tensor,
    V: torch.Tensor,
    layer_scores: torch.Tensor,
    channel_scores: torch.Tensor,
    *,
    gqa_factor: int,
    eviction_mode: str = "joint",
    token_budget: int,
    k_budget_ratio: float,
    obs_window: int,
    pool_kernel_size: int,
    pool_padding: str,
    pool_type: str = "avg",
    v_bit_options: Optional[torch.Tensor],
    k_bit_options: Optional[torch.Tensor],
    eps_V: Dict[int, float],
    eps_K: Dict[int, float],
    n_kept_multiplier: float = 5.0,
    timing: Optional[Dict[str, float]] = None,
    retain_allocation_metadata: bool = False,
):
    """Select and pack one layer's KV: per-head joint knapsack over
    {0,2,4,8,16} across all T tokens, with per-head padded packing and
    FP16 new-zone for v_bits==16.
    """
    if eviction_mode != "joint":
        raise ValueError(
            f"Unsupported eviction_mode={eviction_mode!r}; expected 'joint'."
        )
    del n_kept_multiplier  # unused (legacy topk-only parameter, kept in signature)
    return _perhead_select_and_pack_joint(
        K, V, layer_scores, channel_scores,
        gqa_factor=gqa_factor,
        token_budget=token_budget,
        k_budget_ratio=k_budget_ratio,
        obs_window=obs_window,
        pool_kernel_size=pool_kernel_size,
        pool_padding=pool_padding,
        pool_type=pool_type,
        v_bit_options=v_bit_options,
        k_bit_options=k_bit_options,
        eps_V=eps_V,
        eps_K=eps_K,
        timing=timing,
        retain_allocation_metadata=retain_allocation_metadata,
    )


def _get_layer_devices(model: torch.nn.Module):
    """Return (layer_devs, embed_dev, norm_dev, lm_dev, rotary_dev) read from
    real module weight/buffer devices.

    NOT from `hf_device_map`, which stores mixed types and is only a placement
    plan rather than ground truth.
    """
    layer_devs = [layer.self_attn.q_proj.weight.device for layer in model.model.layers]
    embed_dev = model.model.embed_tokens.weight.device
    norm_dev = model.model.norm.weight.device
    lm_dev = model.lm_head.weight.device
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = model.model.layers[0].self_attn.rotary_emb
    rotary_buf = next(rotary_emb.buffers(), None)
    rotary_dev = rotary_buf.device if rotary_buf is not None else embed_dev
    return layer_devs, embed_dev, norm_dev, lm_dev, rotary_dev


def _streaming_prefill_and_pack(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    token_budget: int,
    k_budget_ratio: float,
    epsilon_K: Optional[Dict[int, float]],
    epsilon_V: Optional[Dict[int, float]],
    obs_window: int,
    pool_kernel_size: int,
    pool_padding: str,
    v_bit_options: Optional[torch.Tensor],
    k_bit_options: Optional[torch.Tensor],
    device: torch.device,
    eviction_mode: str = "joint",
    n_kept_multiplier: float = 5.0,
    v_score_type: str = "attn_linear",
    k_score_type: str = "think",
) -> Tuple:
    """Layer-streaming prefill: forward one layer, score, pack, release FP16 KV.

    Returns (dual_cache, next_token_logits, per_layer_t_eff, timing_dict).
    timing_dict keys: t_streaming (embed+loop+norm/lm_head), t_cache_build, T.
    """
    import inspect as _inspect

    def _sync_time():
        torch.cuda.synchronize()
        return time.perf_counter()

    cfg = model.config
    T = int(input_ids.shape[1])
    num_layers = cfg.num_hidden_layers
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    q_per_kv = H_q // H_kv
    D = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H_q)

    eps_V = epsilon_V if epsilon_V is not None else DEFAULT_EPSILON_V
    eps_K = epsilon_K if epsilon_K is not None else DEFAULT_EPSILON_K

    t_start = _sync_time()

    # Device routing for multi-GPU TP (device_map=balanced). Single-GPU:
    # all entries equal `device` and the per-device caches degenerate to one
    # entry — bytes-identical to the pre-patch code path.
    layer_devs, embed_dev, norm_dev, lm_dev, rotary_dev = _get_layer_devices(model)

    rotary_emb_fn = getattr(model.model, "rotary_emb", None)
    if rotary_emb_fn is None:
        rotary_emb_fn = model.model.layers[0].self_attn.rotary_emb
    sig = _inspect.signature(rotary_emb_fn.forward)
    _has_pos_ids = "position_ids" in sig.parameters

    W = min(obs_window, T)

    # Per-device aux cache: lazily build (rope_cos, rope_sin, obs_mask,
    # correct_norm) for each unique device. RoPE compute strategy: call
    # rotary_emb_fn on rotary_dev (the rotary module's native device),
    # then `.to(dev)` the resulting cos/sin. Avoids the hidden mismatch
    # where rotary_emb's internal buffers (e.g. `inv_freq`) live on a
    # different device than where you'd call it.
    _aux_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _aux_for(dev: torch.device):
        cached = _aux_cache.get(dev)
        if cached is not None:
            return cached
        if rotary_dev.type == "cuda":
            torch.cuda.set_device(rotary_dev)
        position_ids_r = torch.arange(T, device=rotary_dev).unsqueeze(0)
        dummy_x = torch.zeros(1, T, D, device=rotary_dev, dtype=model.dtype)
        if _has_pos_ids:
            cos_r, sin_r = rotary_emb_fn(dummy_x, position_ids_r)
        else:
            cos_r, sin_r = rotary_emb_fn(dummy_x, seq_len=T)
            cos_r = cos_r[position_ids_r.squeeze(0)]
            sin_r = sin_r[position_ids_r.squeeze(0)]
        del dummy_x
        if cos_r.device != dev:
            cos_r = cos_r.to(dev)
            sin_r = sin_r.to(dev)
        # Causal obs_mask + correct_norm built directly on dev (pure tensor allocs).
        query_positions = torch.arange(T - W, T, device=dev)
        key_positions = torch.arange(T, device=dev)
        obs_mask_d = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        correct_norm_d = torch.full((T,), float(W), dtype=torch.float32, device=dev)
        tail_idx = torch.arange(W, device=dev)
        correct_norm_d[T - W + tail_idx] = (W - tail_idx).float()
        result = (cos_r, sin_r, obs_mask_d, correct_norm_d)
        _aux_cache[dev] = result
        return result

    packed_layers: List = []
    new_v_only_list: List = []
    per_layer_t_eff: List[int] = []

    # Reset Phase 1 per-head K diagnostic sinks for this streaming run.
    _reset_perhead_k_diag_sinks()

    # Per-segment profiling (guarded by env var — adds sync overhead per step)
    _PROFILE = os.environ.get("OBKV_STREAMING_PROFILE", "0") == "1"
    prof = {"attn": 0.0, "mlp": 0.0, "score": 0.0, "pack": 0.0}

    def _mark():
        if _PROFILE:
            torch.cuda.synchronize()
            return time.perf_counter()
        return 0.0

    with torch.inference_mode():
        if embed_dev.type == "cuda":
            torch.cuda.set_device(embed_dev)
        hidden = model.model.embed_tokens(input_ids.to(embed_dev))  # [1, T, hidden_size]
        if hidden.device != layer_devs[0]:
            hidden = hidden.to(layer_devs[0]).contiguous()

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            dev = layer_devs[layer_idx]
            # Set active CUDA context to this layer's device. Triton kernels
            # (e.g. flash_attn under FlashAttention) require active context
            # to match tensor device. No-op on single-GPU.
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            if hidden.device != dev:
                hidden = hidden.to(dev).contiguous()
            rope_cos, rope_sin, obs_mask, correct_norm = _aux_for(dev)

            # (a) Attention forward
            t0 = _mark()
            residual = hidden
            normed = layer.input_layernorm(hidden)
            bsz, seq_len, _ = normed.size()

            q = layer.self_attn.q_proj(normed)
            k = layer.self_attn.k_proj(normed)
            v = layer.self_attn.v_proj(normed)
            del normed

            q = q.view(bsz, seq_len, H_q, D).transpose(1, 2)    # [1, H_q, T, D]
            k = k.view(bsz, seq_len, H_kv, D).transpose(1, 2)   # [1, H_kv, T, D]
            v = v.view(bsz, seq_len, H_kv, D).transpose(1, 2)   # [1, H_kv, T, D]

            # Qwen3-style per-head q/k RMSNorm over head_dim (after reshape, before RoPE).
            # getattr is no-op for Llama/Mistral where these attributes don't exist.
            q_norm_mod = getattr(layer.self_attn, "q_norm", None)
            k_norm_mod = getattr(layer.self_attn, "k_norm", None)
            if q_norm_mod is not None:
                q = q_norm_mod(q)
            if k_norm_mod is not None:
                k = k_norm_mod(k)

            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

            if HAS_FLASH_ATTN:
                # Native GQA: avoid 2× repeat_kv (saves ~2 GB alloc/free per layer @128K)
                attn_out = flash_attn_func(
                    q.transpose(1, 2),    # [1, T, H_q, D]
                    k.transpose(1, 2),    # [1, T, H_kv, D]
                    v.transpose(1, 2),    # [1, T, H_kv, D]
                    causal=True,
                ).transpose(1, 2)         # [1, H_q, T, D]
            else:
                k_exp = repeat_kv(k, q_per_kv)
                v_exp = repeat_kv(v, q_per_kv)
                attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
                del k_exp, v_exp

            attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
            hidden = layer.self_attn.o_proj(attn_out)
            del attn_out
            hidden = residual + hidden

            # (b) Extract obs-window slice of Q and release full Q before MLP
            q_obs = q[:, :, -W:, :].clone()   # [1, H_q, W, D] — tiny vs full Q
            del q
            t1 = _mark()

            # (c) MLP (gate_up intermediate ~7 GB released by module on return)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            t2 = _mark()

            # (d) Obs-window scoring (K, V still alive)
            k_exp_s = repeat_kv(k, q_per_kv)   # [1, H_q, T, D]
            v_exp_s = repeat_kv(v, q_per_kv)
            scale = 1.0 / math.sqrt(D)
            aw = torch.matmul(q_obs, k_exp_s.transpose(-1, -2)) * scale  # [1, H_q, W, T]
            aw.masked_fill_(obs_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            aw = aw.softmax(dim=-1)
            o_obs = torch.matmul(aw, v_exp_s)   # [1, H_q, W, D]
            del k_exp_s, v_exp_s

            # Reshape to [1, H_kv, q_per_kv, W, ...] for _score_one_layer
            aw_g = aw.view(bsz, H_kv, q_per_kv, W, T)
            q_g = q_obs.view(bsz, H_kv, q_per_kv, W, D)
            o_g = o_obs.view(bsz, H_kv, q_per_kv, W, D)
            del aw, q_obs, o_obs

            layer_token, layer_channel = _score_one_layer(
                aw_g, q_g, o_g, v, D, v_score_type=v_score_type,
                k=k, k_score_type=k_score_type,
            )
            del aw_g, q_g, o_g

            # Normalize and aggregate across batch; keep on GPU for the layer's knapsack.
            layer_token_scores = (layer_token.sum(dim=0) / correct_norm).float()
            channel_scores = layer_channel.sum(dim=0).float()
            del layer_token, layer_channel
            t3 = _mark()

            # (e) Per-head select + pack
            packed_layer, new_v_only, t_eff = _perhead_select_and_pack(
                k, v, layer_token_scores, channel_scores,
                gqa_factor=q_per_kv,
                eviction_mode=eviction_mode,
                token_budget=token_budget,
                k_budget_ratio=k_budget_ratio,
                obs_window=obs_window,
                pool_kernel_size=pool_kernel_size,
                pool_padding=pool_padding,
                pool_type="avg",
                v_bit_options=v_bit_options,
                k_bit_options=k_bit_options,
                eps_V=eps_V,
                eps_K=eps_K,
                n_kept_multiplier=n_kept_multiplier,
            )
            packed_layers.append(packed_layer)
            new_v_only_list.append(new_v_only)
            per_layer_t_eff.append(t_eff)

            # (f) Release this layer's FP16 KV
            del k, v
            t4 = _mark()

            if _PROFILE:
                prof["attn"] += (t1 - t0)
                prof["mlp"] += (t2 - t1)
                prof["score"] += (t3 - t2)
                prof["pack"] += (t4 - t3)

        # Final norm + lm_head on last token only (full-vocab on 128K would OOM)
        if norm_dev.type == "cuda":
            torch.cuda.set_device(norm_dev)
        hidden = model.model.norm(hidden.to(norm_dev))
        if lm_dev.type == "cuda":
            torch.cuda.set_device(lm_dev)
        next_token_hidden = hidden[:, -1:, :].to(lm_dev)
        next_token_logits = model.lm_head(
            next_token_hidden
        ).squeeze(1).to(device)  # [1, vocab]
        del hidden

    # Flush per-head K diagnostic across all streaming layers.
    if _DIAG_K_BITS_PERHEAD_JOINT:
        _print_perhead_k_diag(
            list(_DIAG_K_BITS_PERHEAD_JOINT),
            method="streaming_joint",
            head_dim=int(_DIAG_K_BITS_PERHEAD_JOINT[0].shape[1]),
        )

    # Per-(layer, head) V/K visualization plots, gated by RDKV_PERHEAD_VIZ_DIR.
    _flush_perhead_viz_plots(obs_window=obs_window)

    t_after_streaming = _sync_time()

    # Raise only when every layer has empty packed zone AND no v=16 V.
    def _n_v16(nv: Optional[torch.Tensor]) -> int:
        return 0 if nv is None else int(nv.shape[2])
    if all(
        t == 0 and _n_v16(nv) == 0
        for t, nv in zip(per_layer_t_eff, new_v_only_list)
    ):
        raise RuntimeError(
            "_streaming_prefill_and_pack: every layer kept zero tokens. "
            "Check token_budget."
        )

    dual_cache = TriZoneCache(
        packed=tuple(packed_layers),
        new_v_only=list(new_v_only_list),
        new_both_k=[None] * num_layers,
        new_both_v=[None] * num_layers,
        original_seq_len=T,
    )

    t_after_cache = _sync_time()

    timing_dict = {
        "T": T,
        "t_streaming": t_after_streaming - t_start,
        "t_cache_build": t_after_cache - t_after_streaming,
        "per_layer_t_eff": per_layer_t_eff,
        "next_token_hidden": next_token_hidden,
    }

    if _PROFILE:
        total_layer = sum(prof.values())
        t_stream = t_after_streaming - t_start
        other = t_stream - total_layer   # embed + rope/mask + final_norm + lm_head
        print(
            f"  [PROFILE streaming] T={T} total={t_stream*1000:.0f}ms "
            f"attn={prof['attn']*1000:.0f}ms "
            f"mlp={prof['mlp']*1000:.0f}ms "
            f"score={prof['score']*1000:.0f}ms "
            f"pack={prof['pack']*1000:.0f}ms "
            f"other={other*1000:.0f}ms",
            flush=True,
        )
        print(
            f"  [PROFILE streaming per-layer avg (32 layers)] "
            f"attn={prof['attn']*1000/num_layers:.1f}ms "
            f"mlp={prof['mlp']*1000/num_layers:.1f}ms "
            f"score={prof['score']*1000/num_layers:.1f}ms "
            f"pack={prof['pack']*1000/num_layers:.1f}ms",
            flush=True,
        )
        timing_dict["profile"] = dict(prof)
        timing_dict["profile_other"] = other

    return dual_cache, next_token_logits, per_layer_t_eff, timing_dict


def run_obkv(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    token_budget: int,
    k_budget_ratio: float,
    epsilon_K: Optional[Dict[int, float]],
    epsilon_V: Optional[Dict[int, float]],
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    chunk_size: int = 128,
    obs_window: int = 32,
    pool_kernel_size: int = 7,
    pool_padding: str = "reflect",
    v_bit_options: Optional[torch.Tensor] = None,
    k_bit_options: Optional[torch.Tensor] = None,
    fp16_topk: int = 0,
    device: Optional[torch.device] = None,
    prefill_chunk_size: Optional[int] = None,
    use_cuda_graph: bool = False,
    eviction_mode: str = "joint",
    n_kept_multiplier: float = 5.0,
    v_score_type: str = "attn_linear",
    k_score_type: str = "think",
    logits_observer: Optional[
        Callable[[int, torch.Tensor, int, bool], None]
    ] = None,
    hidden_observer: Optional[
        Callable[[int, torch.Tensor, int, bool], None]
    ] = None,
    decode_blockwise_rdkv: bool = False,
    decode_block_size: int = 64,
    decode_block_budget_tokens: int = 8,
    decode_final_flush: bool = True,
    decode_correctness_check: bool = False,
    runtime_metrics: Optional[Dict[str, object]] = None,
) -> List[int]:
    """Plan-A streaming per-head OBKV inference.

    Same signature as ``run_obkv_perhead``. Forwards one layer at a time,
    immediately scoring and packing before releasing that layer's FP16 KV.
    Peak memory ~25-27 GB at 128K vs ~49 GB for the two-phase path.
    chunk_size and prefill_chunk_size are accepted for API compatibility
    but not used (streaming loop replaces chunked prefill).

    ``eviction_mode`` / ``n_kept_multiplier`` are forwarded to
    ``_streaming_prefill_and_pack`` — see ``run_obkv_perhead`` for semantics.
    """
    if device is None:
        device = resolve_primary_device(model)

    _VERBOSE_TIMING = os.environ.get("OBKV_TIMING", "0") == "1"

    def _sync_time():
        torch.cuda.synchronize()
        return time.perf_counter()

    if fp16_topk > 0:
        print("  [run_obkv] ignoring fp16_topk — "
              "per-head relies on knapsack allocating v_bits=16 when budget allows")

    dual_cache, next_token_logits, per_layer_t_eff, td = _streaming_prefill_and_pack(
        model=model,
        input_ids=input_ids,
        token_budget=token_budget,
        k_budget_ratio=k_budget_ratio,
        epsilon_K=epsilon_K,
        epsilon_V=epsilon_V,
        obs_window=obs_window,
        pool_kernel_size=pool_kernel_size,
        pool_padding=pool_padding,
        v_bit_options=v_bit_options,
        k_bit_options=k_bit_options,
        device=device,
        eviction_mode=eviction_mode,
        n_kept_multiplier=n_kept_multiplier,
        v_score_type=v_score_type,
        k_score_type=k_score_type,
    )

    _use_cg = use_cuda_graph
    if os.environ.get("OBKV_FORCE_CUDA_GRAPH", "0") == "1":
        _use_cg = True
    if os.environ.get("OBKV_DISABLE_CUDA_GRAPH", "0") == "1":
        _use_cg = False

    t_before_decode = _sync_time()
    blockwise_report = None
    # OBKV_DECOMPRESS_DECODE=1 swaps the Triton-kernel decoder for the
    # HF model.forward path via a fp16 dequantise of the TriZoneCache.
    # Used by cpu_offload / multi-device deployments where
    # greedy_decode_fast's pre_extract_weights would OOM (70B-class).
    # Default off → no behaviour change for existing runs.
    if os.environ.get("OBKV_DECOMPRESS_DECODE", "0") == "1":
        if logits_observer is not None or hidden_observer is not None:
            raise NotImplementedError(
                "distribution observers require the production fast RDKV decoder; "
                "OBKV_DECOMPRESS_DECODE=1 is unsupported."
            )
        from obkv_accel.trizone_decompress import unpack_trizone_to_past_kv
        _cfg = model.config
        _head_dim = getattr(_cfg, "head_dim", None) or (
            _cfg.hidden_size // _cfg.num_attention_heads
        )
        _H_q = _cfg.num_attention_heads

        # Multi-device (balanced TP): keep each layer's past_kv on the
        # layer's own device — passing device=None preserves the packed
        # tensors' devices. HF model.forward then routes per-layer
        # correctly. Single-device: route to the caller's primary device.
        _real_devs = {p.device for p in model.parameters() if p.device.type != "meta"}
        _multi_dev = len(_real_devs) > 1
        _decompress_dev = None if _multi_dev else device
        past_kv, per_layer_masks = unpack_trizone_to_past_kv(
            dual_cache, head_dim=_head_dim, device=_decompress_dev,
            return_masks=True, H_q=_H_q, max_new_tokens=max_new_tokens,
        )

        if any(m is not None for m in per_layer_masks):
            # Per-head ragged path: need 4D mask + SDPA backend.
            assert all(m is not None for m in per_layer_masks), (
                "mixed shared / per-head packed layers in one cache — bug in "
                "unpack_trizone_to_past_kv dispatcher or upstream packing"
            )
            per_layer_base_lens = [
                int(pl.T_eff_k if pl.T_eff_k > 0 else pl.T_eff)
                for pl in dual_cache.packed
            ]

            # Snapshot attn impl with (had_attr, value) so we can fully restore.
            _orig_cfg = (
                hasattr(model.config, "_attn_implementation"),
                getattr(model.config, "_attn_implementation", None),
            )
            _orig_layer = [
                (
                    hasattr(layer.self_attn, "_attn_implementation"),
                    getattr(layer.self_attn, "_attn_implementation", None),
                )
                for layer in model.model.layers
            ]
            try:
                model.config._attn_implementation = "sdpa"
                for layer in model.model.layers:
                    layer.self_attn._attn_implementation = "sdpa"
                generated_ids = greedy_decode_pt_with_perhead_masks(
                    model=model,
                    past_key_values=past_kv,
                    next_token_logits=next_token_logits,
                    per_layer_masks=per_layer_masks,
                    per_layer_base_lens=per_layer_base_lens,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    primary_device=device,
                    decode_position_start=td["T"],
                )
            finally:
                had, val = _orig_cfg
                if had:
                    model.config._attn_implementation = val
                else:
                    try:
                        delattr(model.config, "_attn_implementation")
                    except AttributeError:
                        pass
                for layer, (had_l, val_l) in zip(model.model.layers, _orig_layer):
                    if had_l:
                        layer.self_attn._attn_implementation = val_l
                    else:
                        try:
                            delattr(layer.self_attn, "_attn_implementation")
                        except AttributeError:
                            pass
        else:
            # Shared layout: rectangular cache, no per-head mask needed.
            generated_ids = greedy_decode_pt(
                model=model,
                past_key_values=past_kv,
                next_token_logits=next_token_logits,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                primary_device=device,
                decode_position_start=td["T"],
            )
    elif decode_blockwise_rdkv:
        if not decode_final_flush:
            raise ValueError(
                "Blockwise RDKV requires final flush; "
                "decode_final_flush=False is unsupported."
            )
        if logits_observer is not None or hidden_observer is not None:
            raise NotImplementedError(
                "distribution observers are not implemented for blockwise RDKV"
            )
        from obkv_accel.blockwise_decode import (
            BlockwiseRDKVConfig,
            greedy_decode_blockwise,
        )
        generated_ids, blockwise_report = greedy_decode_blockwise(
            model=model,
            dual_cache=dual_cache,
            next_token_logits=next_token_logits,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            primary_device=device,
            decode_position_start=td["T"],
            use_cuda_graph=_use_cg,
            config=BlockwiseRDKVConfig(
                block_size=decode_block_size,
                block_budget_tokens=decode_block_budget_tokens,
                k_budget_ratio=k_budget_ratio,
                obs_window=obs_window,
                pool_kernel_size=pool_kernel_size,
                pool_padding=pool_padding,
                v_bit_options=v_bit_options,
                k_bit_options=k_bit_options,
                epsilon_v=epsilon_V,
                epsilon_k=epsilon_K,
                correctness_check=decode_correctness_check,
            ),
        )
    else:
        generated_ids = greedy_decode_fast(
            model=model,
            dual_cache=dual_cache,
            next_token_logits=next_token_logits,
            next_token_hidden=td["next_token_hidden"],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            primary_device=device,
            decode_position_start=td["T"],
            use_cuda_graph=_use_cg,
            logits_observer=logits_observer,
            hidden_observer=hidden_observer,
        )
    t_after_decode = _sync_time()

    T = td["T"]
    t_streaming = td["t_streaming"]
    t_cache_build = td["t_cache_build"]
    t_decode = t_after_decode - t_before_decode
    mean_t_eff = sum(per_layer_t_eff) / max(1, len(per_layer_t_eff))
    ttft_ms = (t_streaming + t_cache_build) * 1000.0
    n_gen = len(generated_ids)
    print(
        f"  [TIMING streaming] T={T} mean_T_eff={mean_t_eff:.1f} "
        f"n_gen={n_gen} "
        f"streaming={t_streaming*1000:.0f}ms "
        f"cache_build={t_cache_build*1000:.0f}ms "
        f"(TTFT={ttft_ms:.0f}ms)  "
        f"decode={t_decode*1000:.0f}ms",
        flush=True,
    )

    if _VERBOSE_TIMING:
        per_tok_ms = (t_decode / max(1, n_gen)) * 1000.0
        print(f"  [TIMING streaming verbose] per_tok={per_tok_ms:.2f}ms", flush=True)

    if runtime_metrics is not None:
        runtime_metrics.update(
            {
                "prompt_tokens": T,
                "ttft_ms": ttft_ms,
                "prefill_streaming_ms": t_streaming * 1000.0,
                "prefill_cache_build_ms": t_cache_build * 1000.0,
                "generated_tokens": n_gen,
                "decode_wall_time_ms_runner": t_decode * 1000.0,
            }
        )
        if blockwise_report is not None:
            runtime_metrics.update(blockwise_report)
        else:
            _cfg = model.config
            _head_dim = getattr(_cfg, "head_dim", None) or (
                _cfg.hidden_size // _cfg.num_attention_heads
            )
            _h_kv = _cfg.num_key_value_heads
            _layers = _cfg.num_hidden_layers
            _dtype_bytes = next(model.parameters()).element_size()
            _logical_decode_bytes = (
                n_gen * _layers * _h_kv * 2 * _head_dim * _dtype_bytes
            )
            try:
                from obkv_accel.blockwise_decode import cache_physical_bytes
                _prompt_bytes = cache_physical_bytes(dual_cache)
            except Exception:
                _prompt_bytes = None
            runtime_metrics.update(
                {
                    "decode_blockwise_rdkv": False,
                    "avg_tpot_ms": (t_decode / max(1, n_gen)) * 1000.0,
                    "final_flush_time_ms": 0.0,
                    "completion_latency_ms": t_decode * 1000.0,
                    "prompt_compressed_kv_physical_bytes": _prompt_bytes,
                    "decode_cache_peak_fp16_equivalent_tokens": n_gen,
                    "decode_cache_final_fp16_equivalent_tokens": n_gen,
                    "decode_cache_peak_physical_bytes": _logical_decode_bytes,
                    "decode_cache_final_physical_bytes": _logical_decode_bytes,
                    "num_compressed_decode_blocks": 0,
                }
            )
        if torch.cuda.is_available():
            runtime_metrics.update(
                {
                    "peak_gpu_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_gpu_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )

    return generated_ids
