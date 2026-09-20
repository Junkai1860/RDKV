"""
Triton fused RoPE kernel for decode-time rotary position embedding.

Fuses the ~12 small CUDA kernels from PyTorch's apply_rotary_pos_emb
into a single Triton kernel launch per layer.  Each program handles
one attention head (q or k), processing all head_dim elements at once.

RoPE format: HALF (Llama standard).
  - Channels [0, d) and [d, 2d) are paired, where d = head_dim // 2.
  - out[i]   = x[i] * cos[i] - x[i+d] * sin[i]
  - out[i+d] = x[i+d] * cos[i] + x[i] * sin[i]

Supports BF16 and FP16 inputs; computation is done in FP32 internally.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 128        # head dimension (Llama-3.1-8B)
HALF_D = D // 2  # = 64

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _fused_rope_kernel(
        # Pointers
        Q_ptr,          # [H_q, D]    bf16/fp16
        K_ptr,          # [H_kv, D]   bf16/fp16
        COS_ptr,        # [HALF_D]    bf16/fp16
        SIN_ptr,        # [HALF_D]    bf16/fp16
        # Strides
        stride_q_h,     # Q stride along head dimension
        stride_k_h,     # K stride along head dimension
        # Constexprs
        NUM_Q_HEADS: tl.constexpr,
        HALF_D_C: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # Select q-head or k-head
        if pid < NUM_Q_HEADS:
            base_ptr = Q_ptr + pid * stride_q_h
        else:
            base_ptr = K_ptr + (pid - NUM_Q_HEADS) * stride_k_h

        offs = tl.arange(0, HALF_D_C)  # [0..63]

        # Load first half and second half of head
        x_lo = tl.load(base_ptr + offs).to(tl.float32)
        x_hi = tl.load(base_ptr + offs + HALF_D_C).to(tl.float32)

        # Load cos/sin (shared across all heads)
        cos_f = tl.load(COS_ptr + offs).to(tl.float32)
        sin_f = tl.load(SIN_ptr + offs).to(tl.float32)

        # RoPE formula (HALF format)
        out_lo = x_lo * cos_f - x_hi * sin_f
        out_hi = x_hi * cos_f + x_lo * sin_f

        # Store back in original dtype
        tl.store(base_ptr + offs, out_lo.to(x_lo.dtype))
        tl.store(base_ptr + offs + HALF_D_C, out_hi.to(x_hi.dtype))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def fused_rope(q, k, cos_half, sin_half):
    """Apply RoPE to q and k in-place using a fused Triton kernel.

    Args:
        q:        [H_q, D]    bf16/fp16 — modified in-place
        k:        [H_kv, D]   bf16/fp16 — modified in-place
        cos_half: [HALF_D]    first 64 unique cos values
        sin_half: [HALF_D]    first 64 unique sin values
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available. Cannot run fused_rope.")

    H_q = q.shape[0]
    H_kv = k.shape[0]
    assert q.shape[1] == D, f"Expected head_dim={D}, got {q.shape[1]}"
    assert k.shape[1] == D, f"Expected head_dim={D}, got {k.shape[1]}"
    assert cos_half.shape[0] == HALF_D
    assert sin_half.shape[0] == HALF_D
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"

    grid = (H_q + H_kv,)

    _fused_rope_kernel[grid](
        q, k, cos_half, sin_half,
        q.stride(0), k.stride(0),
        NUM_Q_HEADS=H_q,
        HALF_D_C=HALF_D,
    )
    return q, k


# ---------------------------------------------------------------------------
# Pure PyTorch reference (for testing)
# ---------------------------------------------------------------------------
def fused_rope_ref(q, k, cos_half, sin_half):
    """Pure PyTorch reference for fused_rope.

    NOT in-place — returns new tensors.  Same numerical result.

    Args:
        q:        [H_q, D]    bf16/fp16
        k:        [H_kv, D]   bf16/fp16
        cos_half: [HALF_D]    first 64 unique cos values
        sin_half: [HALF_D]    first 64 unique sin values

    Returns:
        q_out, k_out: same shape/dtype as inputs
    """
    def _apply(x, c, s):
        d = HALF_D
        x_lo = x[:, :d].float()
        x_hi = x[:, d:].float()
        c = c.float()
        s = s.float()
        out_lo = x_lo * c - x_hi * s
        out_hi = x_hi * c + x_lo * s
        return torch.cat([out_lo, out_hi], dim=-1).to(x.dtype)

    return _apply(q, cos_half, sin_half), _apply(k, cos_half, sin_half)
