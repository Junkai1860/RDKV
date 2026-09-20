"""
Triton fused RMSNorm kernel for decode-time layer normalization.

Fuses the 3-5 small CUDA kernels from PyTorch's _rms_norm chain
(float cast, pow+mean, rsqrt+mul, weight*mul, dtype cast) into a
single Triton kernel launch.  Each program handles one row of the
input tensor, processing all hidden_size elements at once.

RMSNorm formula:
  variance = mean(x^2)
  y = x / sqrt(variance + eps) * weight

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
# Triton kernel
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(
        # Pointers
        X_ptr,          # [N_rows, hidden_size]  bf16/fp16
        W_ptr,          # [hidden_size]           bf16/fp16
        Y_ptr,          # [N_rows, hidden_size]  bf16/fp16 (out-of-place)
        # Strides
        stride_x_row,   # X stride between rows
        stride_y_row,   # Y stride between rows
        # Scalars
        eps,             # float, variance epsilon
        hidden_size,     # int, actual hidden dimension
        # Constexprs
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size

        # Load row of x and weight, upcast to FP32
        x_ptr = X_ptr + row_id * stride_x_row
        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_raw.to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # RMS norm: variance = mean(x^2), rstd = 1/sqrt(variance + eps)
        variance = tl.sum(x * x, axis=0) / hidden_size
        rstd = 1.0 / tl.sqrt(variance + eps)
        y = x * rstd * w

        # Store in original dtype
        y_ptr = Y_ptr + row_id * stride_y_row
        tl.store(y_ptr + offs, y.to(x_raw.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """Triton fused RMSNorm.  Out-of-place: returns a new tensor.

    Args:
        x:      [..., hidden_size]  bf16/fp16, contiguous
        weight: [hidden_size]       bf16/fp16, contiguous
        eps:    float

    Returns:
        y: same shape and dtype as x
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available. Cannot run triton_rms_norm.")

    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"

    hidden_size = weight.shape[0]
    assert x.shape[-1] == hidden_size, (
        f"x.shape[-1]={x.shape[-1]} != hidden_size={hidden_size}"
    )

    # Flatten to 2D for the kernel
    x_2d = x.view(-1, hidden_size)
    N_rows = x_2d.shape[0]
    y_2d = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)

    grid = (N_rows,)
    _rms_norm_kernel[grid](
        x_2d, weight, y_2d,
        x_2d.stride(0), y_2d.stride(0),
        eps, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_2d.view_as(x)


# ---------------------------------------------------------------------------
# Pure PyTorch reference (for testing)
# ---------------------------------------------------------------------------
def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor,
                 eps: float) -> torch.Tensor:
    """Pure PyTorch reference for triton_rms_norm.

    NOT in-place — returns a new tensor.  Same numerical result.

    Args:
        x:      [..., hidden_size]  bf16/fp16
        weight: [hidden_size]       bf16/fp16
        eps:    float

    Returns:
        y: same shape and dtype as x
    """
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)
