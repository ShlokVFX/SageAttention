"""
Quantization utilities for SageAttention3 Triton implementation.

THREE STRATEGIES
----------------
1. None (BF16/FP16):
   No quantization. Highest accuracy, widest GPU compatibility.

2. INT8 (Ampere / A100, RTX 30xx and newer):
   Per-token INT8 quantization for Q and K.
   Saves ~2× memory bandwidth vs BF16 for K/V reads in long-context decode.
   Works on SM80+.

3. FP8 E4M3 (H100 / SM90+):
   Per-token FP8 quantization for Q and K.
   Uses FP8 tensor cores via tl.dot on H100, giving ~2× FLOP throughput.
   V is kept in BF16 for output quality.

QUANTIZATION FORMULA
--------------------
  scale   = max(|x|) / QMAX              (per token)
  x_quant = round_clamp(x / scale, QMAX)
  x_dequant ≈ x_quant * scale            (reconstruction)

For INT8: QMAX = 127
For FP8 E4M3: QMAX = 448 (max representable value in float8_e4m3fn)

SMOOTH QUANT INTERACTION
------------------------
For best accuracy, always run preprocess_qkv() (smooth quant + K normalisation)
BEFORE quantising.  The preprocessing centres Q and K around zero, so the
quantisation scale is as tight as possible.

The delta_s correction is computed from the unquantised centered Q mean and
the original K, so it does not need to be re-run after quantisation.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INT8_MAX  = 127.0
FP8_MAX   = 448.0    # max value of float8_e4m3fn


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------

@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,        # [N_tokens, D]   input (any float dtype)
    xq_ptr,       # [N_tokens, D]   output INT8
    scale_ptr,    # [N_tokens]      output FP32 scales
    stride_xn, stride_xd,
    stride_sqn,
    N_tokens,
    D:      tl.constexpr,   # actual head dim (for masking; BLOCK_D >= D)
    BLOCK_D: tl.constexpr,  # power-of-2 >= D
    INT8_MAX: tl.constexpr,
):
    """
    Each program handles one token (one row of the [N_tokens, D] tensor).
    Computes:  scale = max(|row|) / INT8_MAX
               row_q = round(clamp(row / scale, -INT8_MAX, INT8_MAX))
    """
    n      = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    x = tl.load(
        x_ptr + n * stride_xn + offs_d * stride_xd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # Per-token scale
    abs_max   = tl.max(tl.abs(x))                              # scalar
    scale     = tl.where(abs_max == 0.0, 1.0, abs_max / INT8_MAX)

    # Quantise with round-half-up and clamp to INT8 range.
    # tl.libdevice is unavailable in Triton ≥ 3.x; use floor(x + 0.5) instead.
    xq = tl.clamp(tl.floor(x / scale + 0.5), -INT8_MAX, INT8_MAX).to(tl.int8)

    tl.store(xq_ptr   + n * stride_xn   + offs_d * stride_xd, xq,    mask=d_mask)
    tl.store(scale_ptr + n * stride_sqn,                       scale)


def quant_int8_per_token(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise tensor to INT8 with per-token scales.

    Supports any leading dimensions; quantisation is along the last dim (D).

    Args:
        x: [*, D]  BF16, FP16, or FP32.

    Returns:
        x_int8:  [*, D]   torch.int8
        scales:  [*]      torch.float32, one scale per token.
    """
    orig_shape = x.shape
    D          = x.shape[-1]
    x_flat     = x.reshape(-1, D).contiguous()
    N          = x_flat.shape[0]

    BLOCK_D = triton.next_power_of_2(D)

    x_int8  = torch.empty_like(x_flat, dtype=torch.int8)
    scales  = torch.empty(N, dtype=torch.float32, device=x.device)

    _per_token_quant_int8_kernel[(N,)](
        x_flat, x_int8, scales,
        x_flat.stride(0), x_flat.stride(1),
        scales.stride(0),
        N_tokens=N, D=D, BLOCK_D=BLOCK_D,
        INT8_MAX=INT8_MAX,
    )
    return x_int8.reshape(orig_shape), scales.reshape(orig_shape[:-1])


def dequant_int8(x_int8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct float tensor from INT8 + per-token scales.

    Args:
        x_int8: [*, D]   torch.int8
        scales: [*]      torch.float32

    Returns:
        x_fp32: [*, D]   torch.float32
    """
    return x_int8.float() * scales.unsqueeze(-1)


# ---------------------------------------------------------------------------
# FP8 quantization  (H100 / SM90+)
# ---------------------------------------------------------------------------

@triton.jit
def _per_token_quant_fp8_kernel(
    x_ptr,        # [N_tokens, D]   input
    xq_ptr,       # [N_tokens, D]   output FP8 E4M3
    scale_ptr,    # [N_tokens]      output FP32 scales
    stride_xn, stride_xd,
    stride_sqn,
    N_tokens,
    D:       tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    Identical logic to INT8 kernel but casts to float8e4m3 instead of int8.
    Requires Triton ≥ 2.2 and CUDA SM90+ for FP8 tensor core support.
    """
    n      = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    x = tl.load(
        x_ptr + n * stride_xn + offs_d * stride_xd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    abs_max = tl.max(tl.abs(x))
    scale   = tl.where(abs_max == 0.0, 1.0, abs_max / FP8_MAX)

    # Clamp to FP8 E4M3 range before cast to avoid inf / nan.
    # Triton ≥ 3.x name for NVIDIA FP8 E4M3 is float8e4nv (matches torch.float8_e4m3fn).
    xq = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    tl.store(xq_ptr    + n * stride_xn   + offs_d * stride_xd, xq,    mask=d_mask)
    tl.store(scale_ptr + n * stride_sqn,                        scale)


def quant_fp8_per_token(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise tensor to FP8 E4M3 with per-token scales.

    Requires PyTorch >= 2.1 (for torch.float8_e4m3fn dtype) and
    a CUDA GPU (SM90+ for FP8 tensor core benefit).

    Args:
        x: [*, D]  BF16, FP16, or FP32.

    Returns:
        x_fp8:   [*, D]  torch.float8_e4m3fn
        scales:  [*]     torch.float32
    """
    # Require at least one of the FP8 E4M3 variants
    if not hasattr(torch, "float8_e4m3fn") and not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError(
            "torch.float8_e4m3fn not available. "
            "Upgrade to PyTorch >= 2.1 for FP8 support."
        )

    orig_shape = x.shape
    D          = x.shape[-1]
    x_flat     = x.reshape(-1, D).contiguous()
    N          = x_flat.shape[0]

    BLOCK_D = triton.next_power_of_2(D)

    x_fp8   = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    scales  = torch.empty(N, dtype=torch.float32, device=x.device)

    _per_token_quant_fp8_kernel[(N,)](
        x_flat, x_fp8, scales,
        x_flat.stride(0), x_flat.stride(1),
        scales.stride(0),
        N_tokens=N, D=D, BLOCK_D=BLOCK_D,
        FP8_MAX=FP8_MAX,
    )
    return x_fp8.reshape(orig_shape), scales.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# Convenience: quantise Q, K, V in one call
# ---------------------------------------------------------------------------

def quantise_qkv_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Quantise Q and K to FP8; leave V in original dtype.

    Args:
        q, k: [B, H, N, D]  BF16/FP16 (ideally after preprocess_qkv).
        v:    [B, H, N, D]  BF16/FP16.

    Returns:
        q_fp8:   [B, H, N_q, D]  torch.float8_e4m3fn
        k_fp8:   [B, H, N_k, D]  torch.float8_e4m3fn
        v:       [B, H, N_k, D]  unchanged (BF16/FP16)
        q_scale: [B, H, N_q]     FP32
        k_scale: [B, H, N_k]     FP32
    """
    q_fp8, q_scale = quant_fp8_per_token(q)
    k_fp8, k_scale = quant_fp8_per_token(k)
    return q_fp8, k_fp8, v, q_scale, k_scale


def quantise_qkv_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Quantise Q and K to INT8; leave V in original dtype.

    Returns:
        q_int8:  [B, H, N_q, D]  torch.int8
        k_int8:  [B, H, N_k, D]  torch.int8
        v:       [B, H, N_k, D]  unchanged
        q_scale: [B, H, N_q]     FP32
        k_scale: [B, H, N_k]     FP32
    """
    q_int8, q_scale = quant_int8_per_token(q)
    k_int8, k_scale = quant_int8_per_token(k)
    return q_int8, k_int8, v, q_scale, k_scale
