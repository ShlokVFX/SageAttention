"""
triton_sage_attn3 – CUTLASS-free SageAttention3 in pure Triton.

Quick start
-----------
    from triton_sage_attn3 import sageattn3_triton

    out = sageattn3_triton(q, k, v, is_causal=True)          # BF16, any GPU
    out = sageattn3_triton(q, k, v, is_causal=True, quant='fp8')  # H100+
"""

from .api import sageattn3_triton, sageattn3_blackwell_triton
from .attention import sage_attn3_fwd, sage_attn3_fwd_fp8
from .preprocessing import preprocess_qkv, smooth_quant_q, normalize_k, compute_delta_s
from .quantize import (
    quant_fp8_per_token,
    quant_int8_per_token,
    quantise_qkv_fp8,
    quantise_qkv_int8,
)

__version__ = "0.1.0"
__all__ = [
    "sageattn3_triton",
    "sageattn3_blackwell_triton",
    "sage_attn3_fwd",
    "sage_attn3_fwd_fp8",
    "preprocess_qkv",
    "smooth_quant_q",
    "normalize_k",
    "compute_delta_s",
    "quant_fp8_per_token",
    "quant_int8_per_token",
    "quantise_qkv_fp8",
    "quantise_qkv_int8",
]
