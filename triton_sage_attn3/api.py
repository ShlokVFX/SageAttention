"""
SageAttention3 – pure Triton API  (CUTLASS-free, portable).

ENTRY POINT
-----------
    sageattn3_triton(q, k, v, is_causal=False, sm_scale=None,
                     quant='none', per_block_mean=True)

QUANTIZATION MODES
------------------

  quant='none'  (default)
    BF16/FP16 flash attention with smooth quantization preprocessing.
    Compatible with any GPU that supports Triton (Ampere, Hopper, …).
    Highest numerical accuracy.

  quant='fp8'
    FP8 E4M3 quantization for Q and K, V stays in BF16/FP16.
    Requires PyTorch ≥ 2.1 and SM90+ (H100) for FP8 tensor core benefit.
    ~2× FLOP efficiency for QK computation vs BF16.

  quant='int8'  [TODO: wire up INT8 kernel variant]
    INT8 quantization for Q and K, useful for Ampere (A100) where
    INT8 tensor cores provide 2× throughput.  Not yet wired through
    the attention kernel (V still BF16); a separate INT8 kernel PR.

SMOOTH QUANTIZATION (always applied)
--------------------------------------
Regardless of quant mode, we always:
  1. Subtract per-block mean from Q  (reduces quantisation error)
  2. Subtract global mean from K
  3. Compute delta_s = qm @ K^T  (correction injected inside kernel)

For quant='none' this is a no-op mathematically but demonstrates the
pipeline and prepares for future quantisation upgrades.

TRADEOFFS vs CUTLASS / CUDA
----------------------------
┌──────────────┬─────────────────────────┬────────────────────────────┐
│              │ CUTLASS (original)      │ Triton (this impl)         │
├──────────────┼─────────────────────────┼────────────────────────────┤
│ Precision    │ FP4 E2M1  (Blackwell)   │ BF16 / FP8 / INT8          │
│ Portability  │ SM120 only              │ Any Triton-capable GPU     │
│ Throughput   │ Highest (FP4 MMA)       │ ~60–80% of peak            │
│ Memory BW    │ 4-bit for Q, K, V       │ 16-bit (BF16) or 8-bit(FP8)│
│ Complexity   │ CUTE + TMA + Warp groups│ ~600 lines Python/Triton   │
│ Compilability│ Needs CUDA toolkit      │ pip install triton          │
└──────────────┴─────────────────────────┴────────────────────────────┘

OPTIMISATION TIPS (H100 / Hopper)
-----------------------------------
- Use quant='fp8' for best throughput.
- BLOCK_M=128, BLOCK_N=64 is a good starting point; profile with
  triton.testing.do_bench.
- For decode (short Q): set BLOCK_M=64 or even 32 to reduce waste.
- For long contexts: increase BLOCK_N=128 to amortise K load latency.
- num_stages=3 enables async prefetch of K/V into L2 on Hopper.
"""

from typing import Literal, Optional

import torch

from .preprocessing import preprocess_qkv
from .attention import sage_attn3_fwd, sage_attn3_fwd_fp8, sage_attn3_fwd_fp4
from .quantize import quantise_qkv_fp8, quantise_qkv_fp4


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def sageattn3_triton(
    q:              torch.Tensor,
    k:              torch.Tensor,
    v:              torch.Tensor,
    attn_mask:      Optional[torch.Tensor] = None,
    is_causal:      bool                   = False,
    sm_scale:       Optional[float]        = None,
    quant:          Literal["none", "fp8", "fp4"] = "none",
    per_block_mean: bool                   = True,
    group_size:     int                    = 128,
    block_m:        int                    = 128,
    block_n:        int                    = 64,
) -> torch.Tensor:
    """
    SageAttention3 forward pass using pure Triton kernels.

    Computes:
        Attn(Q, K, V) = softmax(Q K^T / sqrt(d) + mask) V

    with optional smooth quantization preprocessing for improved
    numerical range utilization under quantization.

    Args:
        q, k, v:        [B, H, L, D]  BF16 or FP16.
                        For GQA pass H_k < H for k and v.
        attn_mask:      Not yet supported (pass None).
        is_causal:      Apply causal mask (auto-lower triangular).
        sm_scale:       Softmax scale.  Defaults to 1/√D.
        quant:          'none' → BF16 kernel;  'fp8' → FP8 kernel (H100+);
                        'fp4' → MXFP4 E2M1 kernel (SM120 Blackwell, ~4× BF16).
        per_block_mean: True  → per-128-token Q mean (recommended).
                        False → global Q mean (slightly faster preprocess).
        group_size:     Smooth-quant group size.  Must equal block_m (128).
        block_m:        Q tile size.  Default 128.  Must be a power of 2.
        block_n:        K/V tile size.  Default 64.  Must be a power of 2.

    Returns:
        out: [B, H, L, D]  same dtype as q.
    """
    if attn_mask is not None:
        raise NotImplementedError(
            "attn_mask is not yet supported. Use is_causal=True for causal masking."
        )

    orig_L = q.size(2)      # unpadded query sequence length
    orig_Lk = k.size(2)     # unpadded key sequence length
    D      = q.size(-1)

    if sm_scale is None:
        sm_scale = D ** -0.5

    # ── Preprocessing: smooth quant + K normalisation + delta_s ──────────────
    q_pre, k_pre, v_pre, delta_s = preprocess_qkv(
        q, k, v,
        per_block_mean=per_block_mean,
        group_size=group_size,
    )
    # After padding, the padded length may differ from orig_L
    L_pad = q_pre.size(2)

    # ── Route to the right kernel ─────────────────────────────────────────────
    if quant == "none":
        out_pad = sage_attn3_fwd(
            q_pre, k_pre, v_pre,
            softmax_scale=sm_scale,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            block_m=block_m,
            block_n=block_n,
        )

    elif quant == "fp8":
        q_fp8, k_fp8, v_fp8, q_scale, k_scale = quantise_qkv_fp8(
            q_pre, k_pre, v_pre
        )
        out_pad = sage_attn3_fwd_fp8(
            q_fp8, k_fp8, v_fp8,
            q_scale, k_scale,
            softmax_scale=sm_scale,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            block_m=block_m,
            block_n=block_n,
        )

    elif quant == "fp4":
        q_packed, k_T_packed, v_fp4, q_scales, k_scales = quantise_qkv_fp4(
            q_pre, k_pre, v_pre
        )
        out_pad = sage_attn3_fwd_fp4(
            q_packed, k_T_packed, v_fp4,
            q_scales, k_scales,
            softmax_scale=sm_scale,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            block_m=block_m,
            block_n=block_n,
        )

    else:
        raise ValueError(f"Unknown quant mode '{quant}'. Choose 'none', 'fp8', or 'fp4'.")

    # ── Strip sequence padding and return ─────────────────────────────────────
    return out_pad[:, :, :orig_L, :].contiguous()


# ---------------------------------------------------------------------------
# Convenience alias matching the CUDA API signature
# ---------------------------------------------------------------------------

def sageattn3_blackwell_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    per_block_mean: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Drop-in replacement for sageattn3_blackwell() using Triton instead of CUTLASS.

    Accepts the same arguments as the original CUDA API so you can swap
    implementations without changing call sites.
    """
    return sageattn3_triton(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        per_block_mean=per_block_mean,
        **kwargs,
    )
