"""
Quantization utilities for SageAttention3 Triton implementation.

FOUR STRATEGIES
---------------
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

4. MXFP4 E2M1 (Blackwell SM120 / B200):
   Block-scaled FP4 quantization for Q and K using the OCP Microscaling format.
   Uses native SM120 MMA instruction:
     mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale.ue8m0
   via tl.dot_scaled(lhs, lhs_scale, 'e2m1', rhs, rhs_scale, 'e2m1').
   Format details:
     - Data:   FP4 E2M1 (1 sign, 2 exponent, 1 mantissa bits)
               Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
               Packed 2 per byte (low nibble first)
     - Scale:  E8M0 (8-bit unsigned exponent, bias=127)
               One scale per group of 32 elements
               Stored as uint8; actual scale = 2^(stored_value - 127)
   K is pre-transposed: stored as [D//2, N_k] so the kernel can load
   [D//2, BLOCK_N] slices without an in-kernel transpose.
   V stays in BF16 for output quality (same as original SageAttention3).
   ~4× FLOP throughput vs BF16 on SM120.

QUANTIZATION FORMULA
--------------------
  scale   = max(|x|) / QMAX              (per token)
  x_quant = round_clamp(x / scale, QMAX)
  x_dequant ≈ x_quant * scale            (reconstruction)

For INT8: QMAX = 127
For FP8 E4M3: QMAX = 448 (max representable value in float8_e4m3fn)
For MXFP4 E2M1: QMAX = 6 (max representable value), block scale per 32 elems

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


# ---------------------------------------------------------------------------
# MXFP4 E2M1 quantization  (Blackwell SM120+ / tl.dot_scaled)
# ---------------------------------------------------------------------------
#
# FP4 E2M1 encoding (4 bits: [sign | exp[1:0] | mant[0]]):
#   Positive magnitudes: 0b000=0, 0b001=0.5, 0b010=1, 0b011=1.5,
#                        0b100=2, 0b101=3,   0b110=4, 0b111=6
#   Negative: set bit 3.  Example: 0b1010 = -1.0
#
# Round-to-nearest boundaries (linear midpoints between adjacent values):
#   0–0.5:   [0,   0.25)  → 0     [0.25, 0.75) → 0.5
#   0.5–1.5: [0.75, 1.25) → 1.0   [1.25, 1.75) → 1.5
#   1.5–3:   [1.75, 2.5)  → 2.0   [2.5,  3.5)  → 3.0
#   3–6:     [3.5,  5.0)  → 4.0   [5.0,  inf)  → 6.0
#
# E8M0 scale: 1 byte per group of 32 elements.
#   stored_value = floor(log2(group_max_abs)) + 127   (unsigned exponent, bias 127)
#   actual_scale = 2^(stored_value - 127)
#   Reconstruction: fp4_value * 2^(stored_value - 127)
#
# Layout after quantization:
#   Q_packed:   [B, H, N_q, D//2]   uint8 — 2 nibbles per byte, low nibble = even elem
#   Q_scales:   [B, H, N_q, D//32]  uint8 — one E8M0 per group-of-32
#   K_T_packed: [B, H_k, D//2, N_k] uint8 — K transposed for kernel load efficiency
#   K_scales:   [B, H_k, N_k, D//32] uint8
#
# In tl.dot_scaled(lhs, lhs_scale, 'e2m1', rhs, rhs_scale, 'e2m1'):
#   lhs [BLOCK_M, D//2],  lhs_scale [BLOCK_M, D//32]   (Q tile)
#   rhs [D//2, BLOCK_N],  rhs_scale [BLOCK_N, D//32]   (K^T tile)
# ---------------------------------------------------------------------------

@triton.jit
def _quant_mxfp4_kernel(
    x_ptr,       # [N_tokens, D]   input (float, any BF16/FP16/FP32)
    p_ptr,       # [N_tokens, D//2] output packed uint8
    s_ptr,       # [N_tokens, D//32] output E8M0 uint8 scales
    stride_xn, stride_xd,
    stride_pn,   # = D//2
    stride_sn,   # = D//32
    N_tokens,
    D:  tl.constexpr,   # head dimension (must be power of 2, ≥ 32)
):
    """
    One program per token (row).  Processes D elements in groups of 32.

    For each group-of-32:
      1. Find max |x|
      2. E8M0 scale = 2^k where k = ceil(log2(max_abs/6))
         This maps max_abs to ≤ 6 (FP4 E2M1 maximum), maximising dynamic range.
      3. Normalise x / scale
      4. Quantise to FP4 E2M1 nibble (round-to-nearest)
      5. Pack 2 nibbles per uint8 byte (low nibble = even index)

    Grid: (N_tokens,)
    """
    # ── Constexpr constants (declared OUTSIDE loop body) ─────────────────────
    LOG2_6: tl.constexpr = 2.5849625007211563  # log2(6), FP4 E2M1 maximum value

    n = tl.program_id(0)

    # ── Iterate over groups of 32 elements ────────────────────────────────────
    for g in tl.static_range(D // 32):
        # ── Load 32 elements in two halves: even and odd indices ───────────────
        # Even indices within the group: g*32 + 0, 2, 4, ..., 30
        # Odd  indices within the group: g*32 + 1, 3, 5, ..., 31
        base   = g * 32
        offs_e = base + tl.arange(0, 16) * 2       # even: [16]
        offs_o = base + tl.arange(0, 16) * 2 + 1   # odd:  [16]

        xe = tl.load(x_ptr + n * stride_xn + offs_e * stride_xd).to(tl.float32)
        xo = tl.load(x_ptr + n * stride_xn + offs_o * stride_xd).to(tl.float32)

        # ── E8M0 scale: smallest power-of-2 ≥ max_abs/6 ──────────────────────
        # k = ceil(log2(max_abs/6)) = -floor(log2_6 - log2_m)
        # This ensures max_abs / 2^k ≤ 6  (stays within FP4 range)
        max_abs    = tl.maximum(tl.max(tl.abs(xe)), tl.max(tl.abs(xo)))
        log2_m     = tl.log2(tl.maximum(max_abs, 1e-30))
        exp_raw    = -tl.floor(LOG2_6 - log2_m)     # ceil(log2(max/6))
        exp_clamp  = tl.clamp(exp_raw, -127.0, 127.0)
        scale      = tl.exp2(exp_clamp)
        # Biased exponent stored as uint8
        exp_biased = (exp_clamp + 127.0).to(tl.uint8)
        tl.store(s_ptr + n * stride_sn + g, exp_biased)

        # ── Normalise ──────────────────────────────────────────────────────────
        xe_n = xe / scale
        xo_n = xo / scale

        # ── Quantise to FP4 E2M1 (round-to-nearest) ───────────────────────────
        # Even elements → low nibble
        sign_e = tl.where(xe_n < 0.0, 8, 0)
        ax_e   = tl.abs(xe_n)
        mag_e  = tl.where(ax_e < 0.25, 0,
                 tl.where(ax_e < 0.75, 1,
                 tl.where(ax_e < 1.25, 2,
                 tl.where(ax_e < 1.75, 3,
                 tl.where(ax_e < 2.5,  4,
                 tl.where(ax_e < 3.5,  5,
                 tl.where(ax_e < 5.0,  6, 7)))))))
        lo = ((sign_e | mag_e) & 0xF).to(tl.uint8)

        # Odd elements → high nibble
        sign_o = tl.where(xo_n < 0.0, 8, 0)
        ax_o   = tl.abs(xo_n)
        mag_o  = tl.where(ax_o < 0.25, 0,
                 tl.where(ax_o < 0.75, 1,
                 tl.where(ax_o < 1.25, 2,
                 tl.where(ax_o < 1.75, 3,
                 tl.where(ax_o < 2.5,  4,
                 tl.where(ax_o < 3.5,  5,
                 tl.where(ax_o < 5.0,  6, 7)))))))
        hi = (((sign_o | mag_o) & 0xF) << 4).to(tl.uint8)
        packed = (lo | hi).to(tl.uint8)       # [16] packed bytes

        # ── Store 16 packed bytes for this group ───────────────────────────────
        offs_p = g * 16 + tl.arange(0, 16)
        tl.store(p_ptr + n * stride_pn + offs_p, packed)


# ---------------------------------------------------------------------------
# Fused: smooth-quant Q centering + MXFP4 quantization in one kernel pass
# ---------------------------------------------------------------------------
# This fused kernel replaces two separate passes (smooth_quant_q + quant_fp4)
# with a single memory-efficient pass over Q.
#
# For a group of GROUP_SIZE tokens [g*GROUP_SIZE : (g+1)*GROUP_SIZE]:
#   1. Compute per-group mean qm[D]
#   2. Subtract mean: q_c = q - qm   (smooth quant centering)
#   3. For each subgroup of 32 elements within D:
#      - E8M0 scale = 2^ceil(log2(max_abs/6))
#      - Quantise q_c to FP4 E2M1 nibbles, pack 2 per byte
# Also writes back qm for the delta_s computation.
#
# Grid: (B * H * G,)  where G = L // GROUP_SIZE
# ---------------------------------------------------------------------------

@triton.jit
def _smooth_quant_then_fp4_kernel(
    q_ptr,     # [B, H, L, D]      input BF16/FP16
    qm_ptr,    # [B, H, G, D]      output: per-group means (for delta_s)
    p_ptr,     # [B, H, L, D//2]   output: packed FP4
    s_ptr,     # [B, H, L, D//32]  output: E8M0 scales
    # Strides for q [B, H, L, D]
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides for qm [B, H, G, D]
    stride_qmb, stride_qmh, stride_qmg, stride_qmd,
    # Strides for p [B, H, L, D//2]
    stride_pb, stride_ph, stride_pl,
    # Strides for s [B, H, L, D//32]
    stride_sb, stride_sh, stride_sl,
    B, H,
    D:          tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # tokens per smooth-quant group (= BLOCK_M = 128)
):
    """
    Grid: (B * H * G,) where G = L // GROUP_SIZE.
    Each program handles one [GROUP_SIZE, D] tile of Q.
    """
    LOG2_6: tl.constexpr = 2.5849625007211563

    pid   = tl.program_id(0)
    G     = tl.cdiv(tl.load(q_ptr).to(tl.int32) * 0 + 1, 1)  # unused, just for shape
    # Decode (b, h, g) from pid
    # We pass G implicitly via the grid size; decode b,h,g via division
    # pid = b * (H * num_groups) + h * num_groups + g
    # Since num_groups is runtime, we pass it as a constexpr via separate arg.
    # Simpler: just use tl.program_id and pass num_groups as constexpr.
    # But since GROUP_SIZE is constexpr and L is runtime, we can't easily derive G.
    # Workaround: pass G as a separate scalar argument.
    pass


# Simpler approach: one program per (b, h, g) with G passed as a regular arg.
@triton.jit
def _smooth_quant_then_fp4_kernel(
    q_ptr,
    qm_ptr,
    p_ptr,
    s_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qmg, stride_qmd,
    stride_pb, stride_ph, stride_pl,
    stride_sb, stride_sh, stride_sl,
    H,   # number of Q heads
    G,   # number of groups per (b, h)  = L // GROUP_SIZE
    D:          tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Grid: (B * H * G,).
    """
    LOG2_6: tl.constexpr = 2.5849625007211563

    pid      = tl.program_id(0)
    b_idx    = pid // (H * G)
    rem      = pid  % (H * G)
    h_idx    = rem  //  G
    g_idx    = rem   %  G

    t0     = g_idx * GROUP_SIZE
    offs_t = t0 + tl.arange(0, GROUP_SIZE)  # token positions in (b, h) sequence
    offs_d = tl.arange(0, D)

    # ── Load Q tile [GROUP_SIZE, D] ───────────────────────────────────────────
    q_base = b_idx * stride_qb + h_idx * stride_qh
    q_ptrs = q_ptr + q_base \
             + offs_t[:, None] * stride_ql \
             + offs_d[None, :]  * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)   # [GROUP_SIZE, D]

    # ── Smooth-quant: per-group mean over tokens ───────────────────────────────
    qm = tl.sum(q, axis=0) / GROUP_SIZE  # [D]

    # Store mean for delta_s computation
    qm_base = b_idx * stride_qmb + h_idx * stride_qmh + g_idx * stride_qmg
    tl.store(qm_ptr + qm_base + offs_d * stride_qmd,
             qm.to(q_ptr.dtype.element_ty))

    # ── Center Q ──────────────────────────────────────────────────────────────
    q_c = q - qm[None, :]   # [GROUP_SIZE, D]

    # ── FP4 quantize each token in this group ─────────────────────────────────
    # For each token t in the group: process D elements in groups of 32.
    for t_local in tl.static_range(GROUP_SIZE):
        t_global = t0 + t_local
        for g_d in tl.static_range(D // 32):
            base_d = g_d * 32
            # Even/odd split for 32-element group
            offs_e = base_d + tl.arange(0, 16) * 2
            offs_o = base_d + tl.arange(0, 16) * 2 + 1

            xe = q_c[t_local, offs_e]   # [16]
            xo = q_c[t_local, offs_o]   # [16]

            # E8M0 scale
            max_abs   = tl.maximum(tl.max(tl.abs(xe)), tl.max(tl.abs(xo)))
            log2_m    = tl.log2(tl.maximum(max_abs, 1e-30))
            exp_raw   = -tl.floor(LOG2_6 - log2_m)
            exp_clamp = tl.clamp(exp_raw, -127.0, 127.0)
            scale     = tl.exp2(exp_clamp)
            exp_biased = (exp_clamp + 127.0).to(tl.uint8)

            # Store scale
            s_addr = s_ptr \
                     + b_idx * stride_sb + h_idx * stride_sh \
                     + t_global * stride_sl + g_d
            tl.store(s_addr, exp_biased)

            # Quantize
            xe_n = xe / scale
            xo_n = xo / scale

            sign_e = tl.where(xe_n < 0.0, 8, 0)
            ax_e   = tl.abs(xe_n)
            mag_e  = tl.where(ax_e < 0.25, 0,
                     tl.where(ax_e < 0.75, 1,
                     tl.where(ax_e < 1.25, 2,
                     tl.where(ax_e < 1.75, 3,
                     tl.where(ax_e < 2.5,  4,
                     tl.where(ax_e < 3.5,  5,
                     tl.where(ax_e < 5.0,  6, 7)))))))
            lo = ((sign_e | mag_e) & 0xF).to(tl.uint8)

            sign_o = tl.where(xo_n < 0.0, 8, 0)
            ax_o   = tl.abs(xo_n)
            mag_o  = tl.where(ax_o < 0.25, 0,
                     tl.where(ax_o < 0.75, 1,
                     tl.where(ax_o < 1.25, 2,
                     tl.where(ax_o < 1.75, 3,
                     tl.where(ax_o < 2.5,  4,
                     tl.where(ax_o < 3.5,  5,
                     tl.where(ax_o < 5.0,  6, 7)))))))
            hi = (((sign_o | mag_o) & 0xF) << 4).to(tl.uint8)
            packed = (lo | hi).to(tl.uint8)

            # Store packed bytes
            p_addr = p_ptr \
                     + b_idx * stride_pb + h_idx * stride_ph \
                     + t_global * stride_pl + g_d * 16
            tl.store(p_addr + tl.arange(0, 16), packed)


def smooth_quant_and_fp4(
    q: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused smooth-quant centering + MXFP4 quantization for Q.

    Single kernel pass: load Q once, compute group mean, center, quantize.

    Args:
        q:          [B, H, L, D]  BF16/FP16, L must be divisible by group_size.
        group_size: smooth-quant group size (default 128 = BLOCK_M).

    Returns:
        q_packed: [B, H, L, D//2]  uint8 packed FP4
        q_scales: [B, H, L, D//32] uint8 E8M0 scales
        qm:       [B, H, G, D]     same dtype as q (for delta_s)
    """
    B, H, L, D = q.shape
    assert L % group_size == 0
    assert D % 32 == 0

    G = L // group_size

    q_packed = torch.empty(B, H, L, D // 2,  dtype=torch.uint8, device=q.device)
    q_scales = torch.empty(B, H, L, D // 32, dtype=torch.uint8, device=q.device)
    qm       = torch.empty(B, H, G, D,       dtype=q.dtype,    device=q.device)

    grid = (B * H * G,)
    _smooth_quant_then_fp4_kernel[grid](
        q, qm, q_packed, q_scales,
        q.stride(0),        q.stride(1),        q.stride(2),        q.stride(3),
        qm.stride(0),       qm.stride(1),       qm.stride(2),       qm.stride(3),
        q_packed.stride(0), q_packed.stride(1), q_packed.stride(2),
        q_scales.stride(0), q_scales.stride(1), q_scales.stride(2),
        H=H, G=G,
        D=D, GROUP_SIZE=group_size,
    )
    return q_packed, q_scales, qm


def quant_mxfp4_per_token(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise tensor to MXFP4 (E2M1 data + E8M0 block scale, group_size=32).

    Compatible with tl.dot_scaled(..., 'e2m1') which maps to the native
    SM120 mma.m16n8k64.kind::mxf4nvf4.block_scale instruction.

    Args:
        x: [*, D]  BF16, FP16, or FP32.  D must be a multiple of 32.

    Returns:
        packed: [*, D//2]  torch.uint8 — 2 FP4 nibbles per byte, low nibble = even
        scales: [*, D//32] torch.uint8 — E8M0 biased exponent per group-of-32
    """
    orig_shape = x.shape
    D          = x.shape[-1]
    assert D % 32 == 0, f"Head dim D={D} must be a multiple of 32 for MXFP4."
    assert (D & (D - 1)) == 0, f"Head dim D={D} must be a power of 2."

    x_flat = x.reshape(-1, D).contiguous()
    N      = x_flat.shape[0]

    packed = torch.empty(N, D // 2,  dtype=torch.uint8,  device=x.device)
    scales = torch.empty(N, D // 32, dtype=torch.uint8,  device=x.device)

    _quant_mxfp4_kernel[(N,)](
        x_flat, packed, scales,
        x_flat.stride(0), x_flat.stride(1),
        packed.stride(0),
        scales.stride(0),
        N_tokens=N,
        D=D,
    )
    return packed.reshape(*orig_shape[:-1], D // 2), \
           scales.reshape(*orig_shape[:-1], D // 32)


def quantise_qkv_fp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Quantise Q and K to MXFP4 for the Blackwell FP4 attention kernel.

    K is transposed after packing so the kernel can load [D//2, BLOCK_N]
    slices without an in-kernel transpose.

    Args:
        q: [B, H,   N_q, D]  BF16/FP16
        k: [B, H_k, N_k, D]  BF16/FP16
        v: [B, H_k, N_k, D]  BF16/FP16  (unchanged — V stays high-precision)

    Returns:
        q_packed:   [B, H,   N_q, D//2]  uint8
        k_T_packed: [B, H_k, D//2, N_k]  uint8  (transposed for kernel)
        v:          [B, H_k, N_k, D]     unchanged
        q_scales:   [B, H,   N_q, D//32] uint8
        k_scales:   [B, H_k, N_k, D//32] uint8
    """
    q_packed, q_scales = quant_mxfp4_per_token(q)   # [B,H,N_q,D//2], [B,H,N_q,D//32]
    k_packed, k_scales = quant_mxfp4_per_token(k)   # [B,H_k,N_k,D//2], [B,H_k,N_k,D//32]
    # Transpose K's packed data from [B, H_k, N_k, D//2] → [B, H_k, D//2, N_k]
    # This is mathematically valid: byte at [n, d_packed] packs K[n, 2*d_packed]
    # and K[n, 2*d_packed+1], so the transposed byte at [d_packed, n] is the
    # correct rhs layout for tl.dot_scaled with rhs_k_pack=True.
    k_T_packed = k_packed.transpose(-2, -1).contiguous()  # [B, H_k, D//2, N_k]
    return q_packed, k_T_packed, v, q_scales, k_scales
