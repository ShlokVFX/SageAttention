"""
SageAttention3 preprocessing: smooth quantization for Q, normalization for K.

WHY THIS MATTERS
----------------
FP4 / INT8 / FP8 formats have a small absolute range (e.g. FP4 E2M1: [-6, 6]).
If a block of Q tokens has mean 3.0 and std 0.5, the range [2.5, 3.5] uses only
1/6 of the available precision. After mean subtraction the range becomes [-0.5, 0.5],
using the precision far more efficiently.

MATHEMATICAL EQUIVALENCE
------------------------
Let Q_c = Q - qm   (mean-subtracted Q, one mean per 128-token group)

    S_corrected = Q_c @ K^T + delta_s
                = (Q - qm) @ K^T + qm @ K^T
                = Q @ K^T                       (exact in float; approx in quant)

delta_s is cheap: computed once in float, shape [B, H, G, N_k] where G = N_q // 128.
Inside the attention kernel it is just added to each row of QK^T scores.

PIPELINE
--------
  preprocess_qkv(q, k, v)
      ├─ pad_to_multiple(q, k, v, 128)
      ├─ normalize_k(k)           # subtract global sequence mean
      ├─ smooth_quant_q(q, 128)   # subtract per-group mean, return qm
      └─ compute_delta_s(qm, k)   # qm @ K^T  →  [B, H, G, N_k]
"""

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: per-group mean subtraction
# ---------------------------------------------------------------------------

@triton.jit
def _group_mean_kernel(
    q_ptr,          # [B, H, L, D]  input (any float dtype)
    qout_ptr,       # [B, H, L, D]  output: Q - group_mean
    qm_ptr,         # [B, H, G, D]  output: per-group means
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qmg, stride_qmd,
    L,                           # sequence length  (divisible by GROUP_SIZE)
    D:          tl.constexpr,    # head dim  (must be power of 2)
    GROUP_SIZE: tl.constexpr,    # tokens per group (typically 128 = BLOCK_M)
):
    """
    Grid: (B, H, G)  where G = L // GROUP_SIZE.
    Each program loads one [GROUP_SIZE, D] tile of Q, subtracts its mean,
    stores the centered tile back and writes the mean to qm.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)

    t0     = g * GROUP_SIZE
    offs_t = t0 + tl.arange(0, GROUP_SIZE)   # token indices in this group
    offs_d = tl.arange(0, D)                  # head-dim indices

    # ── Load Q[b, h, t0:t0+GROUP_SIZE, :] ──────────────────────────────────
    q_base  = b * stride_qb + h * stride_qh
    q_ptrs  = q_ptr + q_base \
              + offs_t[:, None] * stride_ql \
              + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)   # [GROUP_SIZE, D]  promote for accuracy

    # ── Compute group mean over tokens  →  [D] ──────────────────────────────
    qm = tl.sum(q, axis=0) / GROUP_SIZE

    # ── Store centered Q ────────────────────────────────────────────────────
    tl.store(qout_ptr + q_base
             + offs_t[:, None] * stride_ql
             + offs_d[None, :] * stride_qd,
             (q - qm[None, :]).to(q_ptr.dtype.element_ty))

    # ── Store group mean ────────────────────────────────────────────────────
    qm_base = b * stride_qmb + h * stride_qmh + g * stride_qmg
    tl.store(qm_ptr + qm_base + offs_d * stride_qmd,
             qm.to(qm_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def smooth_quant_q(
    q: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subtract a per-group mean from Q (smooth quantization).

    Each group covers `group_size` consecutive tokens.  With BLOCK_M = 128
    and group_size = 128, one group maps exactly to one attention tile, so
    every kernel program uses a single precomputed correction scalar.

    Args:
        q:          [B, H, L, D]  BF16 or FP16, L must be divisible by group_size.
        group_size: tokens per group (default 128).

    Returns:
        q_centered: [B, H, L, D]  same dtype as q, with group means removed.
        qm:         [B, H, G, D]  same dtype, per-group means (G = L // group_size).
    """
    B, H, L, D = q.shape
    assert L % group_size == 0, (
        f"Sequence length L={L} must be divisible by group_size={group_size}. "
        "Call preprocess_qkv() which pads for you."
    )
    assert (D & (D - 1)) == 0, f"Head dim D={D} must be a power of 2."

    G = L // group_size
    q_centered = torch.empty_like(q)
    qm = torch.empty(B, H, G, D, dtype=q.dtype, device=q.device)

    _group_mean_kernel[(B, H, G)](
        q, q_centered, qm,
        q.stride(0),  q.stride(1),  q.stride(2),  q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        L=L, D=D, GROUP_SIZE=group_size,
    )
    return q_centered, qm


def normalize_k(k: torch.Tensor) -> torch.Tensor:
    """
    Center K by subtracting its global sequence mean.

    This is the K-side analogue of Q smooth quantization: it ensures the K
    distribution is centered around zero before optional quantization.

    Args:
        k: [B, H, N_k, D]

    Returns:
        k_norm: [B, H, N_k, D]  new tensor (k is not modified in-place).
    """
    return k - k.mean(dim=2, keepdim=True)


def compute_delta_s(
    qm: torch.Tensor,
    k:  torch.Tensor,
) -> torch.Tensor:
    """
    Compute the smooth-quantization correction term: delta_s = qm @ K^T.

    In the attention kernel the correction is injected as an additive bias:
        S_corrected[i, j] = (Q_c[i] @ K[j]) + delta_s[group(i), j]
                          ≈ Q[i] @ K[j]

    Args:
        qm: [B, H, G, D]    per-group means from smooth_quant_q.
        k:  [B, H, N_k, D]  key tensor (already padded and normalized).

    Returns:
        delta_s: [B, H, G, N_k]  float32 correction, one row per Q group.
    """
    # Cast to float32 for the correction term so we don't lose precision.
    return torch.matmul(qm.float(), k.float().transpose(-2, -1))


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
    group_size:     int  = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full SageAttention3 preprocessing pipeline.

    Steps
    -----
    1. Pad Q, K, V sequence lengths to a multiple of group_size.
    2. Normalize K  (subtract global sequence mean).
    3. Smooth-quantize Q  (subtract per-group or global mean).
    4. Compute delta_s = qm @ K^T  (float32 correction term).

    Args:
        q, k, v:        [B, H, L, D]  BF16 or FP16.
        per_block_mean: True  → one mean per group_size-token block (recommended).
                        False → single global mean (less accurate but cheaper).
        group_size:     tokens per smooth-quant group; must equal BLOCK_M (128).

    Returns:
        q_smooth: [B, H, L_pad, D]  centered query.
        k_norm:   [B, H, L_pad, D]  centered key.
        v_pad:    [B, H, L_pad, D]  padded value (values unchanged).
        delta_s:  [B, H, G, L_pad]  float32 correction (G = 1 if not per_block_mean).
    """
    def _pad(x: torch.Tensor) -> torch.Tensor:
        L   = x.size(2)
        pad = (group_size - L % group_size) % group_size
        return F.pad(x, (0, 0, 0, pad)).contiguous() if pad else x.contiguous()

    q, k, v = _pad(q), _pad(k), _pad(v)

    # Step 2: center K
    k = normalize_k(k)

    # Step 3: center Q, collect means
    if per_block_mean:
        q, qm = smooth_quant_q(q, group_size=group_size)
    else:
        # Global mean: shape [B, H, 1, D]
        qm = q.mean(dim=2, keepdim=True).to(q.dtype)
        q  = (q.float() - qm.float()).to(q.dtype)

    # Step 4: correction term
    delta_s = compute_delta_s(qm, k)   # [B, H, G, L_pad]

    return q, k, v, delta_s
