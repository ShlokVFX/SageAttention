"""
Triton flash-attention kernels for SageAttention3 (CUTLASS-free).

TWO VARIANTS
------------
_fwd_kernel_bf16  – Q, K, V in BF16/FP16. Works on any GPU with Triton support.
_fwd_kernel_fp8   – Q, K pre-quantized to FP8 E4M3 with per-token scales.
                    Uses FP8 tensor cores on H100+ (SM90+).

ALGORITHM  (Flash Attention 2 with online softmax)
---------
For each Q tile [BLOCK_M, D]:
  acc = 0, m_i = -inf, l_i = 0
  for each K/V block [BLOCK_N, D]:
      qk  = Q @ K^T                    # [BLOCK_M, BLOCK_N]
      qk += delta_s                    # smooth-quant correction (broadcast)
      qk *= softmax_scale
      if causal: apply triangular mask
      m_new = max(m_i, row_max(qk))
      p     = exp2(qk * log2e - m_new * log2e)   # use hardware exp2
      alpha = exp2((m_i - m_new) * log2e)         # correction for stale acc
      l_i   = l_i * alpha + row_sum(p)
      acc   = acc * alpha + p @ V
      m_i   = m_new
  out = acc / l_i

GQA (Grouped-Query Attention)
-----------------------------
Pass H_k < H to enable GQA.  KV head index = Q head index // (H // H_k).
This is handled transparently in the kernel via kv_h_idx computation.

BLOCK SIZE GUIDANCE
-------------------
BLOCK_M = 128, BLOCK_N = 64  →  good default for H100/A100 (balanced registers)
BLOCK_M = 64,  BLOCK_N = 64  →  lower register pressure, higher occupancy
BLOCK_M = 128, BLOCK_N = 128 →  higher throughput for large seq (if regs permit)

Use the provided autotune configs or pass explicit sizes.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# BF16 / FP16 kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_bf16(
    # ── Tensor pointers ──────────────────────────────────────────────────────
    Q,          # [B, H,   N_q, D]  BF16/FP16
    K,          # [B, H_k, N_k, D]  BF16/FP16
    V,          # [B, H_k, N_k, D]  BF16/FP16
    DeltaS,     # [B, H,   G,  N_k] FP32  smooth-quant correction (or dummy)
    Out,        # [B, H,   N_q, D]  BF16/FP16 output
    # ── Scalar arguments ─────────────────────────────────────────────────────
    softmax_scale,          # float: 1 / sqrt(D)
    # ── Q strides ────────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qn, stride_qd,
    # ── K strides ────────────────────────────────────────────────────────────
    stride_kb, stride_kh, stride_kn, stride_kd,
    # ── V strides ────────────────────────────────────────────────────────────
    stride_vb, stride_vh, stride_vn, stride_vd,
    # ── Out strides ──────────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_on, stride_od,
    # ── DeltaS strides ───────────────────────────────────────────────────────
    stride_dsb, stride_dsh, stride_dsg, stride_dsn,
    # ── Dimension arguments ───────────────────────────────────────────────────
    N_q,    # int: query sequence length (padded)
    N_k,    # int: key/value sequence length (padded)
    H,      # int: number of query heads
    H_k,    # int: number of key/value heads (H_k <= H; H % H_k == 0)
    # ── Compile-time constants ────────────────────────────────────────────────
    HEAD_DIM:       tl.constexpr,   # power-of-2 head dimension
    BLOCK_M:        tl.constexpr,   # Q tile size (= smooth-quant group size)
    BLOCK_N:        tl.constexpr,   # K/V tile size
    IS_CAUSAL:      tl.constexpr,   # apply causal (lower-triangular) mask
    HAS_DELTA_S:    tl.constexpr,   # inject smooth-quant correction
    PER_BLOCK_MEAN: tl.constexpr,   # delta_s has G groups (True) or 1 (False)
):
    """
    One Triton program handles one Q tile [BLOCK_M, HEAD_DIM] for one (batch, head).

    Grid:  (ceil(N_q / BLOCK_M),  B * H)
    """
    # ── Identify this program ────────────────────────────────────────────────
    m_pid  = tl.program_id(0)   # which Q tile (along sequence)
    bh_pid = tl.program_id(1)   # linearised batch × head index

    b_idx  = bh_pid // H
    h_idx  = bh_pid  % H
    # GQA: K/V heads are shared across groups of Q heads
    kv_h_idx = h_idx // (H // H_k)

    # ── Offsets for this tile ────────────────────────────────────────────────
    m_start = m_pid * BLOCK_M
    offs_m  = m_start + tl.arange(0, BLOCK_M)   # Q token positions [BLOCK_M]
    offs_d  = tl.arange(0, HEAD_DIM)             # head-dim positions [HEAD_DIM]

    # ── Load Q tile: [BLOCK_M, HEAD_DIM] ────────────────────────────────────
    q_base = b_idx * stride_qb + h_idx * stride_qh
    q_ptrs = Q + q_base \
             + offs_m[:, None] * stride_qn \
             + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_M, HEAD_DIM]

    # ── Running accumulators (Flash Attention 2 online softmax) ──────────────
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"),  dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M],               dtype=tl.float32)   # running sum

    # ── Base pointers for K, V, DeltaS ──────────────────────────────────────
    kv_base = b_idx * stride_kb + kv_h_idx * stride_kh
    vv_base = b_idx * stride_vb + kv_h_idx * stride_vh

    # delta_s group: one group per BLOCK_M tile when PER_BLOCK_MEAN, else group 0
    g_idx = m_pid if PER_BLOCK_MEAN else 0
    if HAS_DELTA_S:
        ds_row_base = b_idx * stride_dsb + h_idx * stride_dsh + g_idx * stride_dsg

    # ── Decide how many K/V blocks to visit ─────────────────────────────────
    # In causal mode Q[i] may only attend to K[j] with j <= i.
    # The last token in this tile is at m_start + BLOCK_M - 1, so we
    # visit K blocks up through ceil((m_start + BLOCK_M) / BLOCK_N).
    n_blocks_total = tl.cdiv(N_k, BLOCK_N)
    if IS_CAUSAL:
        n_block_max = tl.minimum(
            n_blocks_total,
            tl.cdiv(m_start + BLOCK_M, BLOCK_N),
        )
    else:
        n_block_max = n_blocks_total

    # log2(e) baked in for the exp2 trick:  exp(x) = exp2(x * log2e)
    # Using hardware exp2 is faster than software exp on modern GPUs.
    LOG2E: tl.constexpr = 1.4426950408889634

    # ── Main loop over K/V blocks ────────────────────────────────────────────
    for n_block in range(n_block_max):
        n_start = n_block * BLOCK_N
        offs_n  = n_start + tl.arange(0, BLOCK_N)   # K/V token positions

        # ── Load K transposed: [HEAD_DIM, BLOCK_N] ──────────────────────────
        # We need Q @ K^T so we load K with head-dim as the "row" dimension.
        # K in memory is [N_k, D] (row-major).  Transposing here avoids a
        # separate kernel launch; access is not perfectly coalesced but
        # K is typically cached after the first pass.
        k_ptrs = K + kv_base \
                 + offs_d[:, None] * stride_kd \
                 + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_k
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [HEAD_DIM, BLOCK_N]

        # ── QK^T: [BLOCK_M, BLOCK_N] ────────────────────────────────────────
        qk = tl.dot(q, k, out_dtype=tl.float32)       # accumulate in fp32

        # ── Smooth-quant correction ──────────────────────────────────────────
        # delta_s[b, h, g, n] = qm[g] · K[n]  (precomputed in float)
        # Adding this recovers the uncentred QK score: Q_c·K + qm·K = Q·K
        if HAS_DELTA_S:
            ds_ptrs = DeltaS + ds_row_base + offs_n * stride_dsn
            ds      = tl.load(ds_ptrs, mask=offs_n < N_k, other=0.0)  # [BLOCK_N]
            qk      = qk + ds[None, :]    # broadcast over BLOCK_M rows

        # ── Softmax temperature ──────────────────────────────────────────────
        qk = qk * softmax_scale

        # ── Out-of-bounds mask (last K block may be padded) ──────────────────
        qk = tl.where(offs_n[None, :] < N_k, qk, float("-inf"))

        # ── Causal mask ──────────────────────────────────────────────────────
        # Q token at position offs_m[i] may only attend to K tokens with
        # position <= offs_m[i].  Positions are 0-indexed.
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        # ── Online softmax update (Flash Attention 2, Appendix B) ────────────
        # Step 1: update running max
        m_ij  = tl.max(qk, axis=1)            # max over BLOCK_N  [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)         # new global max    [BLOCK_M]

        # Step 2: softmax numerator for this block
        # p[i,j] = exp(qk[i,j] - m_new[i])
        p = tl.math.exp2(qk * LOG2E - m_new[:, None] * LOG2E)   # [BLOCK_M, BLOCK_N]

        # Step 3: correction factor for the stale accumulator
        # When m increases the old partial sum and output are too large by alpha.
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)              # [BLOCK_M]

        # Step 4: update running statistics
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # ── Load V: [BLOCK_N, HEAD_DIM] ──────────────────────────────────────
        v_ptrs = V + vv_base \
                 + offs_n[:, None] * stride_vn \
                 + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_k
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)   # [BLOCK_N, HEAD_DIM]

        # ── Accumulate weighted value sum ─────────────────────────────────────
        # O = O * alpha + P @ V    (alpha rescales stale partial output)
        # Cast p to input dtype so tl.dot uses tensor cores (bf16 / fp16 MMA).
        acc = acc * alpha[:, None] \
              + tl.dot(p.to(q.dtype), v, out_dtype=tl.float32)

    # ── Normalise: divide by the softmax denominator ─────────────────────────
    # Guard against fully-masked rows (l_i == 0) → output zero.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc    = acc / safe_l[:, None]

    # ── Write output ─────────────────────────────────────────────────────────
    o_base = b_idx * stride_ob + h_idx * stride_oh
    o_ptrs = Out + o_base \
             + offs_m[:, None] * stride_on \
             + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < N_q
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------
# FP8 kernel  (SM90+ / H100 only)
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_fp8(
    # ── Tensor pointers ──────────────────────────────────────────────────────
    Q,          # [B, H,   N_q, D]  FP8 E4M3 (pre-quantised)
    K,          # [B, H_k, N_k, D]  FP8 E4M3 (pre-quantised)
    V,          # [B, H_k, N_k, D]  BF16/FP16 (kept in higher precision)
    QScale,     # [B, H,   N_q]     FP32 per-token scale for Q
    KScale,     # [B, H_k, N_k]     FP32 per-token scale for K
    DeltaS,     # [B, H,   G,  N_k] FP32  smooth-quant correction (or dummy)
    Out,        # [B, H,   N_q, D]  BF16/FP16
    # ── Scalar arguments ─────────────────────────────────────────────────────
    softmax_scale,
    # ── Q strides ────────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qn, stride_qd,
    # ── K strides ────────────────────────────────────────────────────────────
    stride_kb, stride_kh, stride_kn, stride_kd,
    # ── V strides ────────────────────────────────────────────────────────────
    stride_vb, stride_vh, stride_vn, stride_vd,
    # ── Out strides ──────────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_on, stride_od,
    # ── Scale strides ────────────────────────────────────────────────────────
    stride_qsb, stride_qsh, stride_qsn,          # QScale strides
    stride_ksb, stride_ksh, stride_ksn,          # KScale strides
    # ── DeltaS strides ───────────────────────────────────────────────────────
    stride_dsb, stride_dsh, stride_dsg, stride_dsn,
    # ── Dimensions ───────────────────────────────────────────────────────────
    N_q, N_k, H, H_k,
    # ── Compile-time constants ────────────────────────────────────────────────
    HEAD_DIM:       tl.constexpr,
    BLOCK_M:        tl.constexpr,
    BLOCK_N:        tl.constexpr,
    IS_CAUSAL:      tl.constexpr,
    HAS_DELTA_S:    tl.constexpr,
    PER_BLOCK_MEAN: tl.constexpr,
):
    """
    FP8 variant of the SageAttention3 forward kernel.

    Q and K are pre-quantised to FP8 E4M3 with per-token scales.
    The QK computation uses FP8 tensor cores (on H100), accumulating into FP32.
    V stays in BF16/FP16 to maintain output quality.

    Dequantisation:
        qk_true = qk_fp8_dot * q_scale[i] * k_scale[j]

    where q_scale[i] and k_scale[j] are the per-token scales.
    """
    m_pid  = tl.program_id(0)
    bh_pid = tl.program_id(1)
    b_idx  = bh_pid // H
    h_idx  = bh_pid  % H
    kv_h_idx = h_idx // (H // H_k)

    m_start = m_pid * BLOCK_M
    offs_m  = m_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_DIM)

    # ── Load FP8 Q: [BLOCK_M, HEAD_DIM] ─────────────────────────────────────
    q_base = b_idx * stride_qb + h_idx * stride_qh
    q_ptrs = Q + q_base \
             + offs_m[:, None] * stride_qn \
             + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_q
    q_fp8  = tl.load(q_ptrs, mask=q_mask, other=0.0)   # [BLOCK_M, HEAD_DIM] FP8

    # Per-token Q scales: [BLOCK_M]
    qs_base = b_idx * stride_qsb + h_idx * stride_qsh
    qs_ptrs = QScale + qs_base + offs_m * stride_qsn
    q_scale = tl.load(qs_ptrs, mask=offs_m < N_q, other=1.0)  # [BLOCK_M]

    # ── Accumulators ─────────────────────────────────────────────────────────
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)

    kv_base = b_idx * stride_kb + kv_h_idx * stride_kh
    vv_base = b_idx * stride_vb + kv_h_idx * stride_vh
    ks_base = b_idx * stride_ksb + kv_h_idx * stride_ksh

    g_idx = m_pid if PER_BLOCK_MEAN else 0
    if HAS_DELTA_S:
        ds_row_base = b_idx * stride_dsb + h_idx * stride_dsh + g_idx * stride_dsg

    n_blocks_total = tl.cdiv(N_k, BLOCK_N)
    if IS_CAUSAL:
        n_block_max = tl.minimum(
            n_blocks_total, tl.cdiv(m_start + BLOCK_M, BLOCK_N)
        )
    else:
        n_block_max = n_blocks_total

    LOG2E: tl.constexpr = 1.4426950408889634

    for n_block in range(n_block_max):
        n_start = n_block * BLOCK_N
        offs_n  = n_start + tl.arange(0, BLOCK_N)

        # ── Load FP8 K transposed: [HEAD_DIM, BLOCK_N] ───────────────────────
        k_ptrs = K + kv_base \
                 + offs_d[:, None] * stride_kd \
                 + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_k
        k_fp8  = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [HEAD_DIM, BLOCK_N]

        # Per-token K scales: [BLOCK_N]
        ks_ptrs = KScale + ks_base + offs_n * stride_ksn
        k_scale = tl.load(ks_ptrs, mask=offs_n < N_k, other=1.0)   # [BLOCK_N]

        # ── FP8 QK^T → FP32 accumulator, then dequantise ─────────────────────
        # tl.dot with FP8 inputs uses FP8 tensor cores on H100 (SM90+).
        # The raw dot product treats the stored bytes as FP8 values ∈ [-448, 448].
        # Multiply by per-token scales to recover the original float values.
        qk_raw = tl.dot(q_fp8, k_fp8, out_dtype=tl.float32)   # [BLOCK_M, BLOCK_N]
        qk     = qk_raw * q_scale[:, None] * k_scale[None, :]  # dequantise

        if HAS_DELTA_S:
            ds_ptrs = DeltaS + ds_row_base + offs_n * stride_dsn
            ds      = tl.load(ds_ptrs, mask=offs_n < N_k, other=0.0)
            qk      = qk + ds[None, :]

        qk = qk * softmax_scale
        qk = tl.where(offs_n[None, :] < N_k, qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij  = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p     = tl.math.exp2(qk * LOG2E - m_new[:, None] * LOG2E)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        # V stays in BF16/FP16 for better output quality
        v_ptrs = V + vv_base \
                 + offs_n[:, None] * stride_vn \
                 + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_k
        v      = tl.load(v_ptrs, mask=v_mask, other=0.0)  # [BLOCK_N, HEAD_DIM]

        acc = acc * alpha[:, None] \
              + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc    = acc / safe_l[:, None]

    o_base = b_idx * stride_ob + h_idx * stride_oh
    o_ptrs = Out + o_base \
             + offs_m[:, None] * stride_on \
             + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < N_q
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------
# Python launchers
# ---------------------------------------------------------------------------

# Default autotune configs.  Triton selects the fastest for your GPU.
_BF16_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
]


# ---------------------------------------------------------------------------
# MXFP4 attention kernel  (SM120 Blackwell — uses tl.dot_scaled)
# ---------------------------------------------------------------------------
#
# Q and K are pre-quantised to MXFP4 E2M1 format:
#   Q_packed:   [B, H,   N_q, D//2]  uint8 — packed FP4, 2 per byte
#   Q_scales:   [B, H,   N_q, D//32] uint8 — E8M0 block scale per 32 elements
#   K_T_packed: [B, H_k, D//2, N_k]  uint8 — K transposed, packed FP4
#   K_scales:   [B, H_k, N_k, D//32] uint8 — per-token E8M0 scales
#
# The QK^T matmul uses:
#   tl.dot_scaled(Q_tile [M, D//2], Q_scale_tile [M, D//32], 'e2m1',
#                 K_T_tile [D//2, N], K_scale_tile [N, D//32], 'e2m1')
# which maps to:
#   mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale.ue8m0  (SM120 native)
#
# V stays in BF16 / FP16 (same choice as original SageAttention3 CUDA kernel).
# The PV matmul therefore uses BF16 tensor cores as in the BF16 kernel.
#
# Softmax and accumulation are identical to the BF16 kernel (FP32 precision).
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_fp4(
    # ── FP4 Q tensors ────────────────────────────────────────────────────────
    QP,         # [B, H,   N_q, D//2]  uint8  packed FP4
    QS,         # [B, H,   N_q, D//32] uint8  E8M0 scales
    # ── FP4 K tensors (K is pre-transposed) ──────────────────────────────────
    KTP,        # [B, H_k, D//2, N_k]  uint8  packed FP4 (transposed)
    KS,         # [B, H_k, N_k, D//32] uint8  E8M0 scales (per-token)
    # ── BF16/FP16 V tensor ───────────────────────────────────────────────────
    V,          # [B, H_k, N_k, D]     BF16/FP16
    # ── Smooth-quant correction ───────────────────────────────────────────────
    DeltaS,     # [B, H,   G,  N_k]   FP32  (or dummy)
    # ── Output ───────────────────────────────────────────────────────────────
    Out,        # [B, H,   N_q, D]     BF16/FP16
    # ── Scalar ───────────────────────────────────────────────────────────────
    softmax_scale,
    # ── QP strides [B, H, N_q, D//2] ─────────────────────────────────────────
    stride_qpb, stride_qph, stride_qpn, stride_qpd,
    # ── QS strides [B, H, N_q, D//32] ───────────────────────────────────────
    stride_qsb, stride_qsh, stride_qsn, stride_qsd,
    # ── KTP strides [B, H_k, D//2, N_k] ─────────────────────────────────────
    stride_ktb, stride_kth, stride_ktd, stride_ktn,
    # ── KS strides [B, H_k, N_k, D//32] ─────────────────────────────────────
    stride_ksb, stride_ksh, stride_ksn, stride_ksd,
    # ── V strides [B, H_k, N_k, D] ───────────────────────────────────────────
    stride_vb, stride_vh, stride_vn, stride_vd,
    # ── Out strides [B, H, N_q, D] ───────────────────────────────────────────
    stride_ob, stride_oh, stride_on, stride_od,
    # ── DeltaS strides ───────────────────────────────────────────────────────
    stride_dsb, stride_dsh, stride_dsg, stride_dsn,
    # ── Dimensions ───────────────────────────────────────────────────────────
    N_q, N_k, H, H_k,
    # ── Compile-time constants ────────────────────────────────────────────────
    HEAD_DIM:       tl.constexpr,   # D (full, not packed)
    BLOCK_M:        tl.constexpr,
    BLOCK_N:        tl.constexpr,
    IS_CAUSAL:      tl.constexpr,
    HAS_DELTA_S:    tl.constexpr,
    PER_BLOCK_MEAN: tl.constexpr,
):
    """
    FP4 (MXFP4 E2M1) flash-attention forward kernel for SM120 Blackwell.

    Grid: (ceil(N_q / BLOCK_M),  B * H)
    """
    # ── Identify this program ─────────────────────────────────────────────────
    m_pid  = tl.program_id(0)
    bh_pid = tl.program_id(1)
    b_idx  = bh_pid // H
    h_idx  = bh_pid  % H
    kv_h_idx = h_idx // (H // H_k)

    m_start  = m_pid * BLOCK_M
    offs_m   = m_start + tl.arange(0, BLOCK_M)
    offs_d   = tl.arange(0, HEAD_DIM)
    offs_dp  = tl.arange(0, HEAD_DIM // 2)   # packed D (uint8)
    offs_ds  = tl.arange(0, HEAD_DIM // 32)  # scale groups

    # ── Load FP4 Q tile ───────────────────────────────────────────────────────
    qp_base = b_idx * stride_qpb + h_idx * stride_qph
    qp_ptrs = QP + qp_base \
              + offs_m[:, None] * stride_qpn \
              + offs_dp[None, :] * stride_qpd
    q_mask  = offs_m[:, None] < N_q
    q_packed = tl.load(qp_ptrs, mask=q_mask, other=0)   # [BLOCK_M, D//2] uint8

    qs_base = b_idx * stride_qsb + h_idx * stride_qsh
    qs_ptrs = QS + qs_base \
              + offs_m[:, None] * stride_qsn \
              + offs_ds[None, :] * stride_qsd
    q_scales = tl.load(qs_ptrs, mask=offs_m[:, None] < N_q, other=0)  # [BLOCK_M, D//32] uint8

    # ── Accumulators (same online softmax as BF16 kernel) ─────────────────────
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)

    # ── Base pointers ─────────────────────────────────────────────────────────
    kv_base_T = b_idx * stride_ktb + kv_h_idx * stride_kth   # K_T_packed
    ks_base   = b_idx * stride_ksb + kv_h_idx * stride_ksh   # K_scales
    vv_base   = b_idx * stride_vb  + kv_h_idx * stride_vh    # V

    g_idx = m_pid if PER_BLOCK_MEAN else 0
    if HAS_DELTA_S:
        ds_row_base = b_idx * stride_dsb + h_idx * stride_dsh + g_idx * stride_dsg

    n_blocks_total = tl.cdiv(N_k, BLOCK_N)
    if IS_CAUSAL:
        n_block_max = tl.minimum(n_blocks_total, tl.cdiv(m_start + BLOCK_M, BLOCK_N))
    else:
        n_block_max = n_blocks_total

    LOG2E: tl.constexpr = 1.4426950408889634

    # ── Main loop over K/V blocks ─────────────────────────────────────────────
    for n_block in range(n_block_max):
        n_start = n_block * BLOCK_N
        offs_n  = n_start + tl.arange(0, BLOCK_N)

        # ── Load K_T_packed: [D//2, BLOCK_N] ─────────────────────────────────
        # KTP layout: [B, H_k, D//2, N_k]  strides: ..., stride_ktd, stride_ktn=1
        # This load is memory-coalesced because N_k is the innermost dim (stride=1).
        ktp_ptrs = KTP + kv_base_T \
                   + offs_dp[:, None] * stride_ktd \
                   + offs_n[None, :]  * stride_ktn
        k_mask   = offs_n[None, :] < N_k
        k_T_packed = tl.load(ktp_ptrs, mask=k_mask, other=0)  # [D//2, BLOCK_N] uint8

        # ── Load K_scales: [BLOCK_N, D//32] ──────────────────────────────────
        # KS layout: [B, H_k, N_k, D//32]
        ks_ptrs  = KS + ks_base \
                   + offs_n[:, None]  * stride_ksn \
                   + offs_ds[None, :] * stride_ksd
        k_scales = tl.load(ks_ptrs, mask=offs_n[:, None] < N_k, other=0)  # [BLOCK_N, D//32] uint8

        # ── FP4 QK^T via tl.dot_scaled → float32 output ──────────────────────
        # lhs: Q_packed   [BLOCK_M, D//2],  lhs_scale: Q_scales   [BLOCK_M, D//32]
        # rhs: K_T_packed [D//2, BLOCK_N],  rhs_scale: K_scales   [BLOCK_N, D//32]
        # tl.dot_scaled internally applies scales and returns dequantised float32.
        # On SM120: maps to mma.m16n8k64.kind::mxf4nvf4.block_scale.ue8m0
        qk = tl.dot_scaled(
            q_packed, q_scales, 'e2m1',
            k_T_packed, k_scales, 'e2m1',
            out_dtype=tl.float32,
        )   # [BLOCK_M, BLOCK_N]  float32, already dequantised

        # ── Smooth-quant delta_s correction ───────────────────────────────────
        if HAS_DELTA_S:
            ds_ptrs = DeltaS + ds_row_base + offs_n * stride_dsn
            ds      = tl.load(ds_ptrs, mask=offs_n < N_k, other=0.0)
            qk      = qk + ds[None, :]

        # ── Softmax temperature + masking ─────────────────────────────────────
        qk = qk * softmax_scale
        qk = tl.where(offs_n[None, :] < N_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        # ── Online softmax (Flash Attention 2) ────────────────────────────────
        m_ij  = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p     = tl.math.exp2(qk * LOG2E - m_new[:, None] * LOG2E)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        # ── Load V (BF16/FP16) and accumulate ────────────────────────────────
        v_ptrs = V + vv_base \
                 + offs_n[:, None] * stride_vn \
                 + offs_d[None, :]  * stride_vd
        v_mask = offs_n[:, None] < N_k
        v      = tl.load(v_ptrs, mask=v_mask, other=0.0)  # [BLOCK_N, HEAD_DIM]

        acc = acc * alpha[:, None] \
              + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

    # ── Normalise and write output ─────────────────────────────────────────────
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc    = acc / safe_l[:, None]

    o_base = b_idx * stride_ob + h_idx * stride_oh
    o_ptrs = Out + o_base \
             + offs_m[:, None] * stride_on \
             + offs_d[None, :]  * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_q)


def sage_attn3_fwd_fp4(
    q_packed:   torch.Tensor,
    k_T_packed: torch.Tensor,
    v:          torch.Tensor,
    q_scales:   torch.Tensor,
    k_scales:   torch.Tensor,
    delta_s:    Optional[torch.Tensor] = None,
    softmax_scale: Optional[float]    = None,
    is_causal:     bool               = False,
    per_block_mean: bool              = True,
    block_m:   int = 128,
    block_n:   int = 64,
) -> torch.Tensor:
    """
    Launch the MXFP4 SageAttention3 forward pass (SM120 Blackwell).

    Uses tl.dot_scaled with 'e2m1' format which lowers to the native
    mma.m16n8k64.kind::mxf4nvf4.block_scale instruction on SM120.

    Args:
        q_packed:   [B, H,   N_q, D//2]  torch.uint8  FP4-packed Q
        k_T_packed: [B, H_k, D//2, N_k]  torch.uint8  FP4-packed K (transposed)
        v:          [B, H_k, N_k, D]     BF16/FP16
        q_scales:   [B, H,   N_q, D//32] torch.uint8  E8M0 scales
        k_scales:   [B, H_k, N_k, D//32] torch.uint8  E8M0 scales
        delta_s:    [B, H,   G,   N_k]   FP32 smooth-quant correction (None → skip)
        softmax_scale: 1/sqrt(D) if None
        is_causal:  apply causal mask
        per_block_mean: delta_s group mode
        block_m, block_n: tile sizes

    Returns:
        out: [B, H, N_q, D]  same dtype as v
    """
    B, H, N_q, D_packed = q_packed.shape
    D    = D_packed * 2
    H_k  = k_T_packed.shape[1]
    N_k  = k_T_packed.shape[3]   # [B, H_k, D//2, N_k]

    assert H % H_k == 0
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    has_delta_s = delta_s is not None
    out = torch.empty(B, H, N_q, D, dtype=v.dtype, device=v.device)

    grid = (triton.cdiv(N_q, block_m), B * H)

    ds_ptr     = delta_s if has_delta_s else q_packed
    ds_strides = (
        delta_s.stride(0), delta_s.stride(1),
        delta_s.stride(2), delta_s.stride(3),
    ) if has_delta_s else (0, 0, 0, 0)

    _fwd_kernel_fp4[grid](
        q_packed, q_scales, k_T_packed, k_scales, v, ds_ptr, out,
        softmax_scale,
        # QP strides
        q_packed.stride(0),   q_packed.stride(1),   q_packed.stride(2),   q_packed.stride(3),
        # QS strides
        q_scales.stride(0),   q_scales.stride(1),   q_scales.stride(2),   q_scales.stride(3),
        # KTP strides
        k_T_packed.stride(0), k_T_packed.stride(1), k_T_packed.stride(2), k_T_packed.stride(3),
        # KS strides
        k_scales.stride(0),   k_scales.stride(1),   k_scales.stride(2),   k_scales.stride(3),
        # V strides
        v.stride(0),          v.stride(1),           v.stride(2),          v.stride(3),
        # Out strides
        out.stride(0),        out.stride(1),          out.stride(2),        out.stride(3),
        # DeltaS strides
        *ds_strides,
        N_q, N_k, H, H_k,
        HEAD_DIM       = D,
        BLOCK_M        = block_m,
        BLOCK_N        = block_n,
        IS_CAUSAL      = is_causal,
        HAS_DELTA_S    = has_delta_s,
        PER_BLOCK_MEAN = per_block_mean,
        num_warps      = 4,
        num_stages     = 2,   # 2 > 3 on SM120 for this kernel (measured)
    )
    return out


def sage_attn3_fwd(
    q:         torch.Tensor,
    k:         torch.Tensor,
    v:         torch.Tensor,
    delta_s:   Optional[torch.Tensor] = None,
    softmax_scale: Optional[float]    = None,
    is_causal:     bool               = False,
    per_block_mean: bool              = True,
    block_m:   int = 128,
    block_n:   int = 64,
) -> torch.Tensor:
    """
    Launch the BF16/FP16 SageAttention3 forward pass.

    Args:
        q:              [B, H,   N_q, D]  BF16 or FP16 query (optionally smooth-quantised).
        k:              [B, H_k, N_k, D]  BF16 or FP16 key   (optionally normalised).
        v:              [B, H_k, N_k, D]  BF16 or FP16 value.
        delta_s:        [B, H,   G, N_k]  FP32 correction term (None → skip).
        softmax_scale:  1/sqrt(D) if None.
        is_causal:      Apply causal (lower-triangular) mask.
        per_block_mean: delta_s has G=N_q//128 groups (True) or G=1 (False).
        block_m:        Q tile size (default 128).  Must equal smooth-quant group_size.
        block_n:        K/V tile size (default 64).

    Returns:
        out: [B, H, N_q, D]  same dtype as q.
    """
    B, H, N_q, D = q.shape
    H_k = k.shape[1]
    N_k = k.shape[2]

    assert D == k.shape[3] == v.shape[3], "Head dimensions must match."
    assert H % H_k == 0, f"H={H} must be divisible by H_k={H_k}."
    assert (D & (D - 1)) == 0, f"Head dim D={D} must be a power of 2."

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    has_delta_s = delta_s is not None

    out = torch.empty_like(q)

    grid = (triton.cdiv(N_q, block_m), B * H)

    # When HAS_DELTA_S=False the DeltaS pointer is never dereferenced;
    # we pass q as a safe dummy to satisfy Triton's pointer requirements.
    ds_ptr      = delta_s if has_delta_s else q
    ds_strides  = (
        delta_s.stride(0), delta_s.stride(1),
        delta_s.stride(2), delta_s.stride(3)
    ) if has_delta_s else (0, 0, 0, 0)

    _fwd_kernel_bf16[grid](
        q, k, v, ds_ptr, out,
        softmax_scale,
        q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
        k.stride(0),   k.stride(1),   k.stride(2),   k.stride(3),
        v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        *ds_strides,
        N_q, N_k, H, H_k,
        HEAD_DIM       = D,
        BLOCK_M        = block_m,
        BLOCK_N        = block_n,
        IS_CAUSAL      = is_causal,
        HAS_DELTA_S    = has_delta_s,
        PER_BLOCK_MEAN = per_block_mean,
        num_warps      = 4,
        num_stages     = 3,
    )
    return out


def sage_attn3_fwd_fp8(
    q_fp8:      torch.Tensor,
    k_fp8:      torch.Tensor,
    v:          torch.Tensor,
    q_scale:    torch.Tensor,
    k_scale:    torch.Tensor,
    delta_s:    Optional[torch.Tensor] = None,
    softmax_scale: Optional[float]    = None,
    is_causal:     bool               = False,
    per_block_mean: bool              = True,
    block_m:   int = 128,
    block_n:   int = 64,
) -> torch.Tensor:
    """
    Launch the FP8 SageAttention3 forward pass (requires H100 / SM90+).

    Q and K must be pre-quantised to FP8 E4M3 with per-token float32 scales.
    Use quantize.quant_fp8_per_token() to produce these inputs.

    Args:
        q_fp8:   [B, H,   N_q, D]  torch.float8_e4m3fn
        k_fp8:   [B, H_k, N_k, D]  torch.float8_e4m3fn
        v:       [B, H_k, N_k, D]  BF16 or FP16
        q_scale: [B, H,   N_q]     FP32 per-token Q scales
        k_scale: [B, H_k, N_k]     FP32 per-token K scales
        delta_s: [B, H,   G, N_k]  FP32 correction (None → skip)
        ...

    Returns:
        out: [B, H, N_q, D]  same dtype as v.
    """
    B, H, N_q, D = q_fp8.shape
    H_k = k_fp8.shape[1]
    N_k = k_fp8.shape[2]

    assert (D & (D - 1)) == 0, "Head dim must be power of 2."
    assert H % H_k == 0

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    has_delta_s = delta_s is not None
    out = torch.empty(B, H, N_q, D, dtype=v.dtype, device=v.device)

    grid = (triton.cdiv(N_q, block_m), B * H)

    ds_ptr     = delta_s if has_delta_s else q_fp8
    ds_strides = (
        delta_s.stride(0), delta_s.stride(1),
        delta_s.stride(2), delta_s.stride(3)
    ) if has_delta_s else (0, 0, 0, 0)

    _fwd_kernel_fp8[grid](
        q_fp8, k_fp8, v, q_scale, k_scale, ds_ptr, out,
        softmax_scale,
        q_fp8.stride(0),  q_fp8.stride(1),  q_fp8.stride(2),  q_fp8.stride(3),
        k_fp8.stride(0),  k_fp8.stride(1),  k_fp8.stride(2),  k_fp8.stride(3),
        v.stride(0),      v.stride(1),      v.stride(2),      v.stride(3),
        out.stride(0),    out.stride(1),    out.stride(2),     out.stride(3),
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        *ds_strides,
        N_q, N_k, H, H_k,
        HEAD_DIM       = D,
        BLOCK_M        = block_m,
        BLOCK_N        = block_n,
        IS_CAUSAL      = is_causal,
        HAS_DELTA_S    = has_delta_s,
        PER_BLOCK_MEAN = per_block_mean,
        num_warps      = 4,
        num_stages     = 3,
    )
    return out
