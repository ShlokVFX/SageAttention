import torch
import triton
import triton.language as tl

# =========================================================
# SAFE CONFIG (NO OOM)
# =========================================================
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2


# =========================================================
# BF16 / FP16 KERNEL (FIXED)
# =========================================================
@triton.jit
def _fwd_kernel_bf16(
    Q, K, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_q, N_k, H, H_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    m_pid = tl.program_id(0)
    bh_pid = tl.program_id(1)

    b = bh_pid // H
    h = bh_pid % H
    kv_h = h // (H // H_k)

    offs_m = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # ---- Load Q ----
    q_ptrs = Q + b*stride_qb + h*stride_qh \
           + offs_m[:, None]*stride_qn \
           + offs_d[None, :]*stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_q, other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_start in range(0, N_k, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # ---- Load K ----
        k_ptrs = K + b*stride_kb + kv_h*stride_kh \
               + offs_d[:, None]*stride_kd \
               + offs_n[None, :]*stride_kn
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N_k, other=0.0)

        qk = tl.dot(q, k) * softmax_scale

        qk = tl.where(offs_n[None, :] < N_k, qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # ---- Load V ----
        v_ptrs = V + b*stride_vb + kv_h*stride_vh \
               + offs_n[:, None]*stride_vn \
               + offs_d[None, :]*stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_k, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]

    # ---- Store ----
    o_ptrs = Out + b*stride_ob + h*stride_oh \
           + offs_m[:, None]*stride_on \
           + offs_d[None, :]*stride_od

    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=offs_m[:, None] < N_q)

# =========================================================
# FP8 KERNEL (UNCHANGED BUT SAFE)
# =========================================================
@triton.jit
def _fwd_kernel_fp8(
    Q, K, V, Qs, Ks, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_qsb, stride_qsh, stride_qsn,
    stride_ksb, stride_ksh, stride_ksn,
    N_q, N_k, H, H_k,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
):
    m_pid = tl.program_id(0)
    bh_pid = tl.program_id(1)

    b = bh_pid // H
    h = bh_pid % H
    kv_h = h // (H // H_k)

    offs_m = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptr = Q + b*stride_qb + h*stride_qh \
          + offs_m[:, None]*stride_qn + offs_d[None, :]*stride_qd
    q = tl.load(q_ptr, mask=offs_m[:, None] < N_q, other=0.0)

    qs_ptr = Qs + b*stride_qsb + h*stride_qsh + offs_m*stride_qsn
    qs = tl.load(qs_ptr, mask=offs_m < N_q, other=1.0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    kv_base_k = b*stride_kb + kv_h*stride_kh
    kv_base_v = b*stride_vb + kv_h*stride_vh
    ks_base = b*stride_ksb + kv_h*stride_ksh

    for start_n in range(0, N_k, BLOCK_N):
        n = start_n + offs_n

        k_ptr = K + kv_base_k \
              + offs_d[:, None]*stride_kd + n[None, :]*stride_kn
        k = tl.load(k_ptr, mask=n[None, :] < N_k, other=0.0)

        ks_ptr = Ks + ks_base + n*stride_ksn
        ks = tl.load(ks_ptr, mask=n < N_k, other=1.0)

        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= qs[:, None] * ks[None, :]
        qk *= softmax_scale

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        v_ptr = V + kv_base_v \
              + n[:, None]*stride_vn + offs_d[None, :]*stride_vd
        v = tl.load(v_ptr, mask=n[:, None] < N_k, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

    acc = acc / l_i[:, None]

    o_ptr = Out + b*stride_ob + h*stride_oh \
          + offs_m[:, None]*stride_on + offs_d[None, :]*stride_od
    tl.store(o_ptr, acc.to(Out.dtype.element_ty),
             mask=offs_m[:, None] < N_q)


# =========================================================
# PYTHON API
# =========================================================
def sage_attn3_fwd(
    q,
    k,
    v,
    softmax_scale=None,
    is_causal=False,
    per_block_mean=True,
    block_m=64,
    block_n=64,
):
    B, H, N_q, D = q.shape
    H_k = k.shape[1]
    N_k = k.shape[2]

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    out = torch.empty_like(q)

    grid = (triton.cdiv(N_q, BLOCK_M), B * H)

    _fwd_kernel_bf16[grid](
        q, k, v, out,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_q, N_k, H, H_k,
        HEAD_DIM=D,
        BLOCK_M=64,
        BLOCK_N=64,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=2,
    )

    return out

def sage_attn3_fwd_fp8(*args, **kwargs):
    # fallback to bf16 kernel
    return sage_attn3_fwd(*args, **kwargs)

def sage_attn3_fwd_fp4(*args, **kwargs):
    # fallback to bf16 kernel
    return sage_attn3_fwd(*args, **kwargs)