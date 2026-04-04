"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

WHAT THIS FILE DO:
Python API that chains together preprocessing + quantization + CUDA attention kernel.

PIPELINE (sageattn3_blackwell):
  1. preprocess_qkv():
     - Normalize K: subtract mean across sequence (reduce quantization error)
     - Pad Q, K, V sequence lengths to multiple of 128 (kernel requirement)
     - Subtract per-block Q mean (smooth quantization):
       * per_block_mean=True: subtract mean per 128-token block using Triton kernel
       * per_block_mean=False: subtract global sequence mean
     - Compute delta_s = qm @ K^T (correction term for attention score)
       delta_s compensates for the mean subtraction in the kernel

  2. scale_and_quant_fp4_permute(q) -> (q_fp4, sfq):
     - Quantize Q to FP4 E2M1 with FP8 scale factors (per 128 elements)
     - "permute" = reorder bytes to match SM120 MMA input format

  3. scale_and_quant_fp4_permute(k) -> (k_fp4, sfk):
     - Same for K

  4. scale_and_quant_fp4_transpose(v) -> (vt_fp4, sfvt):
     - Quantize V AND transpose to [B, H, D, N] layout
     - Transposed because attention does O = P * V^T so V needs to be transposed

  5. blockscaled_fp4_attn() -> O:
     - Call fp4attn_cuda.fwd() (CUDA kernel via pybind11)
     - Returns O in BF16 or FP16, shape [B, H, seqlen_q_padded, D]
     - Slice [:, :, :QL, :] to remove padding

TRITON KERNEL (group_mean_kernel):
  Compute mean of Q tokens in groups of 128. Subtract mean from Q.
  Store Q_centered (mean-subtracted) and qm (group means).
  Grid = (batch, heads, num_groups). Each program handles one 128-token group.
  Uses Triton for GPU-accelerated computation without CUDA boilerplate.

WHY SMOOTH QUANTIZATION:
  Q values have varying magnitudes. FP4 has range [-6, 6].
  If Q has mean 3.0 and std 0.5, range is [2, 4] -> wastes most of FP4 range.
  After mean subtraction: range is [-0.5*6, 0.5*6] = [-3, 3] -> better utilization.
  The correction delta_s = qm @ K^T added back in CUDA kernel to compensate.

SOFTMAX SCALE:
  softmax_scale = 1/sqrt(D) where D = head_dim.
  Standard attention temperature to prevent softmax saturation.
  Note: D * 2 because Q/K stored in FP4 which has D//2 bytes -> actual head_dim = D*2.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          # input Q pointer: [B, H, L, D]
    q_out_ptr,      # output mean-subtracted Q pointer: [B, H, L, D]
    qm_out_ptr,     # output group means pointer: [B, H, num_groups, D]
    B, H, L, D: tl.constexpr,    # batch, heads, seq_len, head_dim
    stride_qb, stride_qh, stride_ql, stride_qd,  # Q strides
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  # qm strides
    GROUP_SIZE: tl.constexpr  # tokens per group (128)
):
    # Each Triton program handles one (batch, head, group) combination
    pid_b = tl.program_id(0)      # batch index
    pid_h = tl.program_id(1)      # head index
    pid_group = tl.program_id(2)  # group index (0 to num_groups-1)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)  # token indices in this group

    # Load GROUP_SIZE x D block of Q values
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)

    # Compute mean over GROUP_SIZE tokens (axis=0), broadcast to all tokens
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE  # shape [D]

    # Subtract group mean from Q values (smooth quantization)
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    # Store group mean (used to compute delta_s correction)
    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    """Subtract per-128-token-group mean from Q. Returns (Q_centered, group_means)."""
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)  # [B, H, num_groups, D]

    grid = (B, H, num_groups)

    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    """
    Prepare Q, K, V for FP4 attention kernel.

    Steps:
    1. K normalization: subtract K's global mean to center distribution
    2. Pad seqlen to multiple of 128 (kernel tile size requirement)
    3. Q smooth quantization: subtract block mean from Q
    4. Compute delta_s = qm @ K^T (correction term for smooth quant)

    Returns: (q_centered, k_normalized, v_padded, delta_s)
    """
    def pad_128(x):
        # Pad sequence length (dim 2) to multiple of 128
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    # Normalize K: subtract sequence mean to center values (better FP4 range usage)
    k -= k.mean(dim=-2, keepdim=True)
    q, k, v = map(lambda x: pad_128(x), [q, k, v])

    if per_block_mean:
        # Per-128-token-block mean subtraction using Triton kernel
        q, qm = triton_group_mean(q)
    else:
        # Single global mean subtraction
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm

    # delta_s = qm @ K^T: correction for attention scores
    # In kernel: S = Q_centered * K + delta_s ≈ Q_original * K (corrected)
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 tensor to FP4 E2M1 + FP8 E4M3 scale factors."""
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)  # 2 FP4 per byte
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)  # 1 scale per 16 FP4
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP4 + permute byte order to match SM120 MMA input format."""
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to FP4 and transpose from [B, H, N, D] to [B, H, D, N//2]."""
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)  # transposed!
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(qlist: Tuple,
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False,
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    """
    Call the FP4 block-scaled attention CUDA kernel.

    qlist = (q_fp4, sfq): FP4 query data + FP8 scale factors
    klist = (k_fp4, sfk): FP4 key data + FP8 scale factors
    vlist = (vt_fp4, sfvt): FP4 value data (transposed) + FP8 scale factors
    delta_s: correction term for smooth quantization [B, H, seqlen_q//128, seqlen_k]
    KL: original (unpadded) key sequence length (for boundary masking)
    is_causal: apply causal mask (lower triangular)
    per_block_mean: True if delta_s is per-128-block, False if per-sequence
    is_bf16: True = BF16 output, False = FP16 output

    Returns: (output_tensor, None) where output has shape [B, H, seqlen_q_padded, D]
    """
    # softmax_scale = 1/sqrt(head_dim)
    # Note: qlist[0].shape[-1] = D//2 (packed FP4), so actual head_dim = D*2
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1],
                            delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask=None, is_causal=False, per_block_mean=True, **kwargs):
    """
    Main entry point: FP4 block-scaled attention on Blackwell (SM120) GPU.

    q, k, v: attention tensors in BF16 or FP16, shape [B, H, L, D]
    is_causal: if True, apply causal mask
    per_block_mean: if True, use per-128-token-block mean for Q normalization

    Returns: output tensor in same dtype as q, shape [B, H, L, D]
    """
    if q.size(-1) >= 256:
        # Headdim 256+ not yet optimized - fall back to PyTorch SDPA
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal=is_causal)

    QL = q.size(2)   # original query length (before padding)
    KL = k.size(2)   # original key length (before padding, for boundary mask)
    is_bf16 = q.dtype == torch.bfloat16

    # Step 1: preprocess (normalize, pad, smooth quant, compute delta_s)
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)

    # Step 2: quantize Q, K, V to FP4
    qlist_from_cuda = scale_and_quant_fp4(q)           # Q: permuted FP4
    klist_from_cuda = scale_and_quant_fp4_permute(k)   # K: permuted FP4
    vlist_from_cuda = scale_and_quant_fp4_transpose(v) # V: transposed FP4

    # Step 3: run FP4 attention kernel
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda,
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16
    )[0][:, :, :QL, :].contiguous()  # slice off padding

    return o_fp4
