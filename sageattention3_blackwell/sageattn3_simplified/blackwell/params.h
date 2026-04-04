/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * WHAT THIS FILE DO:
 * Hold big bag of settings kernel need. Like address of Q, K, V tensors
 * and how they laid out in memory (strides). Also hold scale factors for FP4.
 *
 * WHY TWO STRUCTS:
 * Qkv_params = basic Q/K/V pointers and strides.
 * Flash_fwd_params = extends Qkv_params with output, softmax stuff, causal flag, etc.
 * Split so you can reuse Qkv_params in other code later.
 *
 * WHAT IS STRIDE:
 * Stride tell GPU how many elements to skip to get to next row/head/batch.
 * Example: q_row_stride=128 mean "skip 128 elements to get to next query token row".
 * This handle non-contiguous memory without copying.
 *
 * WHAT IS FP4 SCALE FACTORS (sfq, sfk, sfv):
 * Because Q,K,V stored in tiny FP4 format (only 4 bits!), we need scale factors.
 * Every 16 elements share one FP8 scale factor. Scale factor tell real value = fp4_value * scale.
 * sfq_ptr = pointer to Q scale factors matrix.
 */

#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

////////////////////////////////////////////////////////////////////////////////////////////////////

/* BASE PARAM STRUCT: Q, K, V pointers + strides + scale factor pointers */
struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;    // pointer to Query  tensor (FP4 packed)
    void *__restrict__ k_ptr;    // pointer to Key    tensor (FP4 packed)
    void *__restrict__ v_ptr;    // pointer to Value  tensor (FP4 packed, transposed)
    void *__restrict__ delta_s_ptr;  // pointer to per-block mean of Q (for smooth quant)
    // The QKV scale factor matrices. (FP8 E4M3, one scale per 16 FP4 elements)
    void *__restrict__ sfq_ptr;  // Q scale factors
    void *__restrict__ sfk_ptr;  // K scale factors
    void *__restrict__ sfv_ptr;  // V scale factors (transposed layout)
    // The stride between rows of the Q, K and V matrices.
    // stride_row = how many elements to jump to get to next sequence token
    // stride_head = how many elements to jump to get to next attention head
    // stride_batch = how many elements to jump to get to next batch item
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t ds_batch_stride;  // delta_s (per-block mean) strides
    index_t ds_row_stride;
    index_t ds_head_stride;
    // The stride of the Q, K and V scale factor matrices.
    index_t sfq_batch_stride;
    index_t sfk_batch_stride;
    index_t sfv_batch_stride;
    index_t sfq_row_stride;
    index_t sfk_row_stride;
    index_t sfv_row_stride;
    index_t sfq_head_stride;
    index_t sfk_head_stride;
    index_t sfv_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    // Example: GQA with 8 Q heads and 2 KV heads -> h=8, h_k=2, h_h_k_ratio=4
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* FULL FORWARD PARAMS: adds output, LSE, causal flag, sequence lengths, etc. */
struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output). Shape = [batch, seqlen_q, heads, head_dim]
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;  // accumulator output (for split-K, unused here)
    void * __restrict__ s_ptr;       // attention score matrix (for debugging)

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix. (for debugging attention weights)
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum (log-sum-exp values for numerical stability)
    // LSE(i) = log(sum_j exp(Q_i * K_j / sqrt(d))) -- stored for backward pass
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b;               // batch size
    int seqlen_q;        // sequence length of Query
    int seqlen_k;        // sequence length of Key/Value
    int seqlen_knew;     // sequence length of new KV (for KV cache append)
    int d;               // head dimension
    int seqlen_q_rounded; // seqlen_q rounded up to block size
    int seqlen_k_rounded; // seqlen_k rounded up to block size
    int d_rounded;        // d rounded up
    int rotary_dim;       // dimension for rotary embedding
    int unpadded_seqlen_k; // actual (unpadded) K seqlen
    // Fast divmod: division + modulo in one instruction (faster than separate / and %)
    cutlass::FastDivmod head_divmod, m_block_divmod;
    int total_blocks;    // total number of tile blocks to compute
    int seqlen_s;        // seqlen for delta_s (per-block mean)

    // The scaling factors for the kernel.
    float scale_softmax;          // 1/sqrt(d) softmax temperature
    float scale_softmax_log2;     // scale_softmax * log2(e) -- used for exp2 trick
    uint32_t scale_softmax_log2_half2; // packed half2 version

    // array of length b+1 holding starting offset of each sequence.
    // Used for variable-length (packed/padded) batches.
    // cu_seqlens_q[i] = start position of batch item i in flattened Q tensor
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;  // optional mask for sparse attention

    // The K_new and V_new matrices. (for KV cache append, not used in basic fwd)
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding. (RoPE, not used in basic fwd)
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache (like virtual memory for KV cache, not used in basic fwd)
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size for sliding window attention
    int window_size_left, window_size_right;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;          // true = bfloat16 output, false = float16 output
    bool is_e4m3;          // true = FP8 E4M3 input (not used in FP4 mode)
    bool is_causal;        // true = only attend to past tokens (lower triangle mask)
    bool per_block_mean;   // true = use per-block mean for Q smooth quantization
    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;  // style of rotary embedding

    int num_splits;  // For split-KV version (not used here)

    void * __restrict__ alibi_slopes_ptr;        // ALiBi attention bias
    index_t alibi_slopes_batch_stride;

    int * __restrict__ tile_count_semaphore;  // semaphore for dynamic tile scheduling
};

////////////////////////////////////////////////////////////////////////////////////////////////////
