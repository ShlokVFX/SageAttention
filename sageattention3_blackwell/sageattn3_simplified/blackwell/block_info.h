/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * This code is based on code from FlashAttention3, https://github.com/Dao-AILab/flash-attention
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
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
 * Handle variable-length sequences in a batch.
 *
 * PROBLEM:
 * In normal attention, all sequences in batch have same length (padded to max).
 * But padding waste compute (attend to padding tokens = wasted work).
 *
 * SOLUTION: Variable-length (varlen) mode:
 * Pack all sequences into one long tensor without padding.
 * Use cu_seqlens array to track where each sequence starts/ends.
 * cu_seqlens[i] = cumulative sum of lengths up to sequence i.
 * Example: sequences [3, 5, 2] -> cu_seqlens = [0, 3, 8, 10]
 *
 * BlockInfo computes:
 *   actual_seqlen_q: how long is Q for this batch item
 *   actual_seqlen_k: how long is K for this batch item
 *   sum_s_q: starting position of this batch item's Q in flattened tensor
 *   sum_s_k: starting position of this batch item's K in flattened tensor
 *
 * q_offset/k_offset: compute byte offset to start of this batch item's data.
 *   If Varlen=false or no cu_seqlens: use bidb * batch_stride (regular batched)
 *   If Varlen=true with cu_seqlens: use sum_s * row_stride (packed/varlen)
 */

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        // sum_s_q: start position in packed Q tensor (-1 if not varlen)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        // sum_s_k: start position in packed K tensor (-1 if not varlen)
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        // actual_seqlen_q: real Q length for this batch item
        //   non-varlen: use global seqlen_q
        //   varlen: compute from cumulative sum difference
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // actual_seqlen_k: real K length for this batch item
        //   is_seqlens_k_cumulative=true: cu_seqlens_k stores cumulative sums (end - start)
        //   is_seqlens_k_cumulative=false: cu_seqlens_k[bidb] stores the length directly
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        // seqused_k: if provided, use actual used K length (for KV cache with dynamic lengths)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    // Compute byte offset to start of this batch item's Q data
    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // sum_s_q == -1 means fixed-length batched (not packed)
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    // Compute byte offset to start of this batch item's K data
    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;        // start position of this sequence's Q in packed tensor (-1 = not packed)
    const int sum_s_k;        // start position of this sequence's K in packed tensor (-1 = not packed)
    const int actual_seqlen_q; // actual Q sequence length for this batch item
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;  // K cache sequence length
    const int actual_seqlen_k; // actual K sequence length for this batch item
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
