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
 * Online softmax fused with FP4 quantization of attention scores (P matrix).
 *
 * WHAT IS ONLINE SOFTMAX:
 * Normal softmax need two passes: one to find max, one to compute exp/sum.
 * Online softmax can do it as you stream through K blocks:
 *   - Keep running max and running sum
 *   - When new block comes, update max, rescale old sum, add new exp values
 * This way only need one K-direction pass = less memory traffic = faster.
 *
 * WHAT IS FUSED QUANTIZATION:
 * After computing softmax probabilities, quantize them to FP4 E2M1.
 * Do this on-the-fly during softmax so no need to store full FP32 P matrix.
 * Quantization scale = max_abs_value / 6.0 (6 = max FP4 E2M1 value).
 * Scale stored as FP8 E4M3 = AbsMaxP.
 *
 * MATH:
 * Standard softmax: P_ij = exp(Q_i * K_j / sqrt(d) - max_i) / sum_j_exp(...)
 * With FP4 quant:   P4_ij = round_to_fp4(P_ij / scale_i) where scale_i = max_j(P_ij) / 6
 * Combined:         P4_ij = round_to_fp4(exp(score * scale_log2 - max_scaled) / scale_i)
 *
 * WHY log2 TRICK:
 * exp(x) = exp2(x * log2(e)) = exp2(x / ln2)
 * scale_log2 = softmax_scale * log2(e)
 * Using exp2 is faster (hardware instruction). FMA can combine "x*scale_log2 - max_scaled".
 *
 * ROWS:
 * Template parameter Rows = number of row segments each thread owns.
 * In SM120 MMA with kBlockM=128 and 256 MMA threads: 2 * (2 * 128 / 256) = 2 rows per thread.
 * Each thread tracks max and sum for its rows independently.
 */

#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Rows>
struct SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    // Persistent state across K-block iterations:
    TensorT row_sum;        // running sum of exp values per row
    TensorT row_max;        // running maximum score per row
    TensorT scores_scale;   // rescale factor when max changes between blocks

    // fp8_scalexfp4_scale = 1 / (448 * 6):
    //   448 = max value of FP8 E4M3
    //     6 = max value of FP4 E2M1
    // When AbsMaxP stored as FP8 E4M3, multiply by this to get actual FP4 scale.
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    // Same thing but in log2 space (for exp2 trick): log2(1/(448*6))
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
    // log2(1/6) - offset applied to AbsMaxP to convert to FP4 scale
    static constexpr float fp4_scale_log2 = -2.584962500721156f;
    // How many threads share same row for cross-thread reduction (4-thread quad)
    static constexpr int RowReductionThr = 4;

    CUTLASS_DEVICE SoftmaxFused(){};

    /*
     * online_softmax_with_quant: process one K-block worth of attention scores.
     *
     * Called once per K-block in the mainloop.
     * acc = accumulator from Q*K^T matrix multiply for this tile (shape: (MmaAtom,MmaM,MmaN))
     * AbsMaxP = per-scale-factor absolute max tracker (for FP4 quantization of P)
     * softmax_scale_log2 = (1/sqrt(d)) * log2(e) -- combined scale for exp2 trick
     *
     * FirstTile=true: initialize max/sum from scratch (first K block)
     * FirstTile=false: update max/sum incrementally (subsequent K blocks)
     * InfCheck: check for -inf max (all tokens masked in causal attention)
     */
    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        // Reshape accumulator for row-reduction (see utils.h convert_to_reduction_layout)
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        // Reshape accumulator for FP4 conversion (see utils.h convert_to_conversion_layout)
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        // Flatten to (num_scale_groups, scale_group_size) for quantization
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));

        if constexpr (FirstTile) {
            // FIRST K-BLOCK: initialize running stats
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            // Step 1: find AbsMaxP per scale group and update row_max
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    // Each "ni" group = one FP4 scale factor group (16 elements)
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    // Exchange max with neighbor thread (they own adjacent 8 elements of same scale group)
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1);
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                // Reduce row_max across the 4-thread quad (all 4 threads own same row)
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // Compute fused scale: combine softmax scale with FP4 quantization scale
                // max_scaled = max * softmax_scale_log2 + log2(1/(448*6))
                // This bakes the FP8->float dequant and FP4 quant scale into one offset.
                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // Step 2: compute exp2 on all elements (softmax numerator)
                // acc = exp2(acc * scale - max_scaled) -- now in [0, 1/(448*6)] range
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }

                // Step 3: convert AbsMaxP to FP4 scale factor
                // AbsMaxP = exp2(AbsMaxP * scale - max_scaled + log2(1/6))
                // After this, AbsMaxP holds the scale for FP4 quantization of P.
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }

            // Step 4: accumulate into row_sum (denominator for softmax)
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            // SUBSEQUENT K-BLOCKS: update running stats with new data
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);  // save previous max

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                // Find max within this block (per scale group)
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // Compute rescale factor: how much to multiply old O accumulator by
                // When max increases, all previous exp values are too large by factor exp2(delta)
                // Must rescale running sum and old O output.
                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // Rescale old sum by correction factor, add new exp values
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }
        }

        // Step 5: divide acc by AbsMaxP to get values in FP4 range [-6, 6]
        // After this, acc[i] / AbsMaxP[group(i)] is in correct range for FP4 quantization.
        // The FP4 MMA hardware will then multiply by AbsMaxP (as scale factor) automatically.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
        }
    }

    /*
     * finalize: after all K-blocks processed, divide output O by total sum.
     * This normalizes the weighted sum to get proper attention output.
     * Also finalize cross-thread row_sum reduction (gather from all 4 threads in quad).
     */
    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            // Final reduction: gather row_sum from all 4 threads in this row's quad
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            // Handle edge case: if sum is 0 or NaN (all tokens masked), output 0
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            // Divide each output element by the softmax normalizer
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) *= inv_sum;
             }
        }
    }

    /*
     * rescale_o: update running output accumulator when max changes.
     * o_store = o_store * scores_scale + o_tmp
     * Where:
     *   o_store = accumulated O from previous K-blocks (needs rescaling)
     *   scores_scale = correction factor for max change
     *   o_tmp = new O from current K-block
     *
     * This is the key trick in online attention: old output is rescaled
     * to match new normalization, then new output added on top.
     */
    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }
    }
};

} // namespace flash
