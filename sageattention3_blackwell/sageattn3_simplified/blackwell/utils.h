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
 * Small helper functions used everywhere in kernel:
 *   1. Max/Sum reduction across threads in warp
 *   2. Fast exp2 (base-2 exponential) via PTX assembly
 *   3. Float -> FP8 and Float -> FP4 conversion via PTX assembly
 *   4. float2 vector operations (add, sub, mul, fma) via PTX
 *   5. CuTe layout converters for softmax operations
 *   6. Conditional tile copy with boundary checks
 *
 * WHY PTX ASSEMBLY:
 * PTX = PTX Instruction Set Architecture. Like assembly for GPU.
 * Using `asm volatile(...)` bypass C++ compiler and use exact GPU instruction.
 * This give control over which instruction used (ex: "ex2.approx" = fast exp2).
 * Compiler sometimes use slower instructions unless told specifically.
 *
 * WHY float2:
 * Process 2 floats at same time using vector instructions.
 * GPU have native f32x2 instructions that do two adds/muls in one cycle.
 * 2x throughput for free.
 *
 * WARP REDUCTION:
 * In softmax need max/sum across all threads that own same row.
 * __shfl_xor_sync = exchange value with another thread in warp via butterfly pattern.
 * After log2(32)=5 rounds, all threads have global result.
 */

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// MAX and SUM operator functors for use with reduction templates

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

// Specialization for float uses hardware max() intrinsic (slightly faster)
template <>
struct MaxOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * WARP-LEVEL ALL-REDUCE via butterfly shuffle:
 * Allreduce<32>::run(x, op) reduces x across all 32 threads in a warp.
 *
 * How butterfly works (for 4 threads: 0,1,2,3):
 *   Round 1: thread 0 XORs with thread 2, thread 1 XORs with thread 3
 *   Round 2: thread 0 XORs with thread 1, thread 2 XORs with thread 3
 *   After: all threads have same reduced value.
 *
 * __shfl_xor_sync(mask, val, offset):
 *   Each thread reads val from the thread whose ID = my_id XOR offset.
 *   uint32_t(-1) = all lanes active mask.
 */
template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);  // recursive: halve THREADS each time
    }
};

// Base case: 2 threads - just one shuffle
template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * thread_reduce_: reduce tensor along N dimension within single thread.
 * Tensor shape = [M_rows, N_cols], summary shape = [M_rows].
 * Each thread iterates over its N_cols and accumulates (max or sum) into summary.
 */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

/*
 * quad_allreduce_: reduce across 4 threads in a quad (adjacent 4 lanes in warp).
 * Used when each row is spread across 4 threads.
 */
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

/*
 * reduce_: combined thread_reduce_ + quad_allreduce_.
 * First reduce within thread, then reduce across 4-thread quad.
 */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

/* Shorthand: reduce to find row-wise maximum */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

/* Shorthand: reduce to find row-wise sum. warp_reduce=false to skip quad shuffle. */
template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

/*
 * half_exp: fast FP16x2 exp2 via PTX.
 * "ex2.approx.f16x2" = GPU hardware approximate exp2 on two FP16 values packed in one register.
 * Approximately 10x faster than software exp. Accuracy ~3 ULP.
 */
__forceinline__ __device__ __half2 half_exp(__half2 x) {
    uint32_t tmp_out, tmp_in;
    tmp_in = reinterpret_cast<uint32_t&>(x);
    asm ("ex2.approx.f16x2 %0, %1;\n"
      : "=r"(tmp_out)
      : "r"(tmp_in));
    __half2 out = reinterpret_cast<__half2&>(tmp_out);
    return out;
}

/*
 * max_scale_exp2_sum: fused max-find + exp2 + sum in one pass.
 * For softmax: find max, compute exp2(x*scale - max*scale), sum results.
 *
 * WHY exp2 not exp:
 * exp(x) = exp2(x * log2(e))
 * Compiler can fuse "x * scale - max * scale" into single FMA instruction.
 * exp2 hardware instruction exists on GPU (faster than exp).
 */
template <bool zero_init=false, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor"); static_assert(Layout1::rank == 1, "Only support 1D Tensor"); CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, all elements were masked. Avoid -inf - (-inf) = NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
    }
}

/*
 * scale_apply_exp2: apply exp2(x*scale - max*scale) to every element.
 * Used when max is already known (from a previous pass).
 * Scale_max=true: multiply max by scale before subtracting.
 */
template <bool Scale_max=true, bool Check_inf=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (max(mi) * (Scale_max ? scale : float(M_LOG2E))))
            : (max(mi) * (Scale_max ? scale : float(M_LOG2E)));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * ptx_exp2: fast FP32 approximate base-2 exponential.
 * Uses "ex2.approx.ftz.f32" PTX instruction.
 * "ftz" = flush denormals to zero (slightly less accurate but faster).
 * ~10x faster than calling expf() for same accuracy.
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*
 * packed_float_to_ue4m3: convert 4 float32 values to packed FP8 E4M3 format.
 * Output: 4 FP8 values packed into one uint32 (1 byte each = 32 bits total).
 * E4M3 = 4-bit exponent, 3-bit mantissa, no sign (UE = unsigned).
 * Used for scale factors (sfq, sfk, sfv).
 *
 * PTX instruction "cvt.rn.satfinite.e4m3x2.f32" converts 2 floats to 2 FP8.
 * "rn" = round to nearest, "satfinite" = clamp to FP8 range.
 */
CUTLASS_DEVICE void
packed_float_to_ue4m3(
  float const &f0, float const &f1, float const &f2, float const &f3,
  uint32_t &out
) {
  asm volatile( \
    "{\n" \
    ".reg .b16 lo;\n" \
    ".reg .b16 hi;\n" \
    "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
    "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
    "mov.b32 %0, {lo, hi};\n" \
    "}" \
    : "=r"(out) : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
}

/*
 * packed_float_to_e2m1: convert 8 float32 values to packed FP4 E2M1 format.
 * Output: 8 FP4 values packed into one uint32 (4 bits each = 32 bits total).
 * E2M1 = 2-bit exponent, 1-bit mantissa, 1-bit sign. Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
 * Used for quantizing Q, K, V attention weights.
 *
 * PTX instruction "cvt.rn.satfinite.e2m1x2.f32" converts 2 floats to 2 FP4.
 * 4 such instructions pack 8 FP4 values into one 32-bit register.
 */
CUTLASS_DEVICE void
packed_float_to_e2m1(
  float const &f0, float const &f1, float const &f2, float const& f3,
  float const &f4, float const &f5, float const &f6, float const& f7,
  uint32_t &out
) {
    asm volatile( \
    "{\n" \
    ".reg .b8 byte0;\n" \
    ".reg .b8 byte1;\n" \
    ".reg .b8 byte2;\n" \
    ".reg .b8 byte3;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n" \
    "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n" \
    "mov.b32 %0, {byte0, byte1, byte2, byte3};\n" \
    "}" \
    : "=r"(out) : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
                  "f"(f4), "f"(f5), "f"(f6), "f"(f7));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// float2 vector operations using native GPU vector instructions.
// Process 2 floats at once = 2x throughput.

/* c = a + b (two floats at once) */
CUTLASS_DEVICE void
add(float2      & c,
    float2 const& a,
    float2 const& b)
{
asm volatile("add.f32x2 %0, %1, %2;\n"
  : "=l"(reinterpret_cast<uint64_t      &>(c))
  :  "l"(reinterpret_cast<uint64_t const&>(a)),
      "l"(reinterpret_cast<uint64_t const&>(b)));
}

/* a += b (in-place, two floats at once) */
CUTLASS_DEVICE void
add_inplace(float2 &a,
            float2 const& b)
{
  asm volatile("add.f32x2 %0, %0, %1;\n"
    : "+l"(reinterpret_cast<uint64_t &>(a))
    :  "l"(reinterpret_cast<uint64_t const&>(b))
  );
}

/* c = a - b (two floats at once) */
CUTLASS_DEVICE void
sub(float2      & c,
    float2 const& a,
    float2 const& b)
{
asm volatile("sub.f32x2 %0, %1, %2;\n"
  : "=l"(reinterpret_cast<uint64_t      &>(c))
  :  "l"(reinterpret_cast<uint64_t const&>(a)),
      "l"(reinterpret_cast<uint64_t const&>(b)));
}

/* a -= b (in-place, two floats at once) */
CUTLASS_DEVICE void
sub_inplace(float2 &a,
            float2 const& b)
{
  asm volatile("sub.f32x2 %0, %0, %1;\n"
    : "+l"(reinterpret_cast<uint64_t &>(a))
    :  "l"(reinterpret_cast<uint64_t const&>(b))
  );
}

/* c = a * b (two floats at once) */
CUTLASS_DEVICE void
mul(float2      & c,
    float2 const& a,
    float2 const& b)
{
  asm volatile("mul.f32x2 %0, %1, %2;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(c))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)));
}

/*
 * fma: d = a * b + c (fused multiply-add, two floats at once)
 * "rn" = round to nearest. Single instruction, full precision.
 * Better than separate mul + add because no rounding error in between.
 */
CUTLASS_DEVICE void
fma(float2      & d,
    float2 const& a,
    float2 const& b,
    float2 const& c)
{
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(d))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)),
       "l"(reinterpret_cast<uint64_t const&>(c)));
}

/* a = a * b + c (in-place fma, two floats at once) */
CUTLASS_DEVICE void
fma_inplace(float2 &a,
            float2 const& b,
            float2 const& c)
{
  asm volatile("fma.rn.f32x2 %0, %0, %1, %2;\n"
    : "+l"(reinterpret_cast<uint64_t      &>(a))
    :  "l"(reinterpret_cast<uint64_t const&>(b)),
       "l"(reinterpret_cast<uint64_t const&>(c)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * convert_to_reduction_layout: reshape MMA accumulator layout for softmax row reduction.
 *
 * MMA accumulator from GPU matrix multiply has shape:
 *   (MmaAtom, MmaM, MmaN) where MmaAtom = (AtomN, AtomM)
 *
 * For softmax we need to find max/sum per ROW. So reshape to:
 *   (rows, cols) where each row index groups all N elements for same M position.
 *
 * Input layout: ((AtomN, AtomM), MmaM, MmaN)
 * Output layout: ((AtomM, MmaM), (AtomN, MmaN))
 *   - dim 0 = row index (M dimension)
 *   - dim 1 = column index (N dimension)
 */
template <
  class Layout
>
CUTLASS_DEVICE constexpr
auto convert_to_reduction_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  return make_layout(
    make_layout(get<0,1>(mma_layout), get<1>(mma_layout)),   // row = (AtomM, MmaM)
    make_layout(get<0,0>(mma_layout), get<2>(mma_layout))    // col = (AtomN, MmaN)
  );
}

template <
  class Tensor
>
CUTLASS_DEVICE constexpr
auto convert_to_reduction_tensor(Tensor mma_tensor) {
  return make_tensor(mma_tensor.data(), convert_to_reduction_layout(mma_tensor.layout()));
}

/*
 * convert_to_conversion_layout: reshape MMA accumulator for FP4 quantization.
 *
 * FP4 quantization works on groups of 8 consecutive elements (for E2M1 format).
 * Need to group elements so each group of 8 corresponds to elements that will
 * share one scale factor (packed into 32 bits = 8 x 4-bit values).
 *
 * Rearranges (MmaAtom, MmaM, MmaN) so that element groups map to scale factor groups.
 * MmaAtomN=8 and MmaAtomM=2 are hardware constraints from SM120 MMA shape (16x32x64).
 */
template <
  class Layout
>
CUTLASS_DEVICE constexpr
auto convert_to_conversion_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  constexpr int MmaAtomN = size<0, 0>(mma_layout);
  constexpr int MmaAtomM = size<0, 1>(mma_layout);
  constexpr int MmaM = size<1>(mma_layout);
  constexpr int MmaN = size<2>(mma_layout);

  static_assert(MmaAtomN == 8, "MmaAtomN should be 8.");
  static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
  static_assert(MmaN % 2 == 0, "MmaN should be multiple of 2.");

  // Split MmaN into groups of 2 (so each uint32 holds 8 FP4 = 4 pairs)
  auto mma_n_division = zipped_divide(
    layout<2>(mma_layout), make_tile(_2{})
  );
  return make_layout(
    make_layout(layout<0,0>(mma_layout), make_layout(layout<0,1>(mma_layout), layout<0>(mma_n_division))),
    layout<1>(mma_layout), layout<1>(mma_n_division)
  );
}

template <
  class Tensor
>
CUTLASS_DEVICE constexpr
auto convert_to_conversion_tensor(Tensor mma_tensor) {
  return make_tensor(mma_tensor.data(), convert_to_conversion_layout(mma_tensor.layout()));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * copy: tiled copy with optional out-of-bounds checks.
 *
 * Copy source tensor S to destination tensor D using tiled_copy.
 * Both S and D have shape (MMA, MMA_M, MMA_K).
 *
 * Template parameters control boundary checking:
 *   Is_even_MN: if true, no MN boundary check needed (tile fits exactly)
 *   Is_even_K:  if true, no K boundary check needed
 *   Clear_OOB_MN: if true, zero out out-of-bound MN elements in D
 *   Clear_OOB_K:  if true, zero out out-of-bound K elements in D
 *
 * identity_MN: identity tensor for checking MN coordinates
 * predicate_K: boolean tensor for K boundary
 * max_MN: maximum valid MN index
 */
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

}  // namespace flash
