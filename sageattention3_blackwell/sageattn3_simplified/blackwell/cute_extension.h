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
 * Define the custom FP4 block-scaled MMA (matrix multiply-accumulate) operation for SM120.
 *
 * WHAT IS MMA ATOM:
 * An MMA atom = the smallest hardware matrix multiply unit.
 * SM120 (Blackwell) has a special instruction: mxf4nvf4 block_scale.
 * This instruction does: D = A(FP4) * B(FP4) + C(F32), with scale factors UE4M3.
 *
 * WHAT IS SM120_16x32x64_TN_VS_NVFP4:
 * One call to this atom computes a 16x32 output tile from a 16x64 A tile and 32x64 B tile.
 * T = row-major (Transposed-not) for A, N = col-major (Non-transposed) for B.
 * VS = vectorized scale factors (one scale per 16 elements = "4X" mode).
 * NVFP4 = NVIDIA FP4 format (E2M1: 2-bit exponent, 1-bit mantissa, 1-bit sign).
 *
 * WHY SO MANY PTX CALLS:
 * The hardware atom is 16x8x64 (16 rows, 8 cols, 64 depth).
 * To get 16x32x64 output, need 4 calls covering 4 different 8-column segments of B.
 * tidB0..tidB3 select which 8-column segment each call works on.
 *
 * WHAT IS TiledMMA:
 * TiledMMA = many atoms tiled together to cover larger matrix tile.
 * Example: AtomLayoutMNK = (8,1,1) means 8 atoms in M direction = 8 x 16 = 128 rows.
 * With Tile<128, 32, kHeadDim>, one TiledMMA covers the full [kBlockM, kBlockN, kHeadDim] tile.
 *
 * WHAT IS MMA_Traits:
 * Tells CuTe how threads map to matrix elements for this atom:
 *   - ALayout: which thread (T) owns which value (V) in A matrix
 *   - BLayout: which thread owns which value in B matrix
 *   - CLayout: which thread owns which value in C/D (accumulator) matrix
 *   - SFALayout/SFBLayout: which thread owns which scale factor
 *
 * WHY PARTITION FUNCTIONS:
 * thrfrg_SFA, partition_SFA etc = partition scale factor tensor across threads.
 * Each thread owns a specific set of scale factors matching the A values it owns.
 * CuTe doesn't know about blockscaled MMA natively, so we add these helpers.
 */

#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace cute::SM120::BLOCKSCALED {

using cutlass::float_e2m1_t;
using cutlass::float_ue4m3_t;

/*
 * SM120_16x32x64_TN_VS_NVFP4: hardware MMA atom for Blackwell FP4 block-scaled GEMM.
 *
 * Register layout:
 *   A (query/key/value in FP4): 4 x uint32 = 4 x 8 FP4 values = 32 FP4 = half of 64-deep tile
 *   B (key/value in FP4): 8 x uint32 = 8 x 8 FP4 = 64 FP4 = full 64-deep tile, 8 cols
 *     (B needs 8 regs to cover all 4 column segments of N=32 width)
 *   C/D (accumulator in F32): 16 x float = 16x8 output (2x8 per atom, 4 atoms = 16x32)
 *   SFA (A scale factors in UE4M3): 1 x uint32 = 4 FP8 scale factors
 *   SFB (B scale factors in UE4M3): 1 x uint32 = 4 FP8 scale factors
 */
struct SM120_16x32x64_TN_VS_NVFP4 {
  using DRegisters = float[16];      // output accumulator: 16 float32 values
  using ARegisters = uint32_t[4];    // FP4 input A: 4 uint32 = 32 FP4 values
  using BRegisters = uint32_t[8];    // FP4 input B: 8 uint32 = 64 FP4 values
  using CRegisters = float[16];      // input accumulator: 16 float32 values

  static constexpr int SFBits = 32;
  using RegTypeSF = cute::uint_bit_t<SFBits>;  // scale factor register type (32-bit)

  using SFARegisters = RegTypeSF[1];  // 1 register = 4 FP8 scale factors for A
  using SFBRegisters = RegTypeSF[1];  // 1 register = 4 FP8 scale factors for B

  CUTE_HOST_DEVICE static void
  fma(float         & d0 , float         & d1 , float         & d2 , float         & d3 ,
      float         & d4 , float         & d5 , float         & d6 , float         & d7 ,
      float         & d8 , float         & d9 , float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      uint32_t const& a0 , uint32_t const& a1 , uint32_t const& a2 , uint32_t const& a3 ,
      uint32_t const& b0 , uint32_t const& b1 , uint32_t const& b2 , uint32_t const& b3 ,
      uint32_t const& b4 , uint32_t const& b5 , uint32_t const& b6 , uint32_t const& b7 ,
      float const   & c0 , float const   & c1 , float const   & c2 , float const   & c3 ,
      float const   & c4 , float const   & c5 , float const   & c6 , float const   & c7 ,
      float const   & c8 , float const   & c9 , float const   & c10 , float const   & c11,
      float const   & c12, float const   & c13, float const   & c14, float const   & c15,
      RegTypeSF const& sfa0,  // scale factor for A (4 FP8 values covering 4 groups of 16)
      RegTypeSF const& sfb0)  // scale factor for B (4 FP8 values)
  {
    // bidA, tidA, bidB: block and thread IDs within scale factor tensor.
    // These are 0 here because each MMA call is self-contained with its own sfa/sfb.
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;
    // tidB0..3: selects which 8-column segment of N=32 tile (4 segments of N=8 each)
    static constexpr uint16_t tidB0 = 0;
    static constexpr uint16_t tidB1 = 1;
    static constexpr uint16_t tidB2 = 2;
    static constexpr uint16_t tidB3 = 3;

#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    // Call 1: compute columns 0-7 of output (d0,d1,d8,d9 using b0,b1 and tidB0)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"   // D: 4 output accumulators
      "{%4,  %5,  %6,  %7},"   // A: 4 FP4 input registers
      "{%8,  %9},"             // B: 2 FP4 input registers (8 cols)
      "{%10, %11, %12, %13},"  // C: 4 input accumulators
      "{%14},"                 // SFA: A scale factors
      "{%15, %16},"            // SFB: B scale factors + bidA,tidA
      "{%17},"                 // bidB
      "{%18, %19};\n"          // tidB + padding
      :  "=f"(d0),  "=f"(d1),  "=f"(d8),  "=f"(d9)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b0),   "r"(b1),
          "f"(c0),   "f"(c1),   "f"(c8),   "f"(c9),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB0));

    // Call 2: compute columns 8-15 of output (d2,d3,d10,d11 using b2,b3 and tidB1)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d2),  "=f"(d3),  "=f"(d10),  "=f"(d11)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b2),   "r"(b3),
          "f"(c2),   "f"(c3),   "f"(c10),   "f"(c11),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB1));

    // Call 3: compute columns 16-23 of output (d4,d5,d12,d13 using b4,b5 and tidB2)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d4),  "=f"(d5),  "=f"(d12),  "=f"(d13)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b4),   "r"(b5),
          "f"(c4),   "f"(c5),   "f"(c12),   "f"(c13),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB2));

    // Call 4: compute columns 24-31 of output (d6,d7,d14,d15 using b6,b7 and tidB3)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d6),  "=f"(d7),  "=f"(d14),  "=f"(d15)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b6),   "r"(b7),
          "f"(c6),   "f"(c7),   "f"(c14),   "f"(c15),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x64_TN_VS without CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif
  }
};

} // namespace cute::SM120::BLOCKSCALED

namespace cute {

/*
 * MMA_Traits specialization: tells CuTe thread-value mapping for SM120 FP4 MMA.
 *
 * CuTe needs to know: for each thread (T) in the warp, which value (V) does it own?
 * This mapping expressed as layouts.
 *
 * ALayout (T32,V32) -> (M16,K64):
 *   32 threads each own 32 values of the A matrix (Q or K in FP4).
 *   Together they cover the full 16x64 A tile.
 *
 * BLayout (T32,V64) -> (N32,K64):
 *   32 threads each own 64 values of B (K^T or V^T in FP4).
 *   Together they cover the full 32x64 B tile.
 *
 * CLayout (T32,V16) -> (M16,N32):
 *   32 threads each own 16 float32 accumulator values.
 *   Together they cover the full 16x32 output tile.
 *
 * SFALayout (T32,V64) -> (M16,K64):
 *   Scale factors for A. One scale factor per 16 A values.
 *   So M16,K64 A tile has 1 x 4 = 4 scale factors.
 *   Layout tells where each scale lives in register file.
 *
 * SFBLayout (T32,V64) -> (N32,K64):
 *   Scale factors for B. Similar to SFA.
 */
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>
{
  // The MMA accepts 4-bit inputs regardless of the types for A and B
  using ValTypeA = uint4_t;        // 4-bit values for A matrix (FP4 E2M1)
  using ValTypeB = uint4_t;        // 4-bit values for B matrix (FP4 E2M1)

  using ValTypeD = float;          // 32-bit accumulator output
  using ValTypeC = float;          // 32-bit accumulator input

  using ValTypeSF = cutlass::float_ue4m3_t;  // FP8 UE4M3 scale factors
  constexpr static int SFVecSize = 16;        // one scale factor per 16 FP4 elements

  using Shape_MNK = Shape<_16,_32,_64>;  // atom tile shape: M=16 rows, N=32 cols, K=64 depth
  using ThrID     = Layout<_32>;         // 32 threads per warp participate

  // (T32,V32) -> (M16,K64): Thread-Value to Matrix coordinate mapping for A
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  // (T32,V64) -> (N32,K64): Thread-Value to Matrix coordinate mapping for B
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2, _4>>,
                           Stride<Stride<_256,_1>,Stride<_32,_1024, _8>>>;
  // (T32,V64) -> (M16,K64): Thread-Value to Matrix coordinate mapping for A scale factors
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>,
                           Stride<Stride<_8,_0,_1>,_16>>;
  // (T32,V64) -> (N32,K64): Thread-Value to Matrix coordinate mapping for B scale factors
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,
                           Stride<Stride<_8,_1>, _32>>;
  // (T32,V16) -> (M16,N32): Thread-Value to Matrix coordinate mapping for C/D accumulator
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < Shape<_2, _4>,_2>>,
                              Stride<Stride<_32,_1>,Stride<Stride<_16, _128>,_8>>>;
};

/*
 * thrfrg_SFA: partition scale factor A tensor by threads and value fragments.
 *
 * This is the scale factor version of what TiledMMA::thrfrg_A does for the data tensor.
 * Given a scale factor tensor of shape (M, K) and a TiledMMA, returns a view
 * ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK))) that partitions by thread ID and value index.
 *
 * Steps:
 * 1. logical_divide by TiledPerm to reorder M,K dimensions to match atom permutation
 * 2. zipped_divide by AtomShape to cut into atom-sized tiles
 * 3. .compose(AtomLayoutSFA_TV) to transform (M,K) -> (Thr,Val)
 * 4. zipped_divide by thread tile to separate (Thr, Rest)
 */
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Reorder the tensor for the TiledAtom (apply M,K permutation)
  auto t_tile = make_tile(get<0>(permutation_mnk),
                          get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

  // Tile the tensor for the Atom (divide into atom-sized blocks)
  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

  // Transform the Atom mode from (M,K) to (Thr,Val) using hardware layout
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

  // Tile the tensor for the Thread (separate each thread's values from rest)
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

  return thr_tensor;
}

/* Same as thrfrg_SFA but for B (Key/Value) scale factors */
template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // N,K dimensions for B (vs M,K for A)
  auto t_tile = make_tile(get<1>(permutation_mnk),
                          get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

/*
 * partition_SFA: get the slice of SFA tensor for this specific thread.
 * Returns a tensor view that this thread owns (rows = values, cols = rest of tile).
 */
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

/*
 * partition_fragment_SFA: like partition_SFA but returns a register fragment (in registers, not shared mem).
 * Used to create the tSFQrSFQ (register copy of shared memory scale factors) fragment.
 */
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFA(sfatensor, thread_mma));
}

/* Same as partition_SFA but for B scale factors */
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

/* Same as partition_fragment_SFA but for B */
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFB(sfbtensor, thread_mma));
}

/*
 * get_layoutSFA_TV: get the full (thread, value) -> (M, K) layout for SFA.
 * Used to figure out which elements each thread reads from shared memory.
 * Returns a layout mapping (thread_idx, value_idx) -> position in (M, K) matrix.
 */
template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFA_TV(TiledMma& mma)
{
  // (M,K) -> (M,K) identity reference
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Expand (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK)) by adding N broadcast dimension
  auto atile = make_tile(_,
                        make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                              make_stride(               Int<1>{} ,                Int<0>{} )),
                                  _));

  // Convert thread index to (ThrV,ThrM,ThrN,ThrK) coordinates
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  // Compose to get (thr_idx,val) -> (M,K) mapping
  return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

/* Same as get_layoutSFA_TV but for B scale factors: (thr_idx,val) -> (N,K) */
template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFB_TV(TiledMma& mma)
{
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto btile = make_tile(_,
                        make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                              make_stride(               Int<0>{} ,                Int<1>{} )),
                                  _));

  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

} // namespace cute
