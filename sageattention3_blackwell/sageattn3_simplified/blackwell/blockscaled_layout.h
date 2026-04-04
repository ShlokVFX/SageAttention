/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Blocked Scale configs specific for SM100 BlockScaled MMA
*/

/*
 * WHAT THIS FILE DO:
 * Define the shared memory layout for FP4 scale factors.
 *
 * WHY SPECIAL LAYOUT:
 * Scale factors not stored same way as regular tensor.
 * Block-scaled format = every 64 rows (or cols) share 4 scale factors.
 * These 4 scale factors packed together in memory for efficient loading.
 *
 * WHAT IS SfAtom (scale factor atom):
 * Smallest repeating unit of scale factor storage.
 * Shape = ((16,4), (SFVecSize, 4))
 *   (16,4) = 64 elements in MN direction (rows or cols)
 *   (SFVecSize, 4) = 16 elements per scale group, 4 groups per 64-element block
 * Stride = ((16,4), (0,1))
 *   stride 0 in SFVecSize direction = same scale shared by all 16 elements in group
 *   stride 1 in groups direction = consecutive scale factors
 *
 * THINK OF IT LIKE:
 * 64 elements in a row -> divided into 4 groups of 16.
 * Each group gets 1 scale factor. 4 scale factors total for 64 elements.
 * Scale factors stored contiguously: [sf0, sf1, sf2, sf3] for each 64-element block.
 *
 * SMEM LAYOUT DEDUCTION:
 * deduce_smem_layoutSFQ/SFK/SFV = compute exact shared memory layout
 * that matches what the blockscaled MMA hardware expects.
 * This is critical: if layout wrong, MMA reads wrong scale factors = garbage output.
 *
 * BlockScaledConfig<16>:
 *   SFVecSize=16 = one scale factor for every 16 FP4 values
 *   Blk_MN=64 = one "block" in M or N direction = 64 elements
 *   Blk_SF=4 = 4 scale factors per block (64/16 = 4)
 */

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace flash {

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

/*
 * BlockScaledBasicChunk: the atomic layout unit for scale factors.
 * Blk_MN=64: 64 elements in one "block" (a row or column segment).
 * Blk_SF=4: 4 scale factors per block (one per 16 elements).
 *
 * SfAtom layout: maps (mn_position, k_position) -> scale_factor_index
 *   mn shape (16,4): 16 elements per group * 4 groups = 64 elements total
 *   k  shape (SFVecSize, 4): SFVecSize elements per scale group, 4 groups in K
 *   mn stride (16,4): position within row-block
 *   k  stride (0,1): 0 = all K elements in group share scale, 1 = consecutive scales
 */
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct BlockScaledBasicChunk {

  using Blk_MN    = _64;   // 64 elements per M/N block
  using Blk_SF    =   _4;  // 4 scale factors per block

  using SfAtom  = Layout< Shape< Shape<_16,_4>, Shape<Int<SFVecSize>, _4>>,
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
};

/*
 * BlockScaledConfig<SFVecSize>: full configuration for block-scaled MMA layouts.
 *
 * Provides static functions to compute shared memory layouts for Q, K, V scale factors.
 * These layouts must match exactly what the SM120 blockscaled MMA hardware expects.
 *
 * KEY NUMBERS:
 *   SFVecSize = 16: one FP8 scale per 16 FP4 values
 *   MMA_NSF = 4: number of scale factor groups per MMA K-dimension slice
 *   Blk_MN = 64: elements per M/N block
 *   Blk_SF = 4: scale factors per M/N block
 */
template<int SFVecSize_>
struct BlockScaledConfig {
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr int MMA_NSF = 4;  // scale factors per MMA K-dimension tile
  using BlkScaledChunk = BlockScaledBasicChunk<SFVecSize>;
  using Blk_MN    = _64;   // 64 elements per M/N block
  using Blk_SF    =   _4;  // 4 scale factors per M/N block
  // Innermost M/N block shape: (16 elements, 4 groups) = 64 total
  using mnBasicBlockShape  =  Shape<_16,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  // Innermost K block shape: (SFVecSize elements per group, 4 groups) -> stride (0, 1)
  // stride 0 = broadcast same scale across SFVecSize elements
  // stride 1 = 4 consecutive scale values in K direction
  using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;
  // SfAtom = Layout of one fundamental scale factor block
  using SfAtom  = Layout< Shape< mnBasicBlockShape, kBasicBlockShape>,
                          Stride<mnBasicBlockStride, kBasicBlockStride>>;

  // LayoutSF: complete global memory layout for scale factor tensor
  // Shape (seqlen, dim//16, nhead, batch) with blocked_product applied
  using LayoutSF = decltype(blocked_product(SfAtom{},
                                make_layout(
                                    make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                                    make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));
  // Blk_Elems = 64 * 4 = 256 elements per "super-block" (M*SF in MN direction)
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  // sSF_strideMN: stride for moving between M/N super-blocks in shared memory
  using sSF_strideMN = decltype(prepend(Blk_Elems{},  mnBasicBlockStride{}));


  // tile_atom_to_shape_SFQKV: map global Q/K scale factor tensor to full layout.
  // Input: problem_shape = (seqlen, dim, nheads, batch)
  // Step<_2,_1,_3,_4> = ordering of dimensions (dim is innermost after seqlen)
  template < class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
    auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2,_1,_3,_4>{});
  }

  // tile_atom_to_shape_SFVt: map transposed V scale factor tensor to full layout.
  // V is transposed (D, seqlen) vs Q/K which are (seqlen, D).
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
    auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2,_1,_3,_4>{});
  }

  /*
   * deduce_smem_layoutSFQ: compute shared memory layout for Q scale factors.
   *
   * Scale factors for Q need to be in specific layout that blockscaled MMA can load.
   * Shape organized as (M_groups, K_groups) where each "group" = one 64-element block.
   * Within each group: 4 scale factors stored in (16, 4) MN block * (SFVecSize, 4) K block layout.
   *
   * sSFQ_shapeM: M dimension organized as (inner_block, outer_groups)
   *   inner_block = (16, 4) = 64 elements
   *   outer_groups = kBlockM/64 = number of 64-element M blocks
   * sSFQ_shapeK: K dimension organized as (inner_block, outer_groups)
   *   inner_block = (SFVecSize, 4) with Blk_SF/MMA_NSF factor
   *   outer_groups = kHeadDim/SFVecSize/Blk_SF groups
   */
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFQ(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    using sSFQ_shapeK = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFQ_strideM = sSF_strideMN;
    using sSFQ_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
    using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{},  sSFQ_stride{}));
    return SmemLayoutAtomSFQ{};
  }

  /*
   * deduce_smem_layoutSFKV: compute shared memory layout for K or V scale factors.
   * Similar to SFQ but uses N dimension (kBlockN) instead of M dimension (kBlockM).
   */
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFKV(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    using sSFK_shapeK = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFK_strideN = sSF_strideMN;
    using sSFK_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
    using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
    using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));
    return SmemLayoutAtomSFK{};
  }

  /*
   * deduce_smem_layoutSFVt: compute shared memory layout for transposed V scale factors.
   * V stored transposed (D, seqlen) instead of (seqlen, D).
   * N dimension in tile space maps to seqlen (K in V^T multiply).
   */
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFVt(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    using sSFVt_shapeK = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFVt_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFVt_strideN = sSF_strideMN;
    using sSFVt_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
    using sSFVt_stride = decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
    using SmemLayoutAtomSFVt = decltype(make_layout(sSFVt_shape{}, sSFVt_stride{}));
    return SmemLayoutAtomSFVt{};
  }
};


} // namespace flash
