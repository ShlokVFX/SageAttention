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
 * One big "configuration struct" (kernel traits) that bundles ALL type definitions
 * and compile-time constants for the attention kernel.
 *
 * WHY TEMPLATE-BASED CONFIG:
 * Kernel has many variants: different head sizes (64/128), causal/non-causal,
 * BF16/FP16 output, per-block-mean or not.
 * Instead of copy-pasting kernel code for each variant, use C++ templates.
 * Each variant is a different instantiation of Flash_fwd_kernel_traits<...>.
 * Compiler generates optimal code for each variant at compile time.
 *
 * KEY NUMBERS:
 *   kHeadDim: head dimension D (64 or 128)
 *   kBlockM: tile size along Q sequence length (64 or 128 query tokens per tile)
 *   kBlockN: tile size along K sequence length (128 key/value tokens per tile)
 *   kStages: number of pipeline stages for K/V (how many K/V blocks in-flight = 3)
 *   kNWarps: number of warps per thread block (8 for kBlockM=64, 12 for kBlockM=128)
 *     - 4 warps in producer warp group (load data)
 *     - 4 or 8 warps in consumer warp groups (compute MMA)
 *
 * DATA TYPES:
 *   Element = FP4 E2M1 (4-bit float for Q, K, V)
 *   ElementSF = FP8 UE4M3 (8-bit scale factor, one per 16 FP4 values)
 *   ElementAccum = float32 (accumulator for matrix multiply)
 *   ElementOut = BF16 or FP16 (output tensor)
 *
 * TiledMmaQK: matrix multiply Q*K^T
 *   Shape: [kBlockM x kBlockN x kHeadDim] using SM120 FP4 blockscaled atoms
 *   Tiled with 8 atoms in M direction (for kBlockM=128): 8 * 16 = 128 rows
 *
 * TiledMmaPV: matrix multiply P*V
 *   Same atom but different N dimension (kHeadDim instead of kBlockN)
 *   Produces output of shape [kBlockM x kHeadDim]
 *
 * SharedStorage: struct holding ALL shared memory used by kernel.
 *   Located at beginning of GPU shared memory (SMEM) block.
 *   Contains: Q, K, V tiles + scale factors + delta_s + output + pipeline barriers.
 *   alignas(1024): GPU TMA requires 1KB alignment for its buffers.
 *
 * PIPELINE TYPES:
 *   MainloopPipeline: cutlass TMA pipeline for K and V (kStages=3 stages in flight)
 *   MainloopPipelineQ: separate pipeline for Q (only 1 stage needed, loaded once)
 *   EpilogueBarrier: handoff between consumer MMA and epilogue output write
 *
 * SMEM LAYOUT SELECTION:
 *   sm120_rr_smem_selector: picks optimal shared memory layout for SM120 blockscaled MMA.
 *   "rr" = "register-register" (data loaded from SMEM into registers for MMA).
 *   Layout must match hardware requirements for swizzling (bank conflict avoidance).
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "blockscaled_layout.h"
#include "cute_extension.h"
#include "named_barrier.h"
using namespace cute;

/*
 * SharedStorageQKVOwithSF: all shared memory needed by one thread block.
 *
 * Layout of GPU shared memory (SMEM):
 *   [Q tile] [K tile x kStages] [SFQ] [SFK] [SFV] [delta_s x kStages] [V tile x kStages] [O tile]
 *   [pipeline barriers]
 *
 * kStages: number of K/V blocks buffered simultaneously in SMEM (double/triple buffering).
 *   With kStages=3: while consumer processes block N, producer loads blocks N+1 and N+2.
 *   Hides memory latency behind compute.
 *
 * alignas(1024): TMA (Tensor Memory Accelerator) requires 1KB alignment for source/dest buffers.
 * aligned_struct<128>: outer alignment to 128 bytes for cache line alignment.
 *
 * The pipeline barriers live at the END of the shared storage struct:
 *   pipeline_q: barrier for Q loading (1 stage, loaded once)
 *   pipeline_k: barrier for K loading (kStages stages)
 *   pipeline_v: barrier for V loading (kStages stages)
 *   barrier_o: barrier between MMA consumer and epilogue output writer
 *   tile_count_semaphore: for dynamic persistent scheduling
 */
template <
    int kStages,
    int EpiStages,
    typename Element,
    typename ElementSF,
    typename OutputType,
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV
>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0>{

    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;    // Q tile (FP4)
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;    // K tiles (FP4, kStages)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;            // Q scale factors (FP8)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;            // K scale factors (FP8, kStages)
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;            // V scale factors (FP8, kStages)
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;    // delta_s (per-block mean, kStages)
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;    // V tiles (FP4, kStages, transposed)
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o; // O tile (BF16/FP16, for TMA store)

    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;           // Q load pipeline
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;     // K load pipeline
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;     // V load pipeline
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o; // epilogue handoff
        int tile_count_semaphore;                                                               // for dynamic scheduling
    };
  };

/*
 * Flash_fwd_kernel_traits: compile-time configuration for attention kernel.
 *
 * Template Parameters:
 *   kHeadDim_: head dimension (64 or 128)
 *   kBlockM_: tile size in M (query) direction (64 or 128)
 *   kBlockN_: tile size in N (key) direction (128)
 *   kStages_: pipeline stages for K/V loading (3)
 *   kClusterM_: thread block cluster size in M (1 = no clustering)
 *   BlockMean_: whether to use per-block Q mean subtraction for quantization
 *   ElementPairType_: FP4 type (default: E2M1 paired NVFP4)
 *   ElementOut_: output type (default: BF16)
 */
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits {
    static constexpr int kBlockM = kBlockM_;    // queries per tile (M dimension)
    static constexpr int kBlockN = kBlockN_;    // keys per tile (N dimension)
    static constexpr int kHeadDim = kHeadDim_;  // head dimension (K dimension)
    static constexpr bool BlockMean = BlockMean_;  // subtract per-block Q mean?
    static constexpr bool SmoothQ = true;          // always true: use smooth quantization
    static_assert(kHeadDim % 32 == 0);             // head dim must be multiple of 32
    static_assert(kBlockM == 64 || kBlockM == 128); // only 64 or 128 supported

    // kNWarps = total warps per thread block
    // kBlockM=128: 12 warps = 4 producer + 4 consumer0 + 4 consumer1
    // kBlockM=64:   8 warps = 4 producer + 4 consumer0 (only 1 consumer group needed)
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;  // total threads
    static constexpr int kClusterM = kClusterM_;  // cluster size (1 = no multi-block cluster)
    static constexpr int kStages = kStages_;       // K/V pipeline stages (typically 3)
    static constexpr int EpiStages = 1;            // epilogue pipeline stages (always 1)

    // NumSFQK = number of scale factor groups in Q or K (kHeadDim/16 groups along K)
    static constexpr int NumSFQK = kHeadDim / 16;
    // NumSFPV = number of scale factor groups in P (kBlockN/16 groups along N)
    static constexpr int NumSFPV = kBlockN / 16;

    // Data types
    using ElementSF = cutlass::float_ue4m3_t;                 // FP8 unsigned E4M3 for scale factors
    using Element = cutlass::float_e2m1_t;                    // FP4 E2M1 for Q, K, V data
    using ElementAccum = float;                               // FP32 accumulator
    using ElementOut = ElementOut_;                           // BF16 or FP16 output
    using index_t = int64_t;                                  // index type for strides

    static constexpr auto SFVectorSize = 16;  // 16 FP4 values share one scale factor

    // TileShape_MNK: the tile shape for one thread block's work
    //   M = kBlockM queries, N = kBlockN keys, K = kHeadDim head dimension
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;  // no clustering (1 block per cluster)

    // Permutation tiles for tiling MMA atoms over the full tile shape
    using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));  // min(kBlockM, 128)
    using PermTileN = _32;       // MMA atom covers 32 N elements at a time
    using PermTileK = Int<kHeadDim>;  // full K dimension at once

    // MMA input element types (converted from FP4 for hardware MMA)
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    // AtomLayoutMNK: how many MMA atoms in M, N, K directions
    // kBlockM=128: 8 atoms x 16 rows = 128 rows, 1 atom in N and K (tile handles rest)
    // kBlockM=64:  4 atoms x 16 rows = 64 rows
    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>
                                            >;

    // TiledMmaQK: tiled MMA for computing Q * K^T
    //   Atom: SM120 blockscaled FP4 MMA (16x32x64 with UE4M3 scales)
    //   Tiled over: [kBlockM x kBlockN x kHeadDim] using AtomLayoutMNK and permutation tiles
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
      ));

    // TiledMmaPV: tiled MMA for computing P * V
    //   Same atom, but N is now kHeadDim (output head dimension) not kBlockN (keys)
    //   P has shape [kBlockM x kBlockN] and V^T has shape [kBlockN x kHeadDim]
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
      ));

    // MMA_NSF: number of scale factor values per MMA K-atom dimension
    // = atom K size / SFVectorSize = 64 / 16 = 4
    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    // TMA copy types: SM90 TMA (Tensor Memory Accelerator) for async bulk loading
    // SM90 TMA available on SM90+ (Hopper). Blackwell SM120 is backward compatible.
    using GmemTiledCopy = SM90_TMA_LOAD;    // for Q, K, V data
    using GmemTiledCopySF = SM90_TMA_LOAD;  // for scale factors

    // Shared memory layouts for Q, K, V (FP4 data)
    // sm120_rr_smem_selector: picks the SMEM swizzle pattern that avoids bank conflicts
    // for SM120 MMA register-register mode ("rr").
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());

    // Full Q layout: [kBlockM x kHeadDim] (no stages, Q loaded once)
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    // Full K layout: [kBlockN x kHeadDim x kStages] (kStages buffered)
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // Full V layout: [kBlockN x kHeadDim x kStages]
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // Transposed V layout: [kHeadDim x kBlockN x kStages] (V loaded transposed for PV MMA)
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));
    // delta_s (per-block mean) layout: [kBlockM x kBlockN x kStages]
    // Special: stride 0 in M direction = broadcast same value to all M positions
    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS =
        decltype(tile_to_shape(SmemLayoutAtomDS{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // Copy atoms: how to load data from SMEM into registers for MMA
    // SM75_U32x4_LDSM_N: warp-level SMEM->register copy, 4 uint32 values per thread
    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    // UniversalCopy: generic copy for scale factors (byte-granularity)
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    // BlockScaledConfig: scale factor layout helper
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;     // global memory SF layout type
    using SfAtom = typename BlkScaledConfig::SfAtom;         // atomic SF layout tile

    // Shared memory layouts for scale factors (derived from TiledMMA and TileShape)
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));

    // LayoutSFP: layout of P matrix scale factors in registers
    // P has shape (kBlockM, kBlockN), scale factors organized per 64-element block
    using LayoutSFP = decltype(
      make_layout(
          make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
          make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
      )
    );

    // LayoutP: layout of P matrix values in registers (after softmax, before FP4 quant)
    // Organized so FP4 packing is efficient
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );

    // Full SMEM layouts with pipeline stages
    // SFQ: no stages (loaded once with Q)
    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
      ));
    // SFK: kStages pipeline stages
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))
      ));
    // SFV: kStages pipeline stages
    using SmemLayoutSFV = decltype(make_layout(
        append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))
      ));
    // SFVt: kStages pipeline stages (transposed V scale factors)
    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))
      ));

    // Output O layout in shared memory (for TMA store from SMEM to GMEM)
    // ss_smem_selector: picks swizzle for GMMA-K-major output layout
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));

    // SharedStorage: the complete shared memory layout struct
    using SharedStorage = SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS,
        SmemLayoutO, SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

    // Pipeline types for producer-consumer synchronization
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;   // K/V pipeline
    using PipelineState = typename cutlass::PipelineState<kStages>;          // K/V state
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;                  // Q pipeline (1 stage)
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;               // Q state (1 stage)
    // EpilogueBarrier: ordered 2-group barrier (consumer <-> epilogue producer)
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};
