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
 * The main GPU kernel that coordinates all warp groups.
 *
 * WARP SPECIALIZATION ("_ws" suffix):
 * Split thread block into warp groups with different roles.
 * Each warp group does specific job in parallel.
 *
 * WARP GROUPS:
 *
 * Producer (warp group 0, 4 warps = 128 threads):
 *   - Has TWO sub-roles split by warp index within group:
 *   - Mainloop warp (warp 0): loads Q, K, V, scale factors using TMA
 *     TMA = Tensor Memory Accelerator (hardware async bulk copy engine)
 *     While consumer computes MMA, producer loads NEXT K/V tile
 *     This overlaps compute and memory = hides latency
 *   - Epilogue warp (warp 1): waits for consumer to write O to SMEM,
 *     then uses TMA to write O from SMEM to global memory (GMEM)
 *
 * Consumer0 (warp group 1, 4 warps = 128 threads):
 *   - Does the actual attention computation for top half of tile (M rows 0..kBlockM/2-1)
 *   - Q*K^T matmul + softmax + P*V matmul
 *   - Writes result O to shared memory, signals epilogue warp
 *
 * Consumer1 (warp group 2, 4 warps = 128 threads) - only for kBlockM=128:
 *   - Same as Consumer0 but for bottom half of tile (M rows kBlockM/2..kBlockM-1)
 *   - Both consumers run in parallel
 *
 * NOTE for kBlockM=64:
 *   Only 8 warps total: 4 producer + 4 consumer0 (no consumer1)
 *
 * PIPELINE FLOW:
 *   Producer: load Q tile via TMA, signal "Q ready"
 *   Producer: for each K block:
 *               load K tile + scale via TMA, signal "K ready"
 *               load V tile + scale via TMA, signal "V ready"
 *   Consumer: wait "Q ready", wait "K ready", do QK matmul
 *   Consumer: apply softmax + FP4 quantize
 *   Consumer: wait "V ready", do PV matmul
 *   Consumer: write O to SMEM, signal epilogue
 *   Epilogue: wait signal, TMA store O from SMEM to GMEM, signal back
 *
 * REG ALLOC:
 *   Producer: warpgroup_reg_dealloc<24> - give registers back (only 24 regs needed)
 *   Consumer: warpgroup_reg_alloc<232>  - take extra registers (need 232 for MMA accumulators)
 *   Total registers per SM = 64K. Balancing producers/consumers matters for occupancy.
 *
 * PERSISTENT KERNEL:
 *   Consumers loop over tiles (outer for-loop with scheduler).
 *   Same thread block processes multiple (M, head, batch) tiles sequentially.
 *   Saves SM launch overhead vs launching new kernel per tile.
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "params.h"
#include "utils.h"
#include "tile_scheduler.h"
#include "mainloop_tma_ws.h"
#include "epilogue_tma_ws.h"
#include "named_barrier.h"
#include "softmax_fused.h"

namespace flash {

using namespace cute;

/*
 * compute_attn_ws: the main attention kernel.
 *
 * __launch_bounds__(kNWarps * 32, 1):
 *   First arg = max threads per block (kNWarps * 32).
 *   Second arg = min blocks per SM (1). Set to 1 because we use persistent scheduling.
 *
 * CUTE_GRID_CONSTANT: tells compiler these params are constant for whole kernel launch.
 *   Allows storing in constant memory for faster access.
 */
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                    ) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMmaQK{});  // threads doing MMA
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;       // threads doing TMA load (1 warp group)
    static constexpr int kBlockM = Ktraits::kBlockM;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    // Warp group roles (each warp group = 4 warps = 128 threads)
    enum class WarpGroupRole {
        Producer = 0,    // warp group 0: load data from GMEM -> SMEM
        Consumer0 = 1,   // warp group 1: compute MMA (top half of M tile)
        Consumer1 = 2    // warp group 2: compute MMA (bottom half, only if kBlockM=128)
    };
    // Sub-roles within producer warp group
    enum class ProducerWarpRole {
        Mainloop = 0,    // warp 0 of producer: TMA load Q/K/V
        Epilogue = 1,    // warp 1 of producer: TMA store O
        Warp2 = 2,       // warps 2,3: unused (register deallocated)
        Warp3 = 3
    };

    // Shared memory for all tensors and pipeline barriers
    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    // Identify this thread's role
    int const lane_predicate = cute::elect_one_sync();  // 1 if this is lane 0 of warp
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();  // 0, 1, or 2
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

    // ONE thread prefetches TMA descriptors into L2 cache for faster access later
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Setup V pipeline (shared by producer and consumer)
    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;  // bytes per TMA load
    pipeline_params_v.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = warp_group_thread_idx == 0;   // one leader per warp group
    pipeline_params_v.num_consumers = NumMmaThreads;             // how many threads will consume

    // Setup K pipeline
    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumMmaThreads;

    // Setup Q pipeline (single-stage, Q loaded once per tile)
    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_params_q.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipelineQ::ThreadCategory::Producer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumMmaThreads;

    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    // Setup epilogue barrier: two groups (epilogue producer=group0, consumers=group1)
    // Group 0 = epilogue producer warp (32 threads)
    // Group 1 = consumer warp groups (NumMmaThreads threads)
    uint32_t epilogue_barrier_group_size_list[2] = {cutlass::NumThreadsPerWarp, NumMmaThreads};
    typename EpilogueBarrier::Params params_epilogue_barrier;
    params_epilogue_barrier.group_id = (warp_group_role == WarpGroupRole::Producer);  // 1 for producer, else 0
    params_epilogue_barrier.group_size_list = epilogue_barrier_group_size_list;
    EpilogueBarrier barrier_o(shared_storage.barrier_o, params_epilogue_barrier);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;
    __syncthreads();  // ensure all shared memory initialized before use

    if (warp_group_role == WarpGroupRole::Producer) {
        // ====== PRODUCER WARP GROUP ======
        // Give unused registers back to GPU register file (only need 24 for TMA)
        cutlass::arch::warpgroup_reg_dealloc<24>();
        TileScheduler scheduler;

        if (producer_warp_role == ProducerWarpRole::Mainloop) {
            // ---- MAINLOOP PRODUCER (warp 0 of producer group) ----
            // Loads Q, K, V, scale factors for all tiles this block is responsible for
            PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
                int tile_count_semaphore = 0;
                // load() performs all TMA copies for this tile's Q, K, V sequence
                // Uses pipeline to signal consumer when each K/V block is ready
                collective_mainloop.load(mainloop_params, scheduler_params,
                                         pipeline_q, pipeline_k, pipeline_v,
                                         smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v,
                                         shared_storage, work_tile_info, work_idx, tile_count_semaphore);
            }
            // Signal consumers that no more data is coming
            collective_mainloop.load_tail(pipeline_q, pipeline_k, pipeline_v,
                                          smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v);

        } else if (producer_warp_role == ProducerWarpRole::Epilogue) {
            // ---- EPILOGUE PRODUCER (warp 1 of producer group) ----
            // Waits for consumer to write O to SMEM, then TMA stores to GMEM
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
                barrier_o.wait();   // wait for consumer to write O to SMEM
                collective_epilogue.tma_store(shared_storage, epilogue_params, work_tile_info, scheduler_params, threadIdx.x);
                collective_epilogue.store_tail();  // wait for TMA store to complete
                barrier_o.arrive(); // signal consumer that O slot is free for next tile
            }
        }
        // Warp2 and Warp3 of producer group: do nothing (just use minimal registers)

    } else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
        // ====== CONSUMER WARP GROUPS ======
        // Take extra registers from GPU register file (need 232 for MMA accumulators)
        // The producer dealloc'd registers to make room for these
        cutlass::arch::warpgroup_reg_alloc<232>();

        typename Ktraits::TiledMmaPV tiled_mma_pv;  // for P*V computation
        TileScheduler scheduler{};
        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        PipelineStateQ smem_pipe_read_q;

        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL  // don't unroll outer tile loop (unknown iteration count)
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

            // Allocate output accumulator O (starts zeroed)
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_MNK{}));

            // Softmax state tracker (row-wise max and sum, updated per K block)
            flash::SoftmaxFused<2 * (2 * kBlockM / NumMmaThreads)> softmax_fused;

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            // For causal attention: check if this Q tile has any valid K tiles
            // (Q tokens can only attend to past K tokens in causal mode)
            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {
                // All K tokens are in the future -> output is zero (attend to nothing)
                collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord);
                continue;
            }

            // Main attention computation:
            // 1. Wait for Q to be loaded
            // 2. For each K block: wait K loaded, do QK MMA, softmax, wait V loaded, do PV MMA
            // 3. Finalize softmax (divide by sum)
            collective_mainloop.mma(mainloop_params, pipeline_q, pipeline_k, pipeline_v,
                                    smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax_fused, n_block_max,
                                    threadIdx.x - NumCopyThreads,  // thread idx within consumer group
                                    work_idx, m_block, shared_storage);

            // Wait for epilogue warp to finish writing previous tile's O
            barrier_o.wait();
            // Write O from registers to shared memory (for TMA store)
            collective_epilogue.mma_store(shared_storage, tiled_mma_pv, tOrO, threadIdx.x - NumCopyThreads);
            // Signal epilogue warp that O is ready in SMEM
            barrier_o.arrive();
            ++work_idx;
        }
    }
}

} // namespace flash
