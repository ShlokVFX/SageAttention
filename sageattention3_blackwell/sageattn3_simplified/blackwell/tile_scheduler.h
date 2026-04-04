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
 * Tell each thread block WHICH tile (query block, head, batch) it should compute.
 *
 * WHY NEED SCHEDULER:
 * We have N = (seqlen_q/kBlockM) * nheads * batch total tiles to compute.
 * GPU has M SMs (streaming multiprocessors).
 * Need to assign tiles to SMs without collision and cover all tiles.
 *
 * THREE SCHEDULERS:
 *
 * 1. SingleTileScheduler (simplest):
 *    One GPU block per tile. grid = (num_blocks_m, num_heads, batch).
 *    blockIdx.x = M block index, blockIdx.y = head, blockIdx.z = batch.
 *    Simple but wastes SMs if N < M (some SMs idle).
 *
 * 2. StaticPersistentTileScheduler (USED IN LAUNCH):
 *    grid = (num_SMs,). Each SM gets a "linear tile index" = blockIdx.x.
 *    SM loops: process tile index, then tile index + num_SMs, then + 2*num_SMs, etc.
 *    All SMs stay busy until all tiles done. Better load balancing.
 *    "Static" = tile assignment determined by blockIdx at launch, no runtime coordination.
 *
 * 3. DynamicPersistentTileScheduler:
 *    Like static but uses atomic semaphore to assign tiles dynamically.
 *    SM grabs next available tile from global counter.
 *    Better for irregular workloads (causal attention = variable work per tile).
 *    Slight overhead from atomic operations.
 *
 * WorkTileInfo: struct holding which tile this SM should process.
 *   M_idx = query block index (which kBlockM chunk of Q)
 *   H_idx = head index (which attention head)
 *   B_idx = batch index (which sample in batch)
 *
 * FASTDIVMOD:
 * To convert linear_tile_idx -> (M_idx, H_idx, B_idx), need integer division.
 * cutlass::FastDivmod precomputes multiplication constants so division = multiply+shift.
 * ~4x faster than hardware integer division.
 */

#pragma once

#include "cutlass/fast_math.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////
// OLD/UNUSED StaticPersistentTileSchedulerOld (kept for reference)

class StaticPersistentTileSchedulerOld {
  //
  // Data members
  //

private:
  int current_work_linear_idx_;
  cutlass::FastDivmod const &m_block_divmod, &head_divmod;
  int const total_blocks;

public:
  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

  };

public:

  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerOld(cutlass::FastDivmod const &m_block_divmod_,
                                                        cutlass::FastDivmod const &head_divmod_,
                                                        int const total_blocks_) :
    m_block_divmod(m_block_divmod_), head_divmod(head_divmod_), total_blocks(total_blocks_) {

    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    // current_work_linear_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    current_work_linear_idx_ = blockIdx.x;
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(int linear_idx) const {
    if (linear_idx >= total_blocks) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MHB indices
    int M_idx, H_idx, B_idx;
    int quotient = m_block_divmod.divmod(M_idx, linear_idx);
    B_idx = head_divmod.divmod(H_idx, quotient);
    return {M_idx, H_idx, B_idx, true};
  }

  CUTLASS_DEVICE
  void
  // advance_to_next_work(int advance_count = 1) {
  advance_to_next_work() {
    // current_work_linear_idx_ += int(gridDim.x * gridDim.y * gridDim.z);
    current_work_linear_idx_ += int(gridDim.x);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    WorkTileInfo new_work_tile_info;
    advance_to_next_work();
    new_work_tile_info = get_current_work();
    return new_work_tile_info;
  }

};

///////////////////////////////////////////////////////////////////////////////
/*
 * SingleTileScheduler: simplest scheduler, one GPU block per tile.
 * Grid = (num_blocks_m, num_heads, batch).
 * Each block does exactly one tile and exits.
 * Good for debugging, not optimal for performance.
 */
class SingleTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore = nullptr;
    };

    // Device side kernel params (empty - all info from blockIdx)
    struct Params {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    // Grid = (num_blocks_m, num_heads, batch) - each block does one tile
    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
    }

    struct WorkTileInfo {
        int M_idx = 0;
        int H_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        // Decode (M_idx, H_idx, B_idx) from stored indices
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, H_idx, B_idx};
        }

        // No next work - single tile scheduler does exactly one tile
        CUTLASS_DEVICE
        WorkTileInfo
        get_next_work(Params const& params) const {
            return {-1, -1, -1, false};
        }

    };

    // Initial work = this block's assigned tile (from blockIdx)
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    // No next work
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};

///////////////////////////////////////////////////////////////////////////////
/*
 * StaticPersistentTileScheduler: persistent kernel scheduler.
 * Grid = (num_SMs,). Each SM loops over multiple tiles.
 *
 * Linear tile index -> (M_idx, H_idx, B_idx) via fast integer division.
 * Each SM processes tiles: blockIdx.x, blockIdx.x+num_SMs, blockIdx.x+2*num_SMs, ...
 *
 * "Persistent" = SM stays alive across multiple tiles instead of re-launching.
 * Saves kernel launch overhead and improves SM utilization.
 */
class StaticPersistentTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore = nullptr;
    };

    // Device side kernel params: precomputed fast divmod for tile index decoding
    struct Params {
        int total_blocks;                          // total tiles to process
        cutlass::FastDivmod m_block_divmod;        // for decoding M block index from linear idx
        cutlass::FastDivmod head_divmod;           // for decoding head index from quotient
    };

    // Precompute total tiles and fast divmod constants on host
    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head)};
    }

    // Grid = (num_SMs,): one block per SM
    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;  // linear tile index (encodes M, H, B all in one number)

        // Valid if linear index < total tiles
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        // Decode linear tile_idx -> (M_block, head, batch) using precomputed divmod
        // tile_idx = m_block + num_blocks_m * (bidh + num_heads * bidb)
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
            return {m_block, bidh, bidb};
        }

    };

    // Initial tile = blockIdx.x (SM's starting tile)
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    // Next tile = current + gridDim.x (stride by number of SMs)
    // This ensures each SM processes tiles that are gridDim.x apart
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

/*
 * DynamicPersistentTileScheduler: like static but with dynamic tile assignment.
 * Uses atomic counter (tile_count_semaphore) to assign next available tile.
 * Better load balancing for irregular workloads (causal attention).
 * Currently unused in main path (uses same WorkTileInfo as StaticPersistentTileScheduler).
 */
class DynamicPersistentTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore;  // pointer to GPU-side atomic counter
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        int const* tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    // Reuse StaticPersistentTileScheduler's WorkTileInfo
    using WorkTileInfo = StaticPersistentTileScheduler::WorkTileInfo;

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

} // flash
