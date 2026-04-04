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
 * Make synchronization barrier between warp groups (producer and consumer).
 *
 * WHY NEED BARRIER:
 * Kernel split into warp groups with different jobs:
 *   - Producer warp group: load data (Q, K, V) from global to shared memory
 *   - Consumer warp group: do matrix multiply (MMA) on that data
 *   - Epilogue producer: write output back to global memory
 *
 * Without barrier, consumer might start computing before producer finish loading.
 * That give garbage result. Barrier make consumer WAIT until producer say "done".
 *
 * ORDERED SEQUENCE BARRIER:
 * This is special barrier for epilogue handoff:
 *   - After consumer finish computing tile -> signal epilogue producer to write output
 *   - After epilogue producer finish writing -> signal consumer ready for next tile
 * Groups take turns in order. Group 0 signal group 1, group 1 signal group 0, etc.
 *
 * HOW CLUSTER BARRIER WORK:
 * Hardware barrier that can count arrivals from threads. When all expected threads arrive,
 * everyone waiting is released. Phase bit (0/1) handle multiple uses without reset.
 *
 * SequenceDepth = pipeline depth (how many stages in flight)
 * SequenceLength = number of groups (2: consumer group and epilogue group)
 */

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace flash {

/* Named barriers for synchronization. Integer IDs mapped to semantic names. */
enum class FP4NamedBarriers {
    QueryEmpty = 1,                       // Q shared memory slot is empty (ready for new data)
    WarpSpecializedConsumer = 2,          // Consumer warp group sync point
    WarpSpecializedPingPongConsumer1 = 3, // Ping-pong consumer 1
    WarpSpecializedPingPongConsumer2 = 4, // Ping-pong consumer 2
    ProducerEnd = 5,                      // Producer warp group finished all tiles
    ConsumerEnd = 6,                      // Consumer warp group finished all tiles
    EpilogueBarrier = 7                   // Barrier between compute and output write
};

/* Shared memory storage for the barrier array. Lives in GPU shared memory. */
template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierVarGroupSizeSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  // 2D array: [pipeline_stage][group_id]
  // Each cell = one hardware barrier object
  Barrier barrier_[SequenceDepth][SequenceLength];
};

/*
 * OrderedSequenceBarrierVarGroupSize:
 * Two groups (consumer and epilogue) take turns signaling each other.
 * Group 0 arrives -> releases Group 1 to proceed.
 * Group 1 arrives -> releases Group 0 to proceed.
 *
 * "VarGroupSize" = each group can have different number of threads.
 * Consumer group has NumMmaThreads threads, epilogue group has 32 (one warp).
 */
template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrierVarGroupSize {
public:
  static constexpr int SequenceDepth = SequenceDepth_;   // pipeline depth (1 here)
  static constexpr int SequenceLength = SequenceLength_; // number of groups (2 here)
  using Barrier = cutlass::arch::ClusterBarrier;
  using SharedStorage = flash::OrderedSequenceBarrierVarGroupSizeSharedStorage<SequenceDepth, SequenceLength>;

  struct Params {
    uint32_t group_id;          // which group am I? (0=epilogue producer, 1=consumer)
    uint32_t* group_size_list;  // array of thread counts per group
  };

private:
  Params params_;
  Barrier *barrier_ptr_;                          // pointer into shared memory barrier array
  cutlass::PipelineState<SequenceDepth> stage_;   // tracks which pipeline stage we're on

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  // Delete copy/move constructors - barrier should not be copied
  OrderedSequenceBarrierVarGroupSize() = delete;
  OrderedSequenceBarrierVarGroupSize(const OrderedSequenceBarrierVarGroupSize&) = delete;
  OrderedSequenceBarrierVarGroupSize(OrderedSequenceBarrierVarGroupSize&&) = delete;
  OrderedSequenceBarrierVarGroupSize& operator=(const OrderedSequenceBarrierVarGroupSize&) = delete;
  OrderedSequenceBarrierVarGroupSize& operator=(OrderedSequenceBarrierVarGroupSize&&) = delete;
  ~OrderedSequenceBarrierVarGroupSize() = default;

  CUTLASS_DEVICE
  OrderedSequenceBarrierVarGroupSize(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 starts with opposite phase so it waits immediately
      // Group 1 starts with correct phase so it runs first
      stage_({0, params.group_id == 0, 0}) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    // Only ONE thread initializes barriers (warp 0, lane 0)
    // barrier.init(N) = "wait for N threads to arrive before releasing"
    if (warp_idx == 0 && lane_predicate) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(*(params.group_size_list + l));
        }
      }
    }
    // Memory fence to ensure barrier init is visible to all threads
    cutlass::arch::fence_barrier_init();
  }

  // WAIT: block until my group's barrier is released
  // Called by a group to wait for the other group to signal it
  CUTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // ARRIVE: signal the NEXT group that I'm done, then move to next stage
  // (group_id) signals to (group_id+1 mod Length)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;  // advance pipeline stage counter
  }

  // ADVANCE: just move to next stage without signaling
  CUTLASS_DEVICE
  void advance() {
    ++stage_;
  }

private:
  // Get the barrier for current pipeline stage and given group
  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

} // namespace flash
