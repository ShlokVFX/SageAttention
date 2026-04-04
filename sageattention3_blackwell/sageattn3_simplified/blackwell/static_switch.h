// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

/*
 * WHAT THIS FILE DO:
 * Convert runtime values (bool/int) into compile-time template constants.
 *
 * WHY NEED THIS:
 * C++ templates need constant values known at compile time.
 * But user passes is_causal=True at runtime.
 * BOOL_SWITCH(is_causal, Is_causal, ...) creates TWO code paths:
 *   - one with constexpr Is_causal=true (compiler optimizes knowing it's causal)
 *   - one with constexpr Is_causal=false
 * Then picks correct path at runtime based on actual value.
 * Result: optimized kernel for each case, no runtime branch inside kernel.
 *
 * HEADDIM_SWITCH: similar but for head dimensions (64, 128, 256).
 * Each headdim gets its own kernel with sizes baked in at compile time.
 * Saves registers and allows unrolling.
 *
 * USAGE:
 * BOOL_SWITCH(params.is_causal, Is_causal, [&] {
 *     run_flash_fwd<Is_causal>(...);  // Is_causal is compile-time here!
 * });
 *
 * HEADDIM_SWITCH(d, [&] {
 *     run_flash_fwd<kHeadSize>(...);  // kHeadSize baked in at compile time
 * });
 */

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

/* Runtime bool -> compile-time constexpr bool dispatch */
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

/* Runtime precision type -> compile-time type dispatch (1=FP16, 2=FP8, 3=FP8hybrid, 4=FP8soft) */
#define PREC_SWITCH(PRECTYPE, ...)                                             \
  [&] {                                                                        \
    if (PRECTYPE == 1) {                                                       \
      using kPrecType = cutlass::half_t;                                       \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 2) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 3) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = true;                                    \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 4) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = true;                                  \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

/* Runtime head dimension -> compile-time int kHeadSize dispatch */
#define HEADDIM_SWITCH(HEADDIM, ...)                                           \
  [&] {                                                                        \
    if (HEADDIM == 64) {                                                       \
      constexpr static int kHeadSize = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 128) {                                               \
      constexpr static int kHeadSize = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 256) {                                               \
      constexpr static int kHeadSize = 256;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

/* Runtime variable/fixed seqlen -> compile-time type dispatch */
#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, SEQ_LEN_OUT_OF_BOUND_CHECK, ...)        \
  [&] {                                                                        \
    if (!USE_VAR_SEQ_LEN) {                                                    \
      if (SEQ_LEN_OUT_OF_BOUND_CHECK) {                                        \
        using kSeqLenTraitsType = FixedSeqLenTraits<true>;                     \
        return __VA_ARGS__();                                                  \
      } else {                                                                 \
        using kSeqLenTraitsType = FixedSeqLenTraits<false>;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    } else {                                                                   \
      using kSeqLenTraitsType = VarSeqLenTraits;                               \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
