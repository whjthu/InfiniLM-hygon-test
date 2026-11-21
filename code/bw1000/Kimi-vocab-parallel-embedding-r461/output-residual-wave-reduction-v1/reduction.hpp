#pragma once

#include "score_math.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define KIMI_REDUCE_DEVICE __device__
#else
#define KIMI_REDUCE_DEVICE
#endif

namespace kimi_output_residual_scores_v1 {
inline constexpr unsigned shuffle_width = 32;
inline constexpr unsigned shuffle_threads = 64;
static_assert(lanes == 256 && shuffle_width * 2 == shuffle_threads);

KIMI_REDUCE_DEVICE inline float reduce_pair(float value, float neighbor, unsigned lane, unsigned stride) {
    return lane < stride ? add(value, neighbor) : value;
}
}

#undef KIMI_REDUCE_DEVICE
