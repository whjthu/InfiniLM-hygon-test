#include <cuda_runtime.h>
#include <cmath>
#include "reduction.hpp"

namespace {
using namespace kimi_output_residual_scores_v1;

__device__ float reduce(float* scratch, unsigned lane) {
    for (unsigned stride = lanes / 2; stride >= shuffle_width; stride /= 2) {
        if (lane < stride) scratch[lane] = reduce_pair(scratch[lane], scratch[lane + stride], lane, stride);
        __syncthreads();
    }
    // All lanes in the first 64-thread AMD wave, or both NVIDIA warps, participate.
    // Within the first 32 lanes the old tree still uses offsets 16, 8, 4, 2, 1.
    if (lane < shuffle_threads) {
        float value = scratch[lane];
        for (unsigned stride = shuffle_width / 2; stride; stride /= 2) {
#if defined(__HIPCC__)
            const float neighbor = __shfl_down(value, stride, shuffle_width);
#else
            const float neighbor = __shfl_down_sync(0xffffffffu, value, stride, shuffle_width);
#endif
            value = reduce_pair(value, neighbor, lane, stride);
        }
        return value;
    }
    return 0.0f;
}

__global__ void scores_kernel(std::uint16_t* output, const std::uint16_t* values,
                              const std::uint16_t* norm, const std::uint16_t* projection, float eps) {
    __shared__ float scratch[lanes];
    __shared__ float inv_rms;
    const unsigned lane = threadIdx.x;
    const auto* row = values + std::size_t(blockIdx.x) * width;
    scratch[lane] = square_lane(row, lane);
    __syncthreads();
    const float squares = reduce(scratch, lane);
    if (lane == 0) inv_rms = 1.0f / sqrtf(add(mul(squares, 1.0f / float(width)), eps));
    __syncthreads();
    scratch[lane] = dot_lane(row, norm, projection, inv_rms, lane);
    __syncthreads();
    const float dot = reduce(scratch, lane);
    if (lane == 0) output[blockIdx.x] = encode(dot);
}
}

extern "C" __attribute__((visibility("hidden"))) int kimi_output_residual_scores_v1_launch(
    void* output, const void* values, const void* norm, const void* projection, float eps, void* stream) {
    if (!output || !values || !norm || !projection || !std::isfinite(eps) || eps <= 0) return 1;
    using Word = std::uint16_t;
    scores_kernel<<<kimi_output_residual_scores_v1::rows, kimi_output_residual_scores_v1::lanes,
                    0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<Word*>(output), static_cast<const Word*>(values),
        static_cast<const Word*>(norm), static_cast<const Word*>(projection), eps);
    return int(cudaGetLastError());
}
