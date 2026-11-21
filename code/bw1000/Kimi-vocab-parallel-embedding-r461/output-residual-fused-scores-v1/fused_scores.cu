#include <cuda_runtime.h>
#include <cmath>
#include "score_math.hpp"

namespace {
using namespace kimi_output_residual_scores_v1;

__device__ void reduce(float* scratch, unsigned lane) {
    for (unsigned stride = lanes / 2; stride; stride /= 2) {
        if (lane < stride) scratch[lane] = add(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
}

__global__ void scores_kernel(std::uint16_t* output, const std::uint16_t* values,
                              const std::uint16_t* norm, const std::uint16_t* projection, float eps) {
    __shared__ float scratch[lanes];
    __shared__ float inv_rms;
    const unsigned lane = threadIdx.x;
    const auto* row = values + std::size_t(blockIdx.x) * width;
    scratch[lane] = square_lane(row, lane);
    __syncthreads();
    reduce(scratch, lane);
    if (lane == 0) inv_rms = 1.0f / sqrtf(add(mul(scratch[0], 1.0f / float(width)), eps));
    __syncthreads();
    scratch[lane] = dot_lane(row, norm, projection, inv_rms, lane);
    __syncthreads();
    reduce(scratch, lane);
    if (lane == 0) output[blockIdx.x] = encode(scratch[0]);
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
