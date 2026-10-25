#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

// Fixed component geometry only. No Softmax or weighted-GEMM fusion.
// BF16 normalized values are materialized in registers before projection.
namespace {
__device__ float reduce256(float v, float *tmp) {
    const int t = threadIdx.x;
    tmp[t] = v; __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) tmp[t] += tmp[t + s];
        __syncthreads();
    }
    return tmp[0];
}
__global__ void front(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                      const __nv_bfloat16 *p, __nv_bfloat16 *scores,
                      float eps) {
    constexpr int H = 7168, C = 24, CAP = 25;
    const int row = blockIdx.x, t = threadIdx.x;
    const size_t offset = (size_t(row / C) * CAP + row % C) * H;
    __shared__ float tmp[256];
    float sum = 0.f;
    for (int h = t; h < H; h += 256) {
        const float v = __bfloat162float(x[offset + h]);
        sum += v * v;
    }
    const float ss = reduce256(sum, tmp);
    __shared__ float inv;
    if (!t) inv = rsqrtf(ss / float(H) + eps);
    __syncthreads();
    float dot = 0.f;
    for (int h = t; h < H; h += 256) {
        const float v = __bfloat162float(x[offset + h]);
        // Explicit BF16 rounding boundary: projection consumes BF16 norm output.
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            (v * __bfloat162float(w[h])) * inv);
        dot += __bfloat162float(normalized) * __bfloat162float(p[h]);
    }
    const float result = reduce256(dot, tmp);
    if (!t) scores[row] = __float2bfloat16_rn(result);
}
}
extern "C" int kimi_attnres_front_launch_r8982(
    const void *x, const void *w, const void *p, void *scores, float eps, void *stream) {
    front<<<16 * 24, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w),
        static_cast<const __nv_bfloat16 *>(p), static_cast<__nv_bfloat16 *>(scores), eps);
    return int(cudaGetLastError());
}
