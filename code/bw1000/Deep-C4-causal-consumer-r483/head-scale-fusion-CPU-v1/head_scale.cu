#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned threads = 128;
constexpr size_t token_limit = 512;

template <unsigned Items>
__global__ void merge(float* output, const uint16_t* head, const float* score_scale,
                      const float* query_scale, size_t elements) {
    const size_t first = size_t(blockIdx.x) * (threads * Items) + threadIdx.x;
    const float scalar = score_scale[0];
    #pragma unroll
    for (unsigned item = 0; item < Items; ++item) {
        const size_t i = first + item * threads;
        if (i < elements) {
            const float h = __uint_as_float(uint32_t(head[i]) << 16);
            const float scaled = __fmul_rn(h, scalar);
            output[i] = __fmul_rn(scaled, query_scale[i]);
        }
    }
}

template <unsigned Items>
int launch(void* output, const void* head, const void* score_scale, const void* query_scale,
           size_t tokens, size_t heads, void* stream) {
    if (!output || !head || !score_scale || !query_scale
        || output == head || output == score_scale || output == query_scale
        || tokens == 0 || tokens > token_limit || (heads != 8 && heads != 64)) {
        return 1;
    }
    const size_t elements = tokens * heads;
    const unsigned grid = unsigned((elements + threads * Items - 1) / (threads * Items));
    merge<Items><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(output), static_cast<const uint16_t*>(head),
        static_cast<const float*>(score_scale), static_cast<const float*>(query_scale), elements);
    return int(cudaGetLastError());
}

}

extern "C" int deep_c4_head_scale_i1_v1(void* output, const void* head, const void* score_scale,
                                        const void* query_scale, size_t tokens, size_t heads, void* stream) {
    return launch<1>(output, head, score_scale, query_scale, tokens, heads, stream);
}

extern "C" int deep_c4_head_scale_i4_v1(void* output, const void* head, const void* score_scale,
                                        const void* query_scale, size_t tokens, size_t heads, void* stream) {
    return launch<4>(output, head, score_scale, query_scale, tokens, heads, stream);
}
