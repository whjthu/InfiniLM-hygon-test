#include <cstddef>
#include <cstdint>

#if defined(GLM_HEADSHARE_HOST_CHECK)
#define GLM_DEVICE
#else
#include <cuda_runtime.h>
#define GLM_DEVICE __device__ __forceinline__
#endif

namespace {

constexpr std::size_t heads = 4;
constexpr std::size_t projected_width = 448;
constexpr std::size_t output_width = 256;
constexpr std::size_t key_prefix = 192;
constexpr std::size_t rotary_width = 64;
constexpr std::size_t threads = output_width / 2;
constexpr std::size_t tokens_per_call = 512;

template<unsigned HeadsPerBlock>
GLM_DEVICE void copy_lane(uint16_t* key, uint16_t* value, const uint16_t* projected,
                         const uint16_t* rotary, std::size_t block, unsigned lane) {
    static_assert(HeadsPerBlock == 1 || HeadsPerBlock == 2);
    const std::size_t row0 = block * HeadsPerBlock;
    const unsigned channel = 2 * lane;
    const bool prefix = channel < key_prefix;
    uint16_t rope0 = 0, rope1 = 0;
    if (!prefix) {
        const auto* tail = rotary + (row0 / heads) * rotary_width + (channel - key_prefix);
        rope0 = tail[0];
        rope1 = tail[1];
    }
    for (unsigned h = 0; h < HeadsPerBlock; ++h) {
        const std::size_t row = row0 + h;
        const auto* in = projected + row * projected_width;
        const std::size_t destination = row * output_width + channel;
        key[destination] = prefix ? in[channel] : rope0;
        key[destination + 1] = prefix ? in[channel + 1] : rope1;
        value[destination] = in[key_prefix + channel];
        value[destination + 1] = in[key_prefix + channel + 1];
    }
}

bool overlaps(uintptr_t first, std::size_t first_size, uintptr_t second, std::size_t second_size) {
    return first <= second ? second - first < first_size : first - second < second_size;
}

bool valid(void* key, void* value, const void* projected, const void* rotary,
           std::size_t tokens, std::size_t local_heads, void* stream) {
    if (!key || !value || !projected || !rotary || !stream
        || tokens != tokens_per_call || local_heads != heads) {
        return false;
    }
    const void* regions[] = {key, value, projected, rotary};
    const std::size_t sizes[] = {tokens * heads * output_width * sizeof(uint16_t),
                                 tokens * heads * output_width * sizeof(uint16_t),
                                 tokens * heads * projected_width * sizeof(uint16_t),
                                 tokens * rotary_width * sizeof(uint16_t)};
    for (unsigned i = 0; i < 4; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(regions[i]);
        if (address % alignof(uint16_t) || sizes[i] > UINTPTR_MAX - address) {
            return false;
        }
        for (unsigned j = 0; j < i; ++j) {
            if (overlaps(address, sizes[i], reinterpret_cast<uintptr_t>(regions[j]), sizes[j])) {
                return false;
            }
        }
    }
    return true;
}

#if !defined(GLM_HEADSHARE_HOST_CHECK)
template<unsigned HeadsPerBlock>
__global__ void pack(uint16_t* key, uint16_t* value, const uint16_t* projected, const uint16_t* rotary) {
    copy_lane<HeadsPerBlock>(key, value, projected, rotary, blockIdx.x, threadIdx.x);
}

template<unsigned HeadsPerBlock>
int launch(void* key, void* value, const void* projected, const void* rotary,
           std::size_t tokens, std::size_t local_heads, void* stream) {
    if (!valid(key, value, projected, rotary, tokens, local_heads, stream)) {
        return 1;
    }
    constexpr unsigned grid = tokens_per_call * heads / HeadsPerBlock;
    pack<HeadsPerBlock><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint16_t*>(key), static_cast<uint16_t*>(value),
        static_cast<const uint16_t*>(projected), static_cast<const uint16_t*>(rotary));
    return int(cudaGetLastError());
}
#endif

}

#if !defined(GLM_HEADSHARE_HOST_CHECK)
extern "C" int glm_r474_fresh_kv_headshare_h1(void* key, void* value, const void* projected,
                                              const void* rotary, std::size_t tokens, std::size_t heads, void* stream) {
    return launch<1>(key, value, projected, rotary, tokens, heads, stream);
}

extern "C" int glm_r474_fresh_kv_headshare_h2(void* key, void* value, const void* projected,
                                              const void* rotary, std::size_t tokens, std::size_t heads, void* stream) {
    return launch<2>(key, value, projected, rotary, tokens, heads, stream);
}
#endif
