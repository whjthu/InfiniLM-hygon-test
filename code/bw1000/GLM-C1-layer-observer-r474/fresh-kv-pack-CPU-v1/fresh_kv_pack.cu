#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t heads = 4;
constexpr size_t projected_width = 448;
constexpr size_t output_width = 256;
constexpr size_t key_prefix = 192;
constexpr size_t rotary_width = 64;
constexpr unsigned pairs = output_width / 2;

template<unsigned PairsPerBlock>
__global__ void pack(uint16_t* key, uint16_t* value, const uint16_t* projected,
                     const uint16_t* rotary) {
    constexpr unsigned tiles = pairs / PairsPerBlock;
    const size_t head_row = blockIdx.x / tiles;
    const unsigned channel = 2 * ((blockIdx.x % tiles) * PairsPerBlock + threadIdx.x);
    const uint16_t* in = projected + head_row * projected_width;
    const uint16_t* shared_rotary = rotary + (head_row / heads) * rotary_width;
    const uint16_t* key_in = channel < key_prefix ? in + channel : shared_rotary + (channel - key_prefix);
    const size_t destination = head_row * output_width + channel;
    key[destination] = key_in[0];
    key[destination + 1] = key_in[1];
    value[destination] = in[key_prefix + channel];
    value[destination + 1] = in[key_prefix + channel + 1];
}

bool overlaps(uintptr_t first, size_t first_size, uintptr_t second, size_t second_size) {
    return first <= second ? second - first < first_size : first - second < second_size;
}

template<unsigned PairsPerBlock>
int launch(void* key, void* value, const void* projected, const void* rotary,
           size_t tokens, size_t local_heads, void* stream) {
    if (!key || !value || !projected || !rotary || !stream || !tokens
        || tokens > 512 || local_heads != heads) {
        return 1;
    }
    const void* regions[] = {key, value, projected, rotary};
    const size_t sizes[] = {tokens * heads * output_width * sizeof(uint16_t),
                            tokens * heads * output_width * sizeof(uint16_t),
                            tokens * heads * projected_width * sizeof(uint16_t),
                            tokens * rotary_width * sizeof(uint16_t)};
    for (unsigned i = 0; i < 4; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(regions[i]);
        if (address % alignof(uint16_t)) {
            return 1;
        }
        for (unsigned j = 0; j < i; ++j) {
            if (overlaps(address, sizes[i], reinterpret_cast<uintptr_t>(regions[j]), sizes[j])) {
                return 1;
            }
        }
    }
    constexpr unsigned tiles = pairs / PairsPerBlock;
    const unsigned grid = static_cast<unsigned>(tokens * heads * tiles);
    pack<PairsPerBlock><<<grid, PairsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint16_t*>(key), static_cast<uint16_t*>(value),
        static_cast<const uint16_t*>(projected), static_cast<const uint16_t*>(rotary));
    return int(cudaGetLastError());
}

}

extern "C" int glm_r474_fresh_kv_pack_t64(void* key, void* value, const void* projected,
                                         const void* rotary, size_t tokens, size_t heads, void* stream) {
    return launch<64>(key, value, projected, rotary, tokens, heads, stream);
}

extern "C" int glm_r474_fresh_kv_pack_t128(void* key, void* value, const void* projected,
                                          const void* rotary, size_t tokens, size_t heads, void* stream) {
    return launch<128>(key, value, projected, rotary, tokens, heads, stream);
}
