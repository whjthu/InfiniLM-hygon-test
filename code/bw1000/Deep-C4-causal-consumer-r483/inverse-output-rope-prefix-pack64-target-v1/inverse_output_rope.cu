#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned wave = 64;
constexpr unsigned width = 512;
constexpr unsigned tail_width = 64;
constexpr unsigned prefix_pairs = (width - tail_width) / 2;
constexpr unsigned tail_pairs = tail_width / 2;

__device__ __forceinline__ float from_bf16(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}

__device__ __forceinline__ uint16_t to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void copy_prefix_pair32(uint16_t* out, const uint16_t* in) {
    uint32_t pair;
    __builtin_memcpy(&pair, __builtin_assume_aligned(in, alignof(uint32_t)), sizeof(pair));
    __builtin_memcpy(__builtin_assume_aligned(out, alignof(uint32_t)), &pair, sizeof(pair));
}

__device__ __forceinline__ void copy_prefix_group64(uint16_t* out, const uint16_t* in) {
    uint64_t group;
    __builtin_memcpy(&group, __builtin_assume_aligned(in, alignof(uint64_t)), sizeof(group));
    __builtin_memcpy(__builtin_assume_aligned(out, alignof(uint64_t)), &group, sizeof(group));
}

template<unsigned HeadsPerBlock, unsigned PrefixBits>
__global__ void inverse_output(uint16_t* out, const uint16_t* in, const float* table,
                               size_t heads, size_t total_heads) {
    const size_t head = size_t(blockIdx.x) * HeadsPerBlock + threadIdx.x / wave;
    if (head >= total_heads) {
        return;
    }
    const unsigned lane = threadIdx.x % wave;
    out += head * width;
    in += head * width;
    table += (head / heads) * tail_width;

    static_assert(prefix_pairs % 2 == 0, "prefix must hold whole 64-bit groups");
    if constexpr (PrefixBits == 64) {
        for (unsigned group = lane; group < prefix_pairs / 2; group += wave) {
            copy_prefix_group64(out + 4 * group, in + 4 * group);
        }
    } else {
        for (unsigned pair = lane; pair < prefix_pairs; pair += wave) {
            if constexpr (PrefixBits == 32) {
                copy_prefix_pair32(out + 2 * pair, in + 2 * pair);
            } else {
                out[2 * pair] = in[2 * pair];
                out[2 * pair + 1] = in[2 * pair + 1];
            }
        }
    }
    if (lane < tail_pairs) {
        const unsigned x_index = width - tail_width + 2 * lane;
        const float x = from_bf16(in[x_index]);
        const float y = from_bf16(in[x_index + 1]);
        const float cosine = table[lane];
        const float sine = -table[tail_pairs + lane];
        out[x_index] = to_bf16(x * cosine - y * sine);
        out[x_index + 1] = to_bf16(y * cosine + x * sine);
    }
}

bool overlaps(uintptr_t a, size_t a_size, uintptr_t b, size_t b_size) {
    return a <= b ? b - a < a_size : a - b < b_size;
}

template<unsigned HeadsPerBlock, unsigned PrefixBits>
int launch(void* out, const void* in, const void* table, size_t tokens, size_t heads, void* stream) {
    if (!out || !in || !table || !stream || tokens == 0 || tokens > 512
        || (heads != 8 && heads != 64)) {
        return 1;
    }
    const uintptr_t y = reinterpret_cast<uintptr_t>(out);
    const uintptr_t x = reinterpret_cast<uintptr_t>(in);
    const uintptr_t t = reinterpret_cast<uintptr_t>(table);
    const size_t count = tokens * heads;
    const size_t data_bytes = count * width * sizeof(uint16_t);
    const size_t table_bytes = tokens * tail_width * sizeof(float);
    if (x % alignof(uint16_t) || y % alignof(uint16_t) || t % alignof(float)
        || overlaps(x, data_bytes, y, data_bytes) || overlaps(t, table_bytes, x, data_bytes)
        || overlaps(t, table_bytes, y, data_bytes)) {
        return 1;
    }
    if constexpr (PrefixBits == 64) {
        if ((x | y) % alignof(uint64_t)) {
            return launch<HeadsPerBlock, 32>(out, in, table, tokens, heads, stream);
        }
    }
    if constexpr (PrefixBits == 32) {
        if ((x | y) % alignof(uint32_t)) {
            return launch<HeadsPerBlock, 16>(out, in, table, tokens, heads, stream);
        }
    }
    const unsigned grid = static_cast<unsigned>((count + HeadsPerBlock - 1) / HeadsPerBlock);
    constexpr unsigned threads = wave * HeadsPerBlock;
    inverse_output<HeadsPerBlock, PrefixBits><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint16_t*>(out), static_cast<const uint16_t*>(in), static_cast<const float*>(table),
        heads, count);
    return int(cudaGetLastError());
}

}

extern "C" int deep_inverse_rope_h1_pack64_v1(void* out, const void* in, const void* table,
                                             size_t tokens, size_t heads, void* stream) {
    return launch<1, 64>(out, in, table, tokens, heads, stream);
}
