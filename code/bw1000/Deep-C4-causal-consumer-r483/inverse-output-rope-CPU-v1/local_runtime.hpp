#pragma once
#include <cstdint>
#include <cstring>

#define __global__
#define __device__
#define __forceinline__ inline

using cudaStream_t = void*;
struct Position { unsigned x = 0; };
inline Position blockIdx, threadIdx;
inline unsigned launch_count = 0, last_grid = 0, last_threads = 0;
inline void* last_stream = nullptr;
inline int injected_error = 0;

inline float __uint_as_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t __float_as_uint(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline int cudaGetLastError() { return injected_error; }

template<typename Kernel, typename... Args>
void cpu_launch(Kernel fn, unsigned grid, unsigned threads, unsigned, void* stream, Args... args) {
    ++launch_count;
    last_grid = grid;
    last_threads = threads;
    last_stream = stream;
    for (unsigned block = 0; block < grid; ++block) {
        blockIdx.x = block;
        for (unsigned thread = 0; thread < threads; ++thread) {
            threadIdx.x = thread;
            fn(args...);
        }
    }
}
