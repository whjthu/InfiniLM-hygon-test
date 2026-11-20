#pragma once
#include <cstdint>
#include <cstring>

#define __global__

using cudaStream_t = void *;
struct Position { unsigned x = 0; };
inline Position blockIdx, threadIdx;

inline unsigned last_grid = 0, last_threads = 0, launch_count = 0;
inline void *last_stream = nullptr;
inline int last_error = 0;

inline float __uint_as_float(uint32_t value) {
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}
inline float __fmul_rn(float a, float b) {
    volatile float rounded = a*b;
    return rounded;
}
inline int cudaGetLastError() { return last_error; }

template<typename Kernel, typename... Arguments>
void cpu_launch(Kernel kernel, unsigned grid, unsigned threads, unsigned, void* stream, Arguments... args) {
    last_grid = grid;
    last_threads = threads;
    last_stream = stream;
    ++launch_count;
    for (unsigned block = 0; block < grid; ++block) {
        blockIdx.x = block;
        for (unsigned thread = 0; thread < threads; ++thread) {
            threadIdx.x = thread;
            kernel(args...);
        }
    }
}
