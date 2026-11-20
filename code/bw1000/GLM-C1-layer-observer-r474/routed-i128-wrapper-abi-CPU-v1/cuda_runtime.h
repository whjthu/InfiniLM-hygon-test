#pragma once
#include <array>
#include <cstdint>
#include <type_traits>

#define __device__
#define __global__

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x=1, unsigned y=1, unsigned z=1) : x(x), y(y), z(z) {}
};

using cudaStream_t = void *;
inline dim3 blockIdx;
inline dim3 threadIdx;
inline dim3 blockDim;

struct Launch {
  dim3 grid;
  dim3 block;
  unsigned shared;
  cudaStream_t stream;
  unsigned argc;
  std::array<std::uintptr_t, 11> argv;
};
inline Launch observed{};
inline unsigned launches = 0;
inline unsigned error_calls = 0;
inline int injected_status = 0;

template<typename T> std::uintptr_t word(T value) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<std::uintptr_t>(value);
  } else {
    return static_cast<std::uintptr_t>(value);
  }
}

template<typename Kernel, typename... Args>
void record_launch(Kernel kernel, dim3 grid, dim3 block, unsigned shared, cudaStream_t stream, Args... args) {
  static_assert(std::is_invocable_v<Kernel, Args...>);
  static_assert(sizeof...(Args) <= 11);
  observed = {grid, block, shared, stream, sizeof...(Args), {word(args)...}};
  ++launches;
  // The four-arg route-validation kernel has no wave operations.
  if constexpr (sizeof...(Args) == 4) {
    blockDim = block;
    for (unsigned b=0; b<grid.x; ++b) {
      blockIdx = dim3(b);
      for (unsigned thread=0; thread<block.x; ++thread) {
        threadIdx = dim3(thread);
        kernel(args...);
      }
    }
  } else {
    static_cast<void>(kernel);
  }
}

inline int cudaGetLastError() { ++error_calls; return injected_status; }
inline void atomicExch(int *status, int value) { *status = value; }
inline float __fadd_rn(float a, float b) { return a+b; }
inline float __shfl_down(float v, int, int) { return v; }
inline float __shfl_down_sync(unsigned, float v, int, int) { return v; }
