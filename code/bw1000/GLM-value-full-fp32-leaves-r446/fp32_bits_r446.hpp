#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define R446_HD __host__ __device__
#else
#define R446_HD
#endif
namespace glm_value_r446 {
R446_HD inline bool finite_bits(uint32_t u){return (u&0x7f800000u)!=0x7f800000u;}
R446_HD inline uint32_t bf16_to_f32_bits(uint16_t u){return uint32_t(u)<<16;}
// Explicit IEEE RN-even, including signed zero/subnormals/overflow. A nonfinite
// input/output is flagged by the caller; NaNs use quiet canonical BF16 payload.
R446_HD inline uint16_t f32_to_bf16_bits(uint32_t u){
 if((u&0x7fffffffu)>0x7f800000u)return uint16_t((u>>16)|0x40u);
 return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
constexpr unsigned block=256;
constexpr size_t heads=8,aggregate_k=2048,aggregate_n=512,project_k=512,project_n=256;
constexpr uint64_t aggregation_FLOPs=2ull*heads*aggregate_k*aggregate_n;
constexpr uint64_t projection_FLOPs=2ull*heads*project_k*project_n;
}
#undef R446_HD
