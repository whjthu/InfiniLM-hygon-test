#pragma once
#include "glm_rows32_policy_r5731.hpp"
// The exact resident caller uses owned BF16 K/V [32,640,8,256].
// This private definition is consumed by the actual bridge and its CPU check.
namespace glm_resident_exact_r8854 {
constexpr uint64_t rows32_bank_bytes=2ull*32*640*8*256*2;
inline uint64_t admission_bytes(size_t rows,size_t selected){
 if(rows==32){if(selected>40)throw std::invalid_argument("8854 exact rows32 layer count");return selected*rows32_bank_bytes;}
 return glm_rows32_r5731::admission_bytes(rows,selected);
}
}
