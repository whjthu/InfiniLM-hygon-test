#pragma once
#include "runtime/runtime.hpp"
#include <cstddef>
#include <cstdint>
namespace infinilm::models::glm_moe_dsa {
void glm_m15_oproj_policy_r125(runtime::Runtime&,bool);
}
extern "C" int modern_glm_m15_oproj_stats_r125(int device,uint64_t*out,size_t count);
