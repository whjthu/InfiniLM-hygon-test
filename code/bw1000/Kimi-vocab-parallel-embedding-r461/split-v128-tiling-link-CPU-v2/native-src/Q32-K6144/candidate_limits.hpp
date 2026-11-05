#pragma once
namespace kimi_split_v128 {
constexpr unsigned long candidate_fa_query=32,candidate_fa_keys=6144;
static_assert(candidate_fa_query==32 && candidate_fa_keys>0 && candidate_fa_keys<=32768 && candidate_fa_keys%256==0,"bounded candidate tiling");
}
