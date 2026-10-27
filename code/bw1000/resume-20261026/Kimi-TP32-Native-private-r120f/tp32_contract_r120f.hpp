#pragma once
#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
namespace kimi_tp32_r120f {
constexpr size_t TP=32,PP=1,Heads=3,D=128,Projection=384,QKV=1152,Vocab=163840,HeadRows=5120,Hidden=7168;
constexpr uint64_t HeadBytes=HeadRows*Hidden*2,CacheBytes=1086011136ULL,FloorBytes=4ULL*1024*1024*1024;
inline void flag(const char*name,const char*value){const auto*p=std::getenv(name);if(!p||std::string(p)!=value)throw std::invalid_argument(std::string("r120f immutable flag ")+name);}
inline void enabled_environment(){
 flag("INFINILM_KIMI_TP32_R120F","1");flag("INFINILM_KIMI_SPEC_R3632","1");flag("INFINILM_KIMI_SPEC_MATH_R3676","0");
 for(const char*n:{"INFINILM_KIMI_MOE_WORKSPACE_R4665","INFINILM_KIMI_MOE_EXECUTION_R4665","INFINILM_KIMI_I192_GROUPED_R8669Y","INFINILM_KIMI_BATCH_I192_R9125","INFINILM_KIMI_I192_COMPACT_R8669ZC","INFINILM_KIMI_W2_PAIR_R9280","INFINILM_KIMI_PROFILE_R9125","INFINILM_MXFP4_TILE_TP_R2559","INFINILM_KIMI_KDA_SERIAL_R9027","INFINILM_KIMI_ROWAR_R9043","INFINILM_KIMI_CCL512_R8603","INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562","INFINILM_KIMI_KDA_INPUT_M1_R4305","INFINILM_KIMI_KDA_GATE_M1_R4571"})flag(n,"0");
 flag("INFINILM_OWNER_CPU_WIDTH_R2392","4");flag("KIMI_TP_NODES","4");
}
inline void topology(int node,int local,int tp,int rank,int pp,int stage,int world,int worldrank){if(node<0||node>=4||local<0||local>=8||tp!=32||rank!=node*8+local||pp!=1||stage||world!=32||worldrank!=rank)throw std::invalid_argument("r120f real32 owner topology");}
inline size_t head_begin(size_t rank){if(rank>=32)throw std::invalid_argument("r120f rank");return rank*HeadRows;}
}
