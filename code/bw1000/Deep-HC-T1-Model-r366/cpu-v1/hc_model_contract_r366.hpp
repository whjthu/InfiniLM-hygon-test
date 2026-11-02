#pragma once
#include <atomic>
#include <array>
#include <cstdint>
#include <cstddef>
namespace hc_model_r366 {
struct alignas(64) Counters {
    std::atomic<uint64_t> created{0},ready{0},pre{0},eligible_bf16_t1{0},completed{0},other_t{0},f32_t1{0};
};
inline std::array<Counters,8> counters{};
inline std::atomic<int> mode{0};
inline bool valid(int d){return d>=0&&d<8;}
inline bool enabled(){return mode.load(std::memory_order_relaxed)==1;}
inline void on_ctor(int d){if(valid(d))counters[d].created.fetch_add(1,std::memory_order_relaxed);}
inline void on_ready(int d){if(valid(d))counters[d].ready.fetch_add(1,std::memory_order_relaxed);}
inline void on_pre(int d,size_t t,bool bf16){if(valid(d)){auto&c=counters[d];c.pre.fetch_add(1,std::memory_order_relaxed);(t==1?(bf16?c.eligible_bf16_t1:c.f32_t1):c.other_t).fetch_add(1,std::memory_order_relaxed);}}
inline void on_completed(int d){if(valid(d))counters[d].completed.fetch_add(1,std::memory_order_relaxed);}
}
extern "C" int deep_hc_model_epilogue_set_r366(int value){if(value!=0&&value!=1)return 1;hc_model_r366::mode.store(value,std::memory_order_relaxed);return 0;}
extern "C" int deep_hc_model_epilogue_get_r366(){return hc_model_r366::mode.load(std::memory_order_relaxed);}
extern "C" int deep_hc_model_epilogue_counters_r366(uint64_t*out,int entries){
    if(!out||entries!=56)return 1;
    for(int d=0;d<8;++d){auto&c=hc_model_r366::counters[d];uint64_t v[]={c.created.load(),c.ready.load(),c.pre.load(),c.eligible_bf16_t1.load(),c.completed.load(),c.other_t.load(),c.f32_t1.load()};for(int k=0;k<7;++k)out[d*7+k]=v[k];}
    return 0;
}
