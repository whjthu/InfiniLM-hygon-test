#pragma once
#include <atomic>
#include <array>
#include <cstdint>
#include <cstddef>
#include <exception>
namespace hc_full_r377 {
inline std::atomic<int> flag{0};
// All fields count only calls whose entry captured flag==1. Original r366
// counters remain authoritative for all calls, including this flag's OFF arm.
struct alignas(64) Counters {
 std::atomic<uint64_t> observed{0},eligible{0},selected{0},completed{0},prefill{0},f32_t1{0},layout{0},selected_exceptions{0},fallback_exceptions{0},active{0};
};
inline std::array<Counters,8> counters{};
struct Scope {
 int device,exceptions;bool tracked,chosen,done=false;
 Scope(int d,size_t tokens,bool bf16,bool dense=true)noexcept:device(d),exceptions(0),tracked(d>=0&&d<8&&flag.load(std::memory_order_acquire)==1),chosen(tracked&&tokens==1&&bf16&&dense){
  if(!tracked)return;exceptions=std::uncaught_exceptions();auto&c=counters[d];c.active.fetch_add(1);c.observed.fetch_add(1);
  if(tokens==1&&bf16)c.eligible.fetch_add(1);if(chosen)c.selected.fetch_add(1);else if(tokens!=1)c.prefill.fetch_add(1);else if(!bf16)c.f32_t1.fetch_add(1);else c.layout.fetch_add(1);
 }
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 bool selected()const noexcept{return chosen;}
 void complete()noexcept{if(chosen&&!done){counters[device].completed.fetch_add(1);done=true;}}
 ~Scope()noexcept{
  if(!tracked)return;auto&c=counters[device];if(std::uncaught_exceptions()>exceptions)(chosen?c.selected_exceptions:c.fallback_exceptions).fetch_add(1);c.active.fetch_sub(1);
 }
};
}
// Enable/disable and clear must be called at the existing all-owner drained
// request barrier. Captured calls finish their chosen path; no mid-call cancel.
extern "C" int deep_hc_fullpre_enable_r377(int value){if(value!=0&&value!=1)return 1;hc_full_r377::flag.store(value,std::memory_order_release);return 0;}
extern "C" int deep_hc_fullpre_enabled_r377(){return hc_full_r377::flag.load(std::memory_order_acquire);}
extern "C" int deep_hc_fullpre_clear_r377(){
 if(deep_hc_fullpre_enabled_r377()!=0)return 1;
 for(auto&c:hc_full_r377::counters)if(c.active.load()!=0)return 1;
 for(auto&c:hc_full_r377::counters){c.observed=0;c.eligible=0;c.selected=0;c.completed=0;c.prefill=0;c.f32_t1=0;c.layout=0;c.selected_exceptions=0;c.fallback_exceptions=0;c.active=0;}return 0;
}
extern "C" int deep_hc_fullpre_stats_r377(uint64_t*out,int entries){
 if(!out||entries!=80)return 1;
 for(int d=0;d<8;++d){auto&c=hc_full_r377::counters[d];uint64_t v[]={c.observed.load(),c.eligible.load(),c.selected.load(),c.completed.load(),c.prefill.load(),c.f32_t1.load(),c.layout.load(),c.selected_exceptions.load(),c.fallback_exceptions.load(),c.active.load()};for(int k=0;k<10;++k)out[d*10+k]=v[k];}return 0;
}
