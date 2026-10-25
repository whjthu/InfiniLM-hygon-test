#pragma once
#include "runtime/runtime.hpp"
#include <string>
#include <array>
#include <vector>
#include <stdexcept>
namespace kimi_delta_profile_r9140 {
namespace rt=infinilm::runtime;
void mode(rt::Runtime&,bool);void clear(rt::Runtime&);void finalize(rt::Runtime&);bool enabled(rt::Runtime&);const char*last(rt::Runtime&);void outer(rt::Runtime&,const std::array<double,7>&);
}
#ifdef KIMI_DELTA_PROFILE_IMPL_R9140
#include <cuda_runtime_api.h>
#include <chrono>
#include <unordered_map>
#include <array>
#include <nlohmann/json.hpp>
namespace kimi_delta_profile_r9140 {
using Clock=std::chrono::steady_clock;
struct Saved{bool on=false,pending=false;size_t used=0;std::array<cudaEvent_t,24>events{};std::vector<std::string>names;std::vector<double>cpu;Clock::time_point start;size_t C=0,T=0,pool=0;uint64_t serial=0;std::string result="{}";};
static thread_local std::unordered_map<rt::Runtime*,Saved>saved;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void mode(rt::Runtime&r,bool on){r.assertOwnerThread();auto&s=saved[&r];if(s.pending)throw std::invalid_argument("9140 pending profile");s.on=on;if(on&&!s.events[0])for(auto&e:s.events)ck(cudaEventCreate(&e));}
bool enabled(rt::Runtime&r){auto it=saved.find(&r);return it!=saved.end()&&it->second.on;}
void clear(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);if(it==saved.end())return;r.syncStream();for(auto&e:it->second.events)if(e)ck(cudaEventDestroy(e));saved.erase(it);}
struct Session{rt::Runtime&r;Saved*s=nullptr;Session(rt::Runtime&runtime,size_t T,size_t C,size_t pool):r(runtime){if(T!=16||!enabled(r))return;s=&saved.at(&r);if(s->pending)throw std::logic_error("9140 previous profile not finalized");s->used=0;s->names.clear();s->cpu.clear();s->start=Clock::now();s->C=C;s->T=T;s->pool=pool;++s->serial;mark("start");}
 void mark(const char*name){if(!s)return;if(s->used>=s->events.size())throw std::logic_error("9140 bounded markers");s->names.emplace_back(name);s->cpu.push_back(std::chrono::duration<double,std::milli>(Clock::now()-s->start).count());ck(cudaEventRecord(s->events[s->used++],static_cast<cudaStream_t>(r.stream())));}
 void finish(){if(s){mark("forward_end");s->pending=true;}}
};
void finalize(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);if(it==saved.end()||!it->second.pending)return;auto&s=it->second;nlohmann::json j={{"serial",s.serial},{"C",s.C},{"T",s.T},{"pool",s.pool},{"scope","same compute stream event intervals include host gaps/TP waiting, not GPU busy counters"},{"added_segment_syncs",0}};float full=0;ck(cudaEventElapsedTime(&full,s.events[0],s.events[s.used-1]));j["stream_total_ms"]=full;j["host_enqueue_total_ms"]=s.cpu.back();for(size_t i=1;i<s.used;++i){float ms=0;ck(cudaEventElapsedTime(&ms,s.events[i-1],s.events[i]));j["segments"].push_back({{"name",s.names[i]},{"stream_elapsed_ms",ms},{"host_boundary_ms",s.cpu[i]-s.cpu[i-1]}});}s.result=j.dump();s.pending=false;}
const char*last(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);if(it==saved.end())return "{}";return it->second.result.c_str();}
void outer(rt::Runtime&r,const std::array<double,7>&a){if(!enabled(r))return;auto&s=saved.at(&r);auto j=nlohmann::json::parse(s.result);j["bridge_boundary_ms"]=a;j["bridge_segments"]={"entry0","clear_preflight_capture_begin","input_H2D","metadata_H2D","forward_enqueue","existing_final_drain","output_D2H_and_finaldrain"};s.result=j.dump();}
}
#endif
