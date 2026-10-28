#pragma once
#include "phase_tape_r155.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
namespace moe_phase_r155 {
struct Driver {infinilm::runtime::Runtime&r;using Event=cudaEvent_t;
 void check(int status){if(status!=int(cudaSuccess))throw std::runtime_error("r155 event status "+std::to_string(status));}
 void check_owner(){r.assertOwnerThread();int current=-1;check(cudaGetDevice(&current));if(r.isGraphRecording()||current!=r.device().index())throw std::runtime_error("r155 Runtime owner/device mismatch");}
 Event create_event(){Event e;check(cudaEventCreate(&e));return e;}void destroy_event(Event e){check(cudaEventDestroy(e));}
 int32_t*pinned_i32(size_t n){void*p=nullptr;check(cudaHostAlloc(&p,n*4,cudaHostAllocDefault));return static_cast<int32_t*>(p);}void free_pinned(void*p){check(cudaFreeHost(p));}
 void record(Event e){check(cudaEventRecord(e,static_cast<cudaStream_t>(r.stream())));}bool query(Event e){auto q=cudaEventQuery(e);if(q==cudaErrorNotReady)return false;check(q);return true;}
 float elapsed(Event a,Event b){float ms;check(cudaEventElapsedTime(&ms,a,b));return ms;}
 void copy_ids_async(int32_t*dst,const int32_t*src,size_t n){check(cudaMemcpyAsync(dst,src,n*4,cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(r.stream())));}
};
struct Owner {Driver driver;Tape<Driver>tape;explicit Owner(infinilm::runtime::Runtime&r):driver{r},tape(driver){}};
inline thread_local std::unique_ptr<Owner>current;
inline bool begin(infinilm::runtime::Runtime&r,int layer,size_t tokens){if(current&& &current->driver.r!=&r)throw std::runtime_error("r155 Runtime identity changed");return current?current->tape.begin(layer,tokens):false;}
inline void mark(bool active,size_t n){if(active)current->tape.mark(n);}
inline void finish(bool active,const infinilm::runtime::Tensor&ids){if(active)current->tape.finish(reinterpret_cast<const int32_t*>(ids->data()),[ids]{});}
}
extern "C" int modern_kimi_moe_phase_mark_r155(unsigned index,void*stream){try{auto&v=moe_phase_r155::current;if(v&&v->tape.active()){if(stream!=v->driver.r.stream())return 1;v->tape.mark(index);}return 0;}catch(...){return 1;}}
extern "C" int modern_kimi_moe_event_arm_r9220(void*raw,int layer,uint64_t epoch){try{if(!raw)return 1;auto&r=*static_cast<infinilm::runtime::Runtime*>(raw);r.assertOwnerThread();if(!moe_phase_r155::current)moe_phase_r155::current=std::make_unique<moe_phase_r155::Owner>(r);moe_phase_r155::current->tape.arm(layer,epoch);return 0;}catch(...){return 1;}}
extern "C" const char*modern_kimi_moe_event_read_r9220(){thread_local std::string text;try{auto q=moe_phase_r155::current->tape.read_after_original_completion();text=nlohmann::json{{"layer",q.layer},{"epoch",q.epoch},{"T",q.tokens},{"expert_hist",q.hist},{"event_segment_ms",q.ms},{"phase_names",{"router","down_projection","provider_host_gap","clear_counts_jobs","pack_routes","build_jobs","fused_W13_dequant_Situ","fused_W2_weighted_scatter","provider_return_gap","routed_TP_collective","routed_norm","up_projection","shared_including_TP_reduce","add","route_hist_copy"}},{"scope","same owner original stream intervals include launch gaps/dependencies, not pure GPU busy; fused kernels not falsely decomposed"}}.dump();return text.c_str();}catch(...){return nullptr;}}
extern "C" int modern_kimi_moe_event_close_r9220(){try{if(moe_phase_r155::current){moe_phase_r155::current->tape.close_after_original_completion();moe_phase_r155::current.reset();}return 0;}catch(...){return 1;}}
