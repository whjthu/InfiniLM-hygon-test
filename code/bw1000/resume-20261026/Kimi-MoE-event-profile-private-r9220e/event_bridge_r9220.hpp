#pragma once
#include "event_tape_r9220.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
namespace moe_event_r9220 {
struct HipDriver{infinilm::runtime::Runtime&r;using Event=cudaEvent_t;
 void check(int x){if(x!=int(cudaSuccess))throw std::runtime_error("r9220 HIPevent error "+std::to_string(x));}
 void check_owner(){r.assertOwnerThread();if(r.isGraphRecording())throw std::runtime_error("r9220 eager owner only");int dev=-1;check(cudaGetDevice(&dev));if(dev!=r.device().index())throw std::runtime_error("r9220 current device mismatch");}
 Event create_event(){Event e;check(cudaEventCreate(&e));return e;}void destroy_event(Event e){check(cudaEventDestroy(e));}
 int32_t*pinned_i32(size_t n){void*p=nullptr;check(cudaHostAlloc(&p,n*4,cudaHostAllocDefault));return static_cast<int32_t*>(p);}void free_pinned(void*p){check(cudaFreeHost(p));}
 void record(Event e){check(cudaEventRecord(e,static_cast<cudaStream_t>(r.stream())));}bool query(Event e){auto c=cudaEventQuery(e);if(c==cudaErrorNotReady)return false;check(c);return true;}
 float elapsed(Event a,Event b){float ms=0;check(cudaEventElapsedTime(&ms,a,b));return ms;}void copy_ids_async(int32_t*dst,const int32_t*src,size_t n){check(cudaMemcpyAsync(dst,src,n*4,cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(r.stream())));}
};
struct OwnerTape{HipDriver driver;Tape<HipDriver>tape;explicit OwnerTape(infinilm::runtime::Runtime&r):driver{r},tape(driver){}};
inline thread_local std::unique_ptr<OwnerTape>current;
inline bool begin(infinilm::runtime::Runtime&r,int layer,size_t tokens){if(current && &current->driver.r!=&r)throw std::runtime_error("r9220 Runtimeidentity");return current?current->tape.begin(layer,tokens):false;}
inline void mark(bool active){if(active)current->tape.mark();}
inline void finish(bool active,const infinilm::runtime::Tensor&ids){if(active)current->tape.finish(reinterpret_cast<const int32_t*>(ids->data()),[ids]{});}
}
extern "C" int modern_kimi_moe_event_arm_r9220(void*raw,int layer,uint64_t epoch){try{if(!raw)return 1;auto&r=*static_cast<infinilm::runtime::Runtime*>(raw);r.assertOwnerThread();if(!moe_event_r9220::current)moe_event_r9220::current=std::make_unique<moe_event_r9220::OwnerTape>(r);moe_event_r9220::current->tape.arm(layer,epoch);return 0;}catch(...){return 1;}}
extern "C" const char*modern_kimi_moe_event_read_r9220(){thread_local std::string text;try{if(!moe_event_r9220::current)return nullptr;auto q=moe_event_r9220::current->tape.read_after_original_work_completion();text=nlohmann::json{{"epoch",q.epoch},{"layer",q.layer},{"T",q.tokens},{"route_count",q.routes},{"expert_hist",q.hist},{"event_segment_ms",q.segment_ms},{"hist_copy_event_ms",q.hist_copy_ms},{"scope","same-owner-stream events include host launch gaps and dependencies; NOT pure GPU busy; no extra synchronizations"}}.dump();return text.c_str();}catch(...){return nullptr;}}
extern "C" int modern_kimi_moe_event_close_r9220(){try{if(moe_event_r9220::current){moe_event_r9220::current->tape.close_after_original_work_completion();moe_event_r9220::current.reset();}return 0;}catch(...){return 1;}}
