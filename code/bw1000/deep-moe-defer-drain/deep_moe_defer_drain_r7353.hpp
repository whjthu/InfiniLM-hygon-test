#pragma once
#include "deep_collective_layout_r7040.hpp"
#include <cstdlib>
#include <cstring>
namespace infinilm::models::deepseek_v4::defer_drain_r7353 {
inline std::array<std::atomic<uint64_t>,8> deferred_calls{};
inline bool enabled(){
 const char*p=std::getenv("INFINILM_DEEP_MOE_DEFER_DRAIN_R7353");
 if(!p||std::strcmp(p,"0")==0)return false;
 if(std::strcmp(p,"1")==0)return true;
 throw std::invalid_argument("7353 flag must exactly 0 or1");
}
inline void reduce_moe(runtime::Runtime&r,const runtime::Tensor&v,infinicclComm_t comm){
 if(!enabled()){layout_r7040::reduce_moe(r,v,comm);return;}
 // Preserve existing owner/device/layout/tensor and accepted-precision checks.
 layout_r7040::validate(r,v,comm);
 auto*f=deep_layout_core_r7032::current();
 const bool geometry=f&&f->policy==0&&f->requests==32&&f->layout.lengths.size()==32
  &&std::all_of(f->layout.lengths.begin(),f->layout.lengths.end(),[](size_t n){return n==1;})
  &&v->shape()==runtime::Shape{32,4096}&&v->dtype()==runtime::DataType::kFloat32;
 if(!geometry||wide_r3443::selected(r)!=5){layout_r7040::reduce_moe(r,v,comm);return;}
 // kind1/mode5 has no widening/scratch. Same F32 in-place SUM, same Runtime
 // stream and communicator. Only the immediate post-SUM host drain is deferred.
 // The original following Cast/HC and outer decoder/CAPI drains remain intact.
 engine::distributed::allreduce_(r,v,v,infinicclSum,comm);
 const auto d=r.device().index();
 ++layout_r7040::counts.at(d).fallback_calls;
 ++wide_r3443::counts.at(d).v[0];
 ++deferred_calls.at(d);
}
}
extern "C" uint64_t deep_moe_defer_calls_r7353(int d){
 if(d<0||d>=8)return UINT64_MAX;
 return infinilm::models::deepseek_v4::defer_drain_r7353::deferred_calls.at(d).load();
}
