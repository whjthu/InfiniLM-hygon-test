#pragma once
#include "runtime/runtime.hpp"
#include <stdexcept>
namespace deep_tail_prefill_r7211 {
namespace rt=infinilm::runtime;
struct Frame {rt::Runtime*owner;void*stream;int device;size_t requests,total;bool allpast0;Frame*previous;};
inline thread_local Frame*active=nullptr;
inline bool matches_metadata(size_t requests,size_t total,bool allpast0)noexcept{return allpast0&&requests==1&&total==512;}
class Scope {
 Frame frame_;
 public:
 template<class Requests>Scope(rt::Runtime&r,const Requests&rs):frame_{&r,r.stream(),r.device().index(),rs.size(),0,true,active}{
  r.assertOwnerThread();if(r.isGraphRecording()||rs.empty()||rs.size()>128)throw std::invalid_argument("7211 sidecar eager metadata");
  if(active&&(active->owner!=&r||active->stream!=r.stream()||active->device!=r.device().index()))throw std::logic_error("7211 nested owner differs");
  for(auto q:rs){if(!q.tokens||q.tokens>4096-frame_.total)throw std::invalid_argument("7211 sidecar total");frame_.total+=q.tokens;frame_.allpast0=frame_.allpast0&&q.past==0;}
  active=&frame_;
 }
 ~Scope(){if(active!=&frame_)std::terminate();active=frame_.previous;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline bool target(rt::Runtime&r,const rt::Tensor&v){
 r.assertOwnerThread();if(!active)return false;
 if(active->owner!=&r||active->stream!=r.stream()||active->device!=r.device().index())throw std::logic_error("7211 sidecar different Runtime/stream/device");
 return matches_metadata(active->requests,active->total,active->allpast0)&&v&&v->ndim()==2&&v->size(0)==512&&v->size(1)==4096&&v->dtype()==rt::DataType::kFloat32;
}
}
