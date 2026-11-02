#pragma once
#include "runtime/runtime.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace deep_position_proof_r387 {
namespace rt=infinilm::runtime;
// Controls are changed only between fully drained Model forwards; clear is quiescent-only.
inline std::atomic<int> mode{0};
// scopes_enabled, borrowed_copies, borrowed_bytes, enabled_fallbacks, disabled_copies.
inline std::array<std::array<std::atomic<uint64_t>,5>,8> counts{};
struct Frame {
 rt::Runtime* owner; void* stream; int device; const void* wrapper; const void* data;
 const std::vector<int64_t>* host; size_t tokens; Frame* previous; bool enabled;
};
inline thread_local Frame* active=nullptr;
class Scope {
 Frame frame_;
public:
 Scope(rt::Runtime&r,const rt::Tensor&p,const std::vector<int64_t>&host)
 :frame_{&r,r.stream(),r.device().index(),static_cast<const void*>(&p),p?p->data():nullptr,
         &host,host.size(),active,mode.load(std::memory_order_relaxed)==1} {
  r.assertOwnerThread();
  // Caller has already performed all original position/row/past validation.
  // This additional certificate never changes validation or failure semantics.
  frame_.enabled=frame_.enabled&&!r.isGraphRecording()&&p&&!(p->device()!=r.device())&&
   p->dtype()==rt::DataType::kInt64&&p->ndim()==1&&p->is_contiguous()&&
   p->size(0)>0&&p->size(0)==host.size();
  if(frame_.enabled&&frame_.device>=0&&frame_.device<8)
   counts[frame_.device][0].fetch_add(1,std::memory_order_relaxed);
  active=&frame_;
 }
 ~Scope()noexcept{if(active!=&frame_)std::terminate();active=frame_.previous;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline bool copy(rt::Runtime&r,const rt::Tensor&p,std::vector<int64_t>&out){
 r.assertOwnerThread();const Frame*f=active;const int d=r.device().index();
 const bool match=f&&f->enabled&&f->owner==&r&&f->stream==r.stream()&&f->device==d&&
  !r.isGraphRecording()&&p&&static_cast<const void*>(&p)==f->wrapper&&p->data()==f->data&&
  !(p->device()!=r.device())&&p->dtype()==rt::DataType::kInt64&&p->ndim()==1&&
  p->is_contiguous()&&p->size(0)==f->tokens&&f->host->size()==f->tokens&&out.size()==f->tokens;
 if(!match){if(d>=0&&d<8)counts[d][f&&f->enabled?3:4].fetch_add(1,std::memory_order_relaxed);return false;}
 out.assign(f->host->begin(),f->host->end());
 if(d>=0&&d<8){counts[d][1].fetch_add(1,std::memory_order_relaxed);counts[d][2].fetch_add(out.size()*sizeof(int64_t),std::memory_order_relaxed);}
 return true;
}
}
