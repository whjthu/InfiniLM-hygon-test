#pragma once
#include "runtime/runtime.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace deep_position_proof_r8726 {
namespace rt=infinilm::runtime;
struct Frame {
 rt::Runtime*owner; void*stream; int device; const void*tensor; const void*data;
 const std::vector<int64_t>*host; Frame*previous; bool enabled;
};
inline thread_local Frame*active=nullptr;
inline std::array<std::array<std::atomic<uint64_t>,3>,8>counts{};
inline bool policy(){
 const char*v=std::getenv("INFINILM_DEEP_POSITION_PROOF_R8726");
 if(!v||!std::strcmp(v,"0"))return false;
 if(!std::strcmp(v,"1"))return true;
 throw std::invalid_argument("8722 positions policy must be unset/0/1");
}
class Scope {
 Frame frame_;
 public:
 Scope(rt::Runtime&r,const rt::Tensor&p,const std::vector<int64_t>&host)
 :frame_{&r,r.stream(),r.device().index(),static_cast<const void*>(&p),p?p->data():nullptr,&host,active,policy()}{
  r.assertOwnerThread();
  if(r.isGraphRecording()||!p||p->device()!=r.device()||p->dtype()!=rt::DataType::kInt64||p->ndim()!=1||!p->size(0)||p->size(0)!=host.size()||!p->is_contiguous())throw std::invalid_argument("8722 requires original validated positions");
  active=&frame_;
 }
 ~Scope()noexcept{if(active!=&frame_)std::terminate();active=frame_.previous;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline bool copy(rt::Runtime&r,const rt::Tensor&p,std::vector<int64_t>&out){
 r.assertOwnerThread();const Frame*f=active;const int d=r.device().index();
 const bool match=f&&f->enabled&&f->owner==&r&&f->stream==r.stream()&&f->device==d&&!r.isGraphRecording()&&p&&static_cast<const void*>(&p)==f->tensor&&p->data()==f->data&&!(p->device()!=r.device())&&p->dtype()==rt::DataType::kInt64&&p->ndim()==1&&p->is_contiguous()&&p->size(0)==f->host->size()&&out.size()==f->host->size();
 if(!match){if(d>=0&&d<8)counts[d][2].fetch_add(1,std::memory_order_relaxed);return false;}
 out.assign(f->host->begin(),f->host->end());
 if(d>=0&&d<8){counts[d][0].fetch_add(1,std::memory_order_relaxed);counts[d][1].fetch_add(out.size()*sizeof(int64_t),std::memory_order_relaxed);}return true;
}
}
