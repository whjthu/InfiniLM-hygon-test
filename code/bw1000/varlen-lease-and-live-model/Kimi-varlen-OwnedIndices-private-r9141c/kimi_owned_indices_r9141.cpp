#include "kimi_owned_indices_r9141.hpp"
#include "kimi_owned_indices_raw_r9141.hpp"
#include "runtime/runtime.hpp"
#include <infini/ops.h>
#include <infini/rt/device_guard.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "varlen_metadata_r9141.hpp"
namespace infinilm::models::kimi_native {
namespace {
thread_local OwnedIndicesRequestR9141*active=nullptr;
thread_local std::unordered_map<rt::Runtime*,std::unique_ptr<varlen_metadata_r9141::Ledger>>epochs;
void check(int code){if(code)throw std::runtime_error(std::string("9141owned metadata: ")+modern_kimi_owned_error_r9141());}
std::vector<int64_t>cpu_values(const rt::Tensor&t){
 if(!t||t->device().type()!=rt::Device::Type::kCpu||t->dtype()!=rt::DataType::kInt32||t->ndim()!=1||!t->is_contiguous()||!t->numel())throw std::invalid_argument("9141requires frozen contiguous CPU int32 indices");
 const auto*p=reinterpret_cast<const int32_t*>(t->data());return std::vector<int64_t>(p,p+t->numel());
}
template<size_t N>std::vector<infini::ops::Tensor>views(rt::Runtime&r,const std::array<rt::Tensor,N>&input){
 std::vector<infini::ops::Tensor>out;out.reserve(N);
 for(size_t i=0;i<N;++i){const auto&t=input[i];if(!t||t->device()!=r.device())throw std::invalid_argument("9141operand differs from Runtime device");out.emplace_back(const_cast<std::byte*>(t->data()),t->shape(),t->dtype(),t->device(),t->strides());}
 return out;
}
void drain(rt::Runtime&r)noexcept{try{r.syncStream();}catch(...){std::terminate();}}
}
OwnedIndicesRequestR9141::OwnedIndicesRequestR9141(rt::Runtime&r,const rt::Tensor&offsets,const rt::Tensor&reads,const rt::Tensor&writes,const std::vector<rt::Tensor>&conv_pools,const std::vector<rt::Tensor>&ssm_pools,uint64_t epoch,const void*owner):runtime_(&r){
 r.assertOwnerThread();if(active||r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||r.device().index()<0||r.device().index()>=8)throw std::invalid_argument("9141eager Runtime/request ownership");
 auto off=cpu_values(offsets),ri=cpu_values(reads),wi=cpu_values(writes);const auto n=ri.size();if(!n||n>32||wi.size()!=n||off.size()!=n+1)throw std::invalid_argument("9141decode request count");
 if(!owner||!epoch)throw std::invalid_argument("9141 immutable owner/epoch required");
 
 size_t pool=0,bytes=0;
 for(const auto*group:{&conv_pools,&ssm_pools})for(const auto&t:*group)if(t){if(t->device()!=r.device()||t->ndim()<3||!t->size(0))throw std::invalid_argument("9141state pool owner/geometry");if(pool&&pool!=t->size(0))throw std::invalid_argument("9141state pool capacities differ");pool=t->size(0);bytes=std::max(bytes,t->nbytes());}
 if(!pool||!bytes)throw std::invalid_argument("9141state pools absent");
 for(const auto&t:ssm_pools)if(t)state_owners_.insert(t->data());
 (void)varlen_metadata_r9141::validate(n,size_t(off.back()),pool,off.data(),ri.data(),wi.data());
 auto&ledger=epochs[&r];if(!ledger)ledger=std::make_unique<varlen_metadata_r9141::Ledger>(owner);ledger->check(owner,epoch);
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 try{
  metadata_=r.allocateMemory((3*n+1)*sizeof(int64_t));workspace_=r.allocateMemory(bytes);
  raw_=modern_kimi_owned_begin_r9141(r.device().index(),r.stream(),metadata_->data(),metadata_->size(),workspace_->data(),workspace_->size(),n,size_t(off.back()),pool,off.data(),ri.data(),wi.data(),1);
  if(!raw_)throw std::runtime_error(modern_kimi_owned_error_r9141());ledger->commit(owner,epoch);active=this;
 }catch(...){drain(r);throw;}
}
OwnedIndicesRequestR9141::~OwnedIndicesRequestR9141()noexcept{if(raw_)try{finish();}catch(...){std::terminate();}}
bool OwnedIndicesRequestR9141::matches(rt::Runtime&r)const{r.assertOwnerThread();return runtime_==&r&&raw_&&active==this;}
void OwnedIndicesRequestR9141::finish(){runtime_->assertOwnerThread();if(!raw_||active!=this)throw std::logic_error("9141inactive request finish");infini::rt::DeviceGuard guard{runtime_->device().type(),runtime_->device().index(),true};check(modern_kimi_owned_end_r9141(raw_));raw_=nullptr;active=nullptr;}
void OwnedIndicesRequestR9141::kda(const std::array<rt::Tensor,9>&input,float scale,float lower,bool normalize){
 auto&r=*runtime_;r.assertOwnerThread();if(!matches(r)||r.isGraphRecording()||!input[1]||!state_owners_.count(input[1]->data()))throw std::logic_error("9141KDA requires active eager Runtime/statepool owner");infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 try{auto v=views(r,input);check(modern_kimi_owned_kda_raw_r9141(raw_,v.data(),scale,lower,normalize));}catch(...){drain(r);throw;}
}
void owned_indices_clear_r9141(rt::Runtime&r,const void*owner){r.assertOwnerThread();if(active)throw std::invalid_argument("9141 cannot clear active lease");auto it=epochs.find(&r);if(it!=epochs.end()){it->second->own(owner);epochs.erase(it);}}
bool owned_indices_active_r9141(rt::Runtime&r){r.assertOwnerThread();if(!active)return false;if(!active->matches(r))throw std::logic_error("9141different Runtime owns active request");return true;}
void owned_indices_kda_r9141(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,float scale,float lower,bool normalize){if(!owned_indices_active_r9141(r))throw std::logic_error("9141missing KDA request scope");active->kda({out,state,q,k,v,gate,beta,alog,bias},scale,lower,normalize);}

}
