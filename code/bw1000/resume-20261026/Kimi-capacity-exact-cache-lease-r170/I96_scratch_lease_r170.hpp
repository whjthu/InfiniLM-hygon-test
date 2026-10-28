#pragma once
// Capacity prototype only: one weak-registry workspace per live Runtime owner.
// Every actual I96 consumer holds Lease until a successful owner-stream drain.
// No TLS strong owner, raw-pointer registry key, class-layout or Core changes.
#include "runtime/runtime.hpp"
#include "contract_I96_r163.hpp"
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <exception>
namespace r170_scratch {
namespace rt=infinilm::runtime;
using WeakOwner=std::weak_ptr<rt::Runtime>;
struct Entry {
 WeakOwner owner;
 std::weak_ptr<rt::TensorImpl> tensor;
 uint64_t generation=0;
 uintptr_t stream=0;
 bool busy=false;
};
inline std::mutex registry_mutex;
inline std::map<WeakOwner,std::shared_ptr<Entry>,std::owner_less<WeakOwner>> registry;
inline uint64_t next_generation=1;
inline bool same_owner(const WeakOwner&a,const WeakOwner&b){std::owner_less<WeakOwner>less;return !less(a,b)&&!less(b,a);}
inline void require_owner(rt::Runtime&r){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.graphCaptureState()!=infinilm::execution::graph::GraphManager::CaptureState::kInactive)throw std::logic_error("r170 eager-only scratch lease");
}
inline void prune_locked(){
 for(auto i=registry.begin();i!=registry.end();){
  if((i->first.expired()||i->second->tensor.expired())&&!i->second->busy)i=registry.erase(i);else ++i;
 }
}
inline rt::Tensor shared_workspace(rt::Runtime&r){
 require_owner(r);auto owner=r.shared_from_this();WeakOwner key(owner);
 std::lock_guard<std::mutex>lock(registry_mutex);prune_locked();auto it=registry.find(key);
 if(it!=registry.end()){
  auto tensor=it->second->tensor.lock();
  if(!tensor||it->second->busy||it->second->stream!=reinterpret_cast<uintptr_t>(r.stream()))throw std::logic_error("r170 scratch constructor ownership/lifetime");
  rt::Tensor result;result.reset(std::move(tensor));
  if(!same_owner(WeakOwner(result->allocation_owner()),key)||result->nbytes()!=r163::bytes||result->dtype()!=rt::DataType::kBFloat16)throw std::logic_error("r170 shared scratch owner/geometry");
  return result;
 }
 if(next_generation==std::numeric_limits<uint64_t>::max())throw std::overflow_error("r170 generation exhaustion");
 auto tensor=rt::Tensor::empty(r,{r163::bytes/2},rt::DataType::kBFloat16);
 if(!same_owner(WeakOwner(tensor->allocation_owner()),key))throw std::logic_error("r170 scratch allocation owner");
 auto entry=std::make_shared<Entry>();entry->owner=key;entry->tensor=tensor->shared_from_this();entry->generation=next_generation++;entry->stream=reinterpret_cast<uintptr_t>(r.stream());registry.emplace(key,entry);
 return tensor;
}
class Lease final {
 std::shared_ptr<rt::Runtime> owner_;
 rt::Tensor tensor_;
 std::shared_ptr<Entry> entry_;
 uint64_t generation_=0;
public:
 Lease(rt::Runtime&r,const rt::Tensor&tensor):owner_(r.shared_from_this()),tensor_(tensor){
  require_owner(r);if(!tensor_||tensor_->allocation_owner().get()!=&r||tensor_->device()!=r.device()||tensor_->nbytes()!=r163::bytes||tensor_->dtype()!=rt::DataType::kBFloat16||!tensor_->is_contiguous())throw std::logic_error("r170 lease allocation geometry");
  std::lock_guard<std::mutex>lock(registry_mutex);auto it=registry.find(WeakOwner(owner_));
  if(it==registry.end()||!same_owner(it->second->owner,WeakOwner(owner_))||it->second->tensor.lock()!=tensor_->shared_from_this()||it->second->stream!=reinterpret_cast<uintptr_t>(r.stream())||it->second->busy)throw std::logic_error("r170 crossowner/crossstream/reentrant lease rejected");
  entry_=it->second;generation_=entry_->generation;entry_->busy=true;
 }
 Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;Lease(Lease&&)=delete;Lease&operator=(Lease&&)=delete;
 ~Lease()noexcept{
  if(!entry_)return;
  try{
   owner_->assertOwnerThread();if(entry_->stream!=reinterpret_cast<uintptr_t>(owner_->stream()))std::terminate();
   // Actual provider163 launches both memsets and all four kernels on this stream.
   // Holding owner+Tensor until this fence covers successful launch and unwind.
   owner_->syncStream();
   std::lock_guard<std::mutex>lock(registry_mutex);
   if(!entry_->busy||entry_->generation!=generation_||!same_owner(entry_->owner,WeakOwner(owner_)))std::terminate();
   entry_->busy=false;
  }catch(...){std::terminate();} // never publish reusable scratch after a failed drain
 }
};
} // namespace r170_scratch
