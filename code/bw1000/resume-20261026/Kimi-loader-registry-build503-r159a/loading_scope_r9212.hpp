#pragma once
#include "bindings_cache_r9130.hpp"
#include "loader_registry_clock_r159.hpp"
#include "weights/weights.hpp"
#include "runtime/runtime.hpp"
#include <atomic>
#include <cstdlib>
#include <limits>
#include <sstream>
namespace infinilm::loading_r9212 {
using Bindings=std::unordered_map<std::string,weights::Weight>;
inline std::atomic<uint64_t> generations{1};
inline std::atomic<uint64_t> counters[8][4]{};
class Scope;
inline thread_local Scope*current=nullptr;
inline Bindings timed_state_dict(weights::ModelWeights&m,runtime::Runtime&r){r.assertOwnerThread();loader_registry_r159::Clock clock(r.device().index(),loader_registry_r159::StateDictBuild);return m.state_dict();}
inline uint64_t next_generation(){auto n=generations.fetch_add(1);if(!n||n==std::numeric_limits<uint64_t>::max())std::terminate();return n;}
class Scope final {
 runtime::Runtime&r_;weights::ModelWeights&m_;uint64_t generation_;loading_r9130::ImmutableBindings<Bindings>cache_;bool enabled_,active_=true;
 void check(){r_.assertOwnerThread();if(current!=this||!active_)throw std::logic_error("9212 owner/session closed");}
 std::unordered_map<std::string,std::string>sealed_;
 static std::string fingerprint(const weights::Weight&w){
  std::ostringstream out;out<<w.empty()<<":"<<w.checkpoint_required();
  if(!w.empty()){
   out<<":"<<reinterpret_cast<uintptr_t>(w.tensor().operator->())<<":"<<reinterpret_cast<uintptr_t>(w->data())<<":"<<int(w->dtype())<<":"<<int(w->device().type())<<":"<<w->device().index()<<":shape";
   for(auto v:w->shape())out<<","<<v;out<<":strides";for(auto v:w->strides())out<<","<<v;
  }return out.str();
 }
 bool equal(const Bindings&a,const Bindings&b)const{
  if(a.size()!=b.size()||sealed_.size()!=b.size())return false;
  for(const auto&[k,w]:b){auto it=sealed_.find(k);if(it==sealed_.end()||it->second!=fingerprint(w))return false;}return true;
 }

 public:
 Scope(runtime::Runtime&r,weights::ModelWeights&m):r_(r),m_(m),generation_(next_generation()),cache_(&m,generation_),enabled_(std::getenv("INFINILM_LOADING_BINDINGS_R9212")&&std::string(std::getenv("INFINILM_LOADING_BINDINGS_R9212"))=="1"){
  r_.assertOwnerThread();if(current)throw std::logic_error("9212 nested loading session");current=this;
 }
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 ~Scope()noexcept{
  if(active_){try{if(enabled_&&cache_.holds_bindings())r_.syncStream();cache_.abort(&m_,generation_);active_=false;}catch(...){std::terminate();}}
  if(current!=this)std::terminate();current=nullptr;
 }
 Bindings*get(weights::ModelWeights&m,runtime::Runtime&r,size_t requested){
  if(!enabled_||!active_)return nullptr;check();if(&m!=&m_||&r!=&r_)throw std::logic_error("9212 model/runtime identity");
  loader_registry_r159::Clock clock(r_.device().index(),loader_registry_r159::CacheLookupInclusive);auto builds=cache_.builds;const auto&v=cache_.get(&m_,generation_,[&]{return timed_state_dict(m_,r_);});if(cache_.builds!=builds){for(const auto&[k,w]:v)sealed_.emplace(k,fingerprint(w));}
  auto device=r_.device().index();if(device<0||device>=8)throw std::logic_error("9212 localdevice");
  counters[device][0].fetch_add(cache_.builds-builds);counters[device][1].fetch_add(cache_.builds==builds);counters[device][2].store(v.size());counters[device][3].fetch_add(requested);return const_cast<Bindings*>(&v); // legacy Weight::load_into is nonconst but does not mutate binding metadata
 }
 void finish(){
  if(!active_)return;check();if(!enabled_){cache_.abort(&m_,generation_);active_=false;return;}loader_registry_r159::Clock clock(r_.device().index(),loader_registry_r159::CacheFinishInclusive);r_.syncStream();
  try{cache_.finish(&m_,generation_,[&]{return timed_state_dict(m_,r_);},[&](const Bindings&a,const Bindings&b){return equal(a,b);});sealed_.clear();active_=false;}
  catch(...){sealed_.clear();active_=false;throw;}
 }
};
inline Bindings*lookup(weights::ModelWeights&m,runtime::Runtime&r,size_t count){return current?current->get(m,r,count):nullptr;}
}
