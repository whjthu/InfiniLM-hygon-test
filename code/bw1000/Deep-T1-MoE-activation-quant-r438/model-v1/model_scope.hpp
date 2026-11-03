#pragma once
#include "dispatch.hpp"
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
// No Tensor/SDK/GPU/Runtime initialization in this policy header. The actual
// MoE owner invokes Scope after its original owner/input checks. Default is
// unarmed OFF. set(0/1) explicitly arms per-request experimental counters;
// caller must hold the original whole-engine quiescence between requests.
namespace deep438_model {
constexpr size_t owners=8,width=12;
enum Field:size_t{Entered,T1,Mode1T1,RCalls,RHits,RFallback,SCalls,SHits,SFallback,HostReturned,Unwind,LastMode};
struct alignas(64) Row{std::array<std::atomic<uint64_t>,width>value{};std::atomic<bool>active{false};};
inline std::array<Row,owners>rows{};
inline std::atomic<bool>armed{false};inline std::atomic<int>mode{0};inline std::mutex control;
inline bool active(){for(auto&r:rows)if(r.active.load())return true;return false;}
inline int set(int v){if(v!=0&&v!=1)return 1;std::lock_guard<std::mutex>lock(control);if(active())return 2;mode.store(v);armed.store(true);return 0;}
inline int clear(){std::lock_guard<std::mutex>lock(control);if(active())return 2;armed.store(false);mode.store(0);for(auto&r:rows)for(auto&v:r.value)v.store(0);return 0;}
inline int read(int owner,uint64_t*out,size_t n){if(owner<0||owner>=int(owners)||!out||n!=width)return 1;for(size_t k=0;k<n;++k)out[k]=rows[owner].value[k].load();return 0;}
class Scope{
 Row*row_=nullptr;bool returned_=false;
 public:
 Scope(int owner,bool t1){
  if(!armed.load()||owner<0||owner>=int(owners))return;
  if(deep438::state.active)throw std::logic_error("r438 Model nested policy scope");
  row_=&rows[size_t(owner)];if(row_->active.exchange(true)){row_=nullptr;throw std::logic_error("r438 experimental Model requires one Runtime owner per visible device");}
  deep438::state={};deep438::state.active=true;deep438::state.t1=t1;deep438::state.mode=mode.load();
  ++row_->value[Entered];if(t1)++row_->value[T1];if(t1&&deep438::state.mode==1)++row_->value[Mode1T1];row_->value[LastMode].store(uint64_t(deep438::state.mode));
 }
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 void returned(){returned_=true;}
 ~Scope()noexcept{
  if(!row_)return;
  auto s=deep438::state;deep438::state={};
  for(size_t role=0;role<2;++role){size_t first=role?SCalls:RCalls;row_->value[first].fetch_add(s.calls[role]);row_->value[first+1].fetch_add(s.hits[role]);row_->value[first+2].fetch_add(s.fallbacks[role]);}
  ++row_->value[returned_?HostReturned:Unwind];row_->active.store(false);
 }
};
}
