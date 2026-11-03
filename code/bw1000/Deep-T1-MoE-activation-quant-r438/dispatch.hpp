#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
namespace deep438 {
enum Role:int {Routed=0,Shared=1};
struct State {bool active=false,t1=false;int mode=0;void*diagnostic=nullptr;std::array<uint64_t,2>calls{},hits{},fallbacks{};};
inline thread_local State state{};
inline bool select(Role role,size_t rows,bool actual_eligible){
 ++state.calls.at(role);
 if(state.active&&state.t1&&state.mode==1&&actual_eligible&&rows==(role==Routed?6u:1u)){++state.hits.at(role);return true;}
 ++state.fallbacks.at(role);return false;
}
template<class F>State owner_phase(int mode,bool t1,void*diagnostic,F f){
 if(state.active||mode<0||mode>1)throw std::invalid_argument("r438 explicit nonnested owner0/1 phase");
 state={};state.active=true;state.mode=mode;state.t1=t1;state.diagnostic=diagnostic;
 try{f();auto out=state;state={};return out;}catch(...){state={};throw;}
}
inline void CPU_contract(){
 std::array<std::thread,8>threads;std::array<bool,8>ok{};
 for(size_t i=0;i<8;++i)threads[i]=std::thread([&,i]{try{
  for(int mode:{0,1,1,0}){auto s=owner_phase(mode,true,nullptr,[&]{if(select(Routed,6,true)!=bool(mode)||select(Shared,1,true)!=bool(mode))throw std::logic_error("dispatch");});
   for(int r=0;r<2;++r)if(s.calls[r]!=1||s.hits[r]!=uint64_t(mode)||s.fallbacks[r]!=uint64_t(!mode))throw std::logic_error("counts");}
  for(int n=0;n<4;++n){auto s=owner_phase(1,n!=0,nullptr,[&]{bool a=select(Routed,n==1?12:6,n!=2),b=select(Shared,n==1?2:1,n!=3);
    if(n<2&&(a||b))throw std::logic_error("T2 fallback");});if(n<2&&(s.hits[0]||s.hits[1]))throw std::logic_error("ineligible");}
  bool caught=false;try{owner_phase(1,true,nullptr,[&]{select(Routed,6,true);throw std::runtime_error("original failure");});}catch(...){caught=true;}
  if(!caught||state.active||state.mode||state.diagnostic)throw std::logic_error("restore");ok[i]=true;
 }catch(...){ok[i]=false;}});
 for(auto&t:threads)t.join();for(bool x:ok)if(!x)throw std::logic_error("r438 CPU owner contract failed");
}
}
