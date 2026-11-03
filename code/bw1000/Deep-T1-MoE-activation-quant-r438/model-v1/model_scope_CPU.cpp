#include "model_scope.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <array>
void demand(bool value){if(!value)throw std::logic_error("r438 Model policy CPU contract");}
void run(bool t1,bool fail){
 std::array<bool,8>ok{};std::vector<std::thread>threads;
 for(int owner=0;owner<8;++owner)threads.emplace_back([&,owner]{try{
  try{deep438_model::Scope scope(owner,t1);deep438::select(deep438::Routed,t1?6:12,true);
   if(fail)throw std::runtime_error("original math error");deep438::select(deep438::Shared,t1?1:2,true);scope.returned();
  }catch(const std::runtime_error&e){if(!fail)throw;demand(std::string(e.what())=="original math error");}
  demand(!deep438::state.active&&!deep438::state.diagnostic);ok[owner]=true;
 }catch(...){ok[owner]=false;}});
 for(auto&t:threads)t.join();for(bool b:ok)demand(b);
}
int main(){try{
 using namespace deep438_model;
 demand(!armed.load()&&mode.load()==0&&!active());
 {Scope scope(0,true);demand(!deep438::state.active);scope.returned();}demand(rows[0].value[Entered]==0);
 for(int v:{-1,2,99})demand(set(v)==1&&!armed.load());
 for(int v:{0,1}){demand(clear()==0&&set(v)==0);run(true,false);
  for(auto&r:rows){demand(r.value[Entered]==1&&r.value[T1]==1&&r.value[Mode1T1]==uint64_t(v));demand(r.value[RCalls]==1&&r.value[RHits]==uint64_t(v)&&r.value[RFallback]==uint64_t(!v));demand(r.value[SCalls]==1&&r.value[SHits]==uint64_t(v)&&r.value[SFallback]==uint64_t(!v));demand(r.value[HostReturned]==1&&r.value[Unwind]==0);}}
 demand(clear()==0&&set(1)==0);run(false,false);for(auto&r:rows)demand(r.value[T1]==0&&r.value[RHits]==0&&r.value[SHits]==0);
 demand(clear()==0&&set(1)==0);run(true,true);for(auto&r:rows)demand(r.value[Unwind]==1&&r.value[HostReturned]==0&&r.value[SCalls]==0);
 demand(clear()==0&&set(0)==0);{Scope scope(0,true);demand(set(1)==2&&clear()==2);scope.returned();}demand(clear()==0&&!armed.load()&&mode.load()==0&&!active());
 uint64_t data[width]{};demand(read(0,data,width)==0&&read(-1,data,width)==1&&read(8,data,width)==1&&read(0,nullptr,width)==1&&read(0,data,width-1)==1);
 std::cout<<"{\"passed\":true,\"actual_Model_scope_header\":true,\"default_unarmed_OFF\":true,\"owner_threads\":8,\"modes_0_1\":true,\"T2_fallback\":true,\"original_failure_unwinds\":true,\"active_MoE_change_rejected\":true,\"whole_engine_quiescence_is_caller_obligation\":true,\"Native_dlopen\":false,\"SDK_or_Runtime_executed\":false,\"GPU\":false}"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
