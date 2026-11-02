#include "scale_exact_scope_r201.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
template<class F>void reject(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}assert(bad);}
int main(){
 using namespace kimi_scale_exact_r201;
 int owner=0,object=0,other=0;
 Geometry old{true,16,2,896,3584,192,true,true};
 Geometry candidate{true,8,4,896,3584,384,true,true,true};
 for(auto g:{old,candidate}){
  {Scope scope(g,&owner,&object);assert(selected(&owner,&object));
   reject([&]{selected(&owner,&other);});reject([&]{Scope nested(g,&owner,&object);});}
  assert(!active);try{Scope scope(g,&owner,&object);throw 1;}catch(int){}assert(!active);
 }
 for(int i=0;i<9;++i){auto g=candidate;
  if(i==0)g.ordinary_tp8_pp4=false;if(i==1)g.tp=16;if(i==2)g.pp=2;
  if(i==3)g.experts=895;if(i==4)g.hidden=7168;if(i==5)g.local_intermediate=192;
  if(i==6)g.hygon=false;if(i==7)g.eager=false;if(i==8)g.tp=0;
  reject([&]{Scope scope(g,&owner,&object);});assert(!active);
 }
 auto regular=candidate;regular.long_factory=false;
 {Scope scope(regular,&owner,&object);assert(!selected(&owner,&object));}
 for(bool opt:{false,true}){
  auto tp=opt?8u:16u,pp=opt?4u:2u;kimi_tp8_pp4_r410::cache_scope(opt,tp,pp);
  reject([&]{kimi_tp8_pp4_r410::cache_scope(!opt,tp,pp);});
 }
 // All 32768 prompt positions appended in contiguous blocks, exact decode tail.
 int positives=0;
 for(int past=0;past<32768;past+=512){kimi_tp8_pp4_r410::call_scope(0,1,1,past,512,3);++positives;}
 for(int past:{1,8192,32639,32767}){kimi_tp8_pp4_r410::call_scope(1,1,1,past,1,3);++positives;}
 reject([]{kimi_tp8_pp4_r410::call_scope(0,1,1,32767,2,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(0,1,1,0,513,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,1,0,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,1,512,2,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,2,1,512,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,2,512,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,1,-1,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,1,32768,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(2,1,1,512,1,3);});
 reject([]{kimi_tp8_pp4_r410::call_scope(1,1,1,512,1,7);});
 // The actual persistent UInt8 scales have full I384 storage, no padded fallback.
 constexpr std::size_t E=896,H=3584,I=384;
 constexpr std::size_t s13=E*2*I*(H/32),s2=E*H*(I/32);
 static_assert(s13==77070336&&s2==38535168);
 std::size_t aliases=0;
 for(std::size_t e=0;e<E;++e){
  for(std::size_t row:{std::size_t(0),I}){
   auto begin=e*2*I*(H/32)+row*(H/32),end=begin+I*(H/32);
   assert(end<=s13);++aliases;
  }
  for(std::size_t col:{std::size_t(0),std::size_t(6)}){
   auto last=e*H*(I/32)+(H-1)*(I/32)+col+6-1;
   assert(last<s2);++aliases;
  }
 }
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":0,\"call_positives\":"<<positives
 <<",\"I384_scale_alias_bounds\":"<<aliases<<",\"scale13_bytes\":"<<s13
 <<",\"scale2_bytes\":"<<s2<<",\"Runtime_created\":false}"<<std::endl;
}
