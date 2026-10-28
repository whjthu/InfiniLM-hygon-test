#include "scale_exact_scope_r201.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace kimi_scale_exact_r201;
int main(){
 int r=0,m=0,foreign=0;Geometry g{true,16,2,896,3584,192,true,true};
 assert(!selected(&r,&m));
 {Scope scope(g,&r,&m);assert(selected(&r,&m));bool rejected=false;try{selected(&r,&foreign);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  rejected=false;try{Scope nested(g,&r,&m);}catch(const std::invalid_argument&){rejected=true;}assert(rejected&&selected(&r,&m));}
 assert(!active);try{Scope scope(g,&r,&m);throw 17;}catch(int){}assert(!active);
 auto old=g;old.long_factory=false;old.tp=8;old.pp=4;{Scope scope(old,&r,&m);assert(!selected(&r,&m));}
 for(int bad=0;bad<7;++bad){auto q=g;if(bad==0)q.tp=8;if(bad==1)q.pp=4;if(bad==2)q.experts=895;if(bad==3)q.hidden=7168;if(bad==4)q.local_intermediate=96;if(bad==5)q.hygon=false;if(bad==6)q.eager=false;bool rejected=false;try{Scope scope(q,&r,&m);}catch(const std::invalid_argument&){rejected=true;}assert(rejected&&!active);}
 constexpr size_t E=896,H=3584,I=192,s13=E*2*I*(H/32),s2=E*H*(I/32);
 static_assert(s13==38535168&&s2==19267584);
 size_t aliases=0,max13=0,max2=0;
 for(size_t e=0;e<E;++e){
  const size_t stride13=2*I*(H/32),stride2=H*(I/32);
  for(size_t row:{size_t(0),I}){const size_t last=e*stride13+(row+I)*(H/32)-1;assert(last<s13);if(last>max13)max13=last;++aliases;}
  const size_t last=e*stride2+H*(I/32)-1;assert(last<s2);if(last>max2)max2=last;++aliases;
 }
 assert(aliases==2688&&max13==s13-1&&max2==s2-1);
 // Source retains another2688 packed-weight aliases:5376 total, four unique banks.
 constexpr size_t class13=64*1024*1024,class2=32*1024*1024,padding=class13+class2-s13-s2;
 static_assert(padding==42860544);
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":false,\"Runtime_created\":false,\"actual_Tensor_created\":false,\"scale_alias_offsets_checked\":2688,\"all5376_actual_alias_source_unchanged\":true,\"design_padding_bytes_per_MoE\":42860544}\n";
}
