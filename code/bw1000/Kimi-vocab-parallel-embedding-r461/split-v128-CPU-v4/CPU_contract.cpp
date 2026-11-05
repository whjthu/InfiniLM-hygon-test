#include "contract.hpp"
#include "merge_math.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#ifdef KIMI_COMPILE_SDK_PLANNING
#include <base/flash_attn_with_kvcache.h>
#endif
#include <type_traits>
int main(){
 using namespace kimi_split_v128;size_t cases=0,queries=0,maximum_calls=0;
#ifdef KIMI_COMPILE_SDK_PLANNING
 {using infini::ops::Tensor;using infini::ops::DataType;using Device=std::decay_t<decltype(std::declval<Tensor>().device())>;
  Device device{Device::Type::kHygon,0};Tensor q(nullptr,{1,32,3,192},DataType::kBFloat16,device),k(nullptr,{1,3072,3,192},DataType::kBFloat16,device),v(nullptr,{1,3072,3,192},DataType::kBFloat16,device),bad_v(nullptr,{1,3072,3,128},DataType::kBFloat16,device),out(nullptr,{1,32,3,192},DataType::kBFloat16,device),lse(nullptr,{1,3,32},DataType::kFloat32,device),lengths(nullptr,{1},DataType::kInt32,device);
  const std::optional<Tensor>none;
  auto supports=[&](Tensor value){return infini::ops::FlashAttnWithKvcache::supports(q,k,value,none,none,none,none,std::optional<Tensor>{lengths},none,none,none,none,std::optional<double>{1.0},true,std::vector<int64_t>{-1,-1},0.0,true,int64_t{1},true,out,std::optional<Tensor>{lse});};
  assert(supports(v)&&!supports(bad_v));
 }
#endif

 for(size_t total:{size_t(1),size_t(32),size_t(256),size_t(3071),size_t(3072),size_t(3073),size_t(32768)})for(size_t count:{size_t(1),size_t(8),size_t(32),size_t(512)}){
  if(count>total){continue;}auto p=plan(count,total);size_t done=0,calls=0;
  while(done<count){auto t=tile(p,done);assert(t.rows>0&&t.rows<=32&&t.end<=total);assert((p.past+done)/3072==(t.end-1)/3072);
   for(size_t start=0;start<t.end;start+=3072){size_t len=std::min(size_t(3072),t.end-start);bool causal=start+len==t.end;++calls;
    for(size_t row=0;row<t.rows;++row)for(size_t j=0;j<len;++j){bool visible=!causal||j<=len-t.rows+row;assert(visible==(start+j<=p.past+done+row));}
   }done+=t.rows;
  }assert(done==count);queries+=count;++cases;maximum_calls=std::max(maximum_calls,calls);
 }
 authorize(false,false,false,false,2,2,1);authorize(true,true,true,true,4,1,1);int rejected=0;
 for(int bad=0;bad<4;++bad)try{authorize(true,bad!=0,bad!=1,true,bad==2?2:4,bad==3?2:1,1);}catch(const std::invalid_argument&){++rejected;}
 assert(rejected==4);
 for(float base:{-10000.f,-100.f,0.f,100.f,10000.f}){
  float m=0,den=0,sum=0;double refden=0,refsum=0;int i=0;
  for(float delta:{-2.f,-1.f,0.f}){float l=base+delta,partial=float(++i)/4.f;auto f=factors(m,den,l,i==1);sum=merge_value(sum,partial,f.a,f.b,i==1);m=f.maximum;den=f.denominator;double w=std::exp(double(delta));refden+=w;refsum+=w*partial;}
  assert(std::abs(sum/den-refsum/refden)<1e-6);
 }
 assert(old_bytes==77856768&&new_bytes==64880640&&per_layer_saving==12976128);
 assert(24*per_layer_saving==311427072);
 std::cout<<"{\"passed\":true,\"actual_shared_tile_cases\":"<<cases<<",\"actual_queries_covered\":"<<queries<<",\"maximum_FA_calls_per_layer\":"<<maximum_calls<<",\"old_factory_rejected\":4,\"stable_F32_LSE_extreme_cases\":5,\"GPU\":0,\"SDK_device_execution\":false,\"SDK_planning_supports_executed\":false}"<<std::endl;
}
