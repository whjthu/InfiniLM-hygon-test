#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <thread>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define D419_HD __host__ __device__
#else
#define D419_HD
#endif
namespace deep419 {
constexpr int N=256,K=6,BLOCK=256;
D419_HD inline float score(float x){float soft=x>20.f?x:log1pf(expf(x));return sqrtf(soft);}
D419_HD inline uint32_t bits(float x){union{float f;uint32_t u;}v{x};return v.u;}
D419_HD inline uint32_t value_key(float x){auto b=bits(x),a=b&0x7fffffffu;if(a>0x7f800000u)return 0xffffffffu;if(a==0)return 0x80000000u;return b&0x80000000u?~b:b^0x80000000u;}
D419_HD inline uint64_t rank_key(float x,int id){return (uint64_t(value_key(x))<<32)|(0xffffffffu-uint32_t(id));}
// Explicit F32 intermediates. This tree is disclosed, not asserted to be ATen's GPU sum6 tree.
D419_HD inline float normalize(const float*v,float epsilon,float factor,int i){float a=v[0]+v[1],b=v[2]+v[3],c=v[4]+v[5];float d=a+b;d=d+c;d=d+epsilon;float q=v[i]/d;return q*factor;}
struct Output{std::array<float,N>scores,selection;std::array<int64_t,K>ids;std::array<float,K>weights;};
inline Output host_suffix(const float*logits,const float*bias,float epsilon,float factor){
 Output out{};std::array<int,N>order{};
 for(int i=0;i<N;++i){out.scores[i]=score(logits[i]);out.selection[i]=out.scores[i]+bias[i];order[i]=i;}
 std::sort(order.begin(),order.end(),[&](int a,int b){return rank_key(out.selection[a],a)>rank_key(out.selection[b],b);});
 float selected[K];for(int i=0;i<K;++i){out.ids[i]=order[i];selected[i]=out.scores[order[i]];}
 for(int i=0;i<K;++i)out.weights[i]=normalize(selected,epsilon,factor,i);return out;
}
struct Geometry{size_t tokens,experts,topk;bool hash,f32logits,f32bias,f32scalars,i64ids,contiguous;};
inline bool eligible(const Geometry&g){return g.tokens==1&&g.experts==N&&g.topk==K&&!g.hash&&g.f32logits&&g.f32bias&&g.f32scalars&&g.i64ids&&g.contiguous;}
struct State{bool active=false;int mode=0;uint64_t calls=0,hits=0,fallbacks=0;};
inline thread_local State state{};
inline bool select(const Geometry&g){++state.calls;if(state.active&&state.mode==1&&eligible(g)){++state.hits;return true;}++state.fallbacks;return false;}
template<class F>State owner_phase(int mode,F f){if(state.active||mode<0||mode>1)throw std::invalid_argument("r419 nonnested explicit0/1 owner phase");state={true,mode,0,0,0};try{f();auto result=state;state={};return result;}catch(...){state={};throw;}}
inline int CPU_contract(){
 std::array<std::thread,8>threads;std::array<bool,8>ok{};
 for(int i=0;i<8;++i)threads[i]=std::thread([&,i]{Geometry g{1,N,K,false,true,true,true,true,true};
  for(int mode:{0,1,0,1,1,0}){auto s=owner_phase(mode,[&]{bool hit=select(g);if(hit!=bool(mode))throw std::logic_error("r419 actualdispatch differs");});if(s.calls!=1||s.hits!=uint64_t(mode)||s.fallbacks!=uint64_t(!mode))throw std::logic_error("r419 counters");}
  for(int n=0;n<9;++n){auto x=g;switch(n){case 0:x.tokens=2;break;case 1:x.experts=255;break;case 2:x.topk=5;break;case 3:x.hash=true;break;case 4:x.f32logits=false;break;case 5:x.f32bias=false;break;case 6:x.f32scalars=false;break;case 7:x.i64ids=false;break;case 8:x.contiguous=false;break;}
   auto s=owner_phase(1,[&]{if(select(x))throw std::logic_error("r419 ineligible hit");});if(s.fallbacks!=1)throw std::logic_error("r419 fallback count");}
  bool caught=false;try{owner_phase(1,[&]{select(g);throw std::runtime_error("original error");});}catch(...){caught=true;}
  if(!caught||state.active||state.mode)throw std::logic_error("r419 error restoration");ok[i]=true;
 });for(auto&t:threads)t.join();for(bool x:ok)if(!x)throw std::logic_error("r419 thread failure");return 0;
}
}
#undef D419_HD
