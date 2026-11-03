#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <cassert>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define D415_HD __host__ __device__
#else
#define D415_HD
#endif
namespace deep415 {
constexpr int N=2124,K=512,PAD=4096,BLOCK=256;
// NaNs are one highest-value equivalence class. Both zero signs are ties.
// Equal-value IDs may differ from Torch; values always copy original score[id].
D415_HD inline uint32_t value_key(uint32_t b){
 const auto a=b&0x7fffffffu;
 if(a>0x7f800000u)return 0xffffffffu;
 if(a==0)return 0x80000000u;
 return (b&0x80000000u)?~b:(b^0x80000000u);
}
D415_HD inline bool better(uint32_t a,int ai,uint32_t b,int bi){
 const bool av=ai>=0&&ai<N,bv=bi>=0&&bi<N;
 if(av!=bv)return av;
 if(!av)return ai<bi;
 auto ak=value_key(a),bk=value_key(b);
 return ak!=bk?ak>bk:ai<bi;
}
struct Geometry{size_t rows,n,k;bool f32,i64,contiguous,largest,sorted;int64_t dim;};
inline bool eligible(const Geometry&g){return g.rows==1&&g.n==N&&g.k==K&&g.f32&&g.i64&&g.contiguous&&g.largest&&g.sorted&&g.dim==1;}
struct Stats{uint64_t calls=0,hits=0,fallbacks=0;};
struct State{bool active=false;int mode=0;Stats stats{};};
inline thread_local State state{};
template<class Candidate,class Original>void dispatch(const Geometry&g,Candidate&&candidate,Original&&original){
 ++state.stats.calls;
 if(state.active&&state.mode==1&&eligible(g)){++state.stats.hits;candidate();}
 else{++state.stats.fallbacks;original();}
}
template<class F>Stats owner_phase(int mode,F&&forward){
 if(mode<0||mode>1||state.active)throw std::invalid_argument("r415 explicit0/1 nonnested owner phase");
 state.stats={};state.active=true;state.mode=mode;
 try{forward();auto q=state.stats;state.active=false;state.mode=0;return q;}
 catch(...){state.active=false;state.mode=0;throw;}
}
inline std::vector<int32_t> host_network(const std::vector<uint32_t>&input){
 if(input.size()!=N)throw std::invalid_argument("r415 fixed input count");
 std::array<uint32_t,PAD>bits{};std::array<int,PAD>ids{};
 for(int i=0;i<PAD;++i){bits[i]=i<N?input[i]:0xff800000u;ids[i]=i;}
 for(int span=2;span<=PAD;span<<=1)for(int stride=span>>1;stride;stride>>=1)
  for(int i=0;i<PAD;++i){int j=i^stride;if(j<=i)continue;bool descending=(i&span)==0;
   bool exchange=descending?better(bits[j],ids[j],bits[i],ids[i]):better(bits[i],ids[i],bits[j],ids[j]);
   if(exchange){std::swap(bits[i],bits[j]);std::swap(ids[i],ids[j]);}}
 return std::vector<int32_t>(ids.begin(),ids.begin()+K);
}
inline int CPU_contract(){
 Geometry g{1,N,K,true,true,true,true,true,1};std::array<std::thread,8>threads;std::array<bool,8>ok{};
 for(int d=0;d<8;++d)threads[d]=std::thread([&,d]{
  assert(!state.active&&state.mode==0);
  int candidate=0,original=0;
  for(int mode:{0,1,0,1,1,0}){auto q=owner_phase(mode,[&]{dispatch(g,[&]{++candidate;},[&]{++original;});});assert(q.calls==1&&q.hits==uint64_t(mode)&&q.fallbacks==uint64_t(!mode));}
  assert(candidate==3&&original==3);
  for(int which=0;which<11;++which){auto x=g;switch(which){case 0:x.rows=2;break;case 1:x.n=N-1;break;case 2:x.n=N+1;break;case 3:x.k=511;break;case 4:x.k=513;break;case 5:x.f32=false;break;case 6:x.i64=false;break;case 7:x.contiguous=false;break;case 8:x.largest=false;break;case 9:x.sorted=false;break;case 10:x.dim=0;break;}auto q=owner_phase(1,[&]{dispatch(x,[]{throw std::logic_error("ineligible reached kernel");},[&]{++original;});});assert(q.hits==0&&q.fallbacks==1);}
  bool threw=false;try{owner_phase(1,[&]{dispatch(g,[]{throw std::invalid_argument("original failure containment");},[]{});});}catch(const std::invalid_argument&){threw=true;}assert(threw&&!state.active&&state.mode==0);
  threw=false;try{owner_phase(2,[]{});}catch(const std::invalid_argument&){threw=true;}assert(threw&&!state.active);
  auto restored=owner_phase(0,[&]{dispatch(g,[]{throw std::logic_error("defaultOFF kernel");},[]{});});assert(restored.hits==0&&restored.fallbacks==1);ok[d]=true;
 });
 for(auto&t:threads)t.join();for(bool x:ok)assert(x);return 0;
}
}
#undef D415_HD
