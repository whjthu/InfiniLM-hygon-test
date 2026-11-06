// Actual PagedIndexerCache consumer; no C4 weights, RankInfo, or replacement SDK.
#include "deepseek_v4_c4_indexer.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cstdint>
extern "C" int deep_constant_slot_metadata_set_r482(int);
extern "C" uint64_t deep_constant_slot_metadata_stats_r482(int,int);
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4;
namespace {
using Clock=std::chrono::steady_clock;
thread_local std::string error;
thread_local bool ever_created=false;
struct Context {
 std::shared_ptr<rt::Runtime> r;
 std::vector<std::unique_ptr<md::PagedIndexerCache>> caches;
 std::thread::id thread=std::this_thread::get_id();
 uint64_t stores=0,gathers=0,guards=0,resets=0;
 bool poisoned=false;
};
void cuda_check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void memory(uint64_t*out){size_t f=0,t=0;cuda_check(cudaMemGetInfo(&f,&t));out[0]=f;out[1]=t;}
void floor(){uint64_t x[2];memory(x);if(x[0]<uint64_t(16)*1024*1024*1024)throw std::runtime_error("retained16GiB observed floor");}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||!c->r||c->poisoned||c->thread!=std::this_thread::get_id())throw std::logic_error("null/poisoned/wrong owner");c->r->assertOwnerThread();int d=-1;cuda_check(cudaGetDevice(&d));if(d!=0)throw std::logic_error("exact visible local device0");}
md::PagedIndexerCache&cache(Context*c,size_t layer){owner(c);if(layer>=21||c->caches.size()!=21)throw std::invalid_argument("21 C4 caches");return *c->caches[layer];}
template<class F>int invoke(Context*c,F f){try{owner(c);floor();f();floor();return 0;}catch(const std::exception&e){error=e.what();if(c)c->poisoned=true;drain(c);return 1;}}
struct Inputs {
 rt::Runtime&r;
 std::vector<rt::Tensor> held;
 std::vector<uint8_t> values;
 std::vector<float> scales;
 ~Inputs(){try{r.syncStream();}catch(...){std::terminate();}}
};
}
extern "C" const char*deep_slot_probe_error_r482(){return error.c_str();}
extern "C" void*deep_slot_probe_create_r482(int mode){
 std::unique_ptr<Context>c;
 try{
  if(ever_created||(mode!=0&&mode!=1))throw std::logic_error("one phase/Runtime per process; fixed startup policy");ever_created=true;
  int n=0;cuda_check(cudaGetDeviceCount(&n));if(n!=1)throw std::logic_error("one physical allocation visible");cuda_check(cudaSetDevice(0));floor();
  if(deep_constant_slot_metadata_set_r482(mode))throw std::logic_error("policy already frozen");
  c=std::make_unique<Context>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,0});c->r->enableOwnerDrainFailStop();
  for(size_t i=0;i<21;++i)c->caches.emplace_back(std::make_unique<md::PagedIndexerCache>(*c->r,128));
  c->r->syncStream();floor();if(deep_constant_slot_metadata_set_r482(mode)!=2)throw std::logic_error("firstcache must freeze policy");return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c)c->caches.clear();drain(c.get());return nullptr;}
}
extern "C" int deep_slot_probe_seed_r482(void*raw,size_t layer,size_t begin,size_t count,const void*values,const float*scales){
 auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
  auto&k=cache(c,layer);if(!count||count>8192||begin>=8192||count>8192-begin||!values||!scales)throw std::invalid_argument("full32768 compressed8192 seed scope");
  auto&r=*c->r;Inputs work{r};work.values.assign(count*128+32,uint8_t(0xa5));work.scales.assign(count+32,-3.25f);std::memcpy(work.values.data()+16,values,count*128);std::memcpy(work.scales.data()+16,scales,count*4);
  auto vb=rt::Tensor::empty(r,{work.values.size()},rt::DataType::kUInt8),sb=rt::Tensor::empty(r,{work.scales.size()},rt::DataType::kFloat32);work.held={vb,sb};
  r.memcpyH2D(vb->data(),work.values.data(),work.values.size(),false);r.memcpyH2D(sb->data(),work.scales.data(),work.scales.size()*4,false);
  std::vector<int64_t>slots(count);for(size_t i=0;i<count;++i)slots[i]=int64_t(begin+i);
  k.store(r,slots,{vb->narrow({{0,16,count*128}})->view({count,128}),sb->narrow({{0,16,count}})->view({count,1})});
  std::vector<uint8_t>actual_v(work.values.size());std::vector<float>actual_s(work.scales.size());r.memcpyD2H(actual_v.data(),vb->data(),actual_v.size());r.memcpyD2H(actual_s.data(),sb->data(),actual_s.size()*4);r.syncStream();
  if(actual_v!=work.values||std::memcmp(actual_s.data(),work.scales.data(),actual_s.size()*4))throw std::runtime_error("actual GPU seed input bits or guards changed");++c->stores;
 });
}
extern "C" int deep_slot_probe_gather_r482(void*raw,size_t layer,const int64_t*slots,size_t n,void*out,size_t bytes,float*scales,size_t ns,double*seconds,size_t clocks){
 auto*c=static_cast<Context*>(raw);auto start=Clock::now();return invoke(c,[&]{
  auto&k=cache(c,layer);if(!slots||!n||n>8192||!out||bytes!=n*128||!scales||ns!=n||!seconds||clocks!=3)throw std::invalid_argument("fullcache gather output scope");
  std::vector<int64_t>ids(slots,slots+n);auto before=Clock::now();auto result=k.gather(*c->r,ids);auto after=Clock::now();
  if(result.values->shape()!=rt::Shape{n,128}||result.scales->shape()!=rt::Shape{n,1})throw std::runtime_error("original gather shape changed");
  c->r->memcpyD2H(out,result.values->data(),bytes);c->r->memcpyD2H(scales,result.scales->data(),n*4);c->r->syncStream();auto copied=Clock::now();
  ++c->gathers;seconds[0]=std::chrono::duration<double>(after-before).count();seconds[1]=std::chrono::duration<double>(copied-after).count();seconds[2]=std::chrono::duration<double>(copied-start).count();
 });
}
extern "C" int deep_slot_probe_guard_r482(void*raw,size_t layer,int kind){
 auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
  auto&k=cache(c,layer);if(kind<0||kind>4)throw std::invalid_argument("guard kind");bool rejected=false;
  try{
   if(kind==0)k.gather(*c->r,{});
   else if(kind==1)k.gather(*c->r,{-1});
   else if(kind==2)k.gather(*c->r,{8192});
   else if(kind==3)k.store(*c->r,{0,0},{});
   else k.gather(*c->r,{8191}); // Only call before seed or after reset.
  }catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("required original guard did not reject");++c->guards;
 });
}
extern "C" int deep_slot_probe_reset_r482(void*raw,size_t layer){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{cache(c,layer).reset(*c->r);++c->resets;});}
extern "C" int deep_slot_probe_raw_r482(void*raw,size_t layer,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto&k=cache(c,layer);if(!out||bytes!=128*8448||k.raw()->shape()!=rt::Shape{128,8448}||k.raw()->nbytes()!=bytes)throw std::invalid_argument("exact rawcache layout");c->r->memcpyD2H(out,k.raw()->data(),bytes);c->r->syncStream();});}
extern "C" int deep_slot_probe_observe_r482(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(!out||n!=16)throw std::invalid_argument("observation width");memory(out);out[2]=21;out[3]=128;out[4]=128*8448;out[5]=c->stores;out[6]=c->gathers;out[7]=c->guards;out[8]=c->resets;for(int i=0;i<5;++i)out[9+i]=deep_constant_slot_metadata_stats_r482(0,i);out[14]=reinterpret_cast<uintptr_t>(c->r.get());out[15]=reinterpret_cast<uintptr_t>(c->r->stream());});}
extern "C" int deep_slot_probe_destroy_r482(void*raw,uint64_t*out,size_t n){
 auto*c=static_cast<Context*>(raw);try{if(!c||!c->r||c->thread!=std::this_thread::get_id())throw std::logic_error("destroy null/wrong owner");c->r->assertOwnerThread();if(!out||n!=4)throw std::invalid_argument("close width");uint64_t m[2];memory(m);out[0]=m[0];drain(c);c->caches.clear();c->r->syncStream();c->r->trimMemory();std::weak_ptr<rt::Runtime>last=c->r;c->r.reset();delete c;c=nullptr;memory(m);out[1]=m[0];out[2]=last.expired();out[3]=21;return out[2]?0:1;}catch(const std::exception&e){error=e.what();drain(c);return 1;}
}
