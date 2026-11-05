#include "contract.hpp"
#include "mixed_prefill.hpp"
#include "layers/linear/dense_prefill_f32_r2475.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
namespace rt=infinilm::runtime;
namespace {
using Clock=std::chrono::steady_clock;
thread_local std::string error;
struct Context {
 std::shared_ptr<rt::Runtime> r;
 std::thread::id owner;
 rt::Tensor x,weight,backing,y;
 int id=-1; bool poison=false;
 std::uint64_t counts[4]{};
};
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){
 if(!c||c->poison||c->owner!=std::this_thread::get_id())throw std::invalid_argument("mixed null/poison/wrong owner");
 int device=-1;check(cudaGetDevice(&device));if(device!=0)throw std::invalid_argument("mixed wrong visible device");
 c->r->assertOwnerThread();
}
void memory(std::uint64_t*out){std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));out[0]=free;out[1]=total;}
void clear(Context*c){drain(c);c->x={};c->weight={};c->y={};c->backing={};drain(c);}
struct Work {Context*c;rt::Tensor sum,host;std::vector<std::uint16_t>guard;~Work(){drain(c);}};
}
extern "C" const char*kimi_mixed_error_r461(){return error.c_str();}
extern "C" void*kimi_mixed_create_r461(int device){
 std::unique_ptr<Context>c;
 try{
  if(device!=0)throw std::invalid_argument("mixed one visible Hygon device0");
  check(cudaSetDevice(device));c=std::make_unique<Context>();c->owner=std::this_thread::get_id();
  c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
  return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}
}
extern "C" int kimi_mixed_prepare_r461(void*raw,int id,const void*x,std::size_t nx,const void*w,std::size_t nw){
 auto*c=static_cast<Context*>(raw);
 if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong owner before Runtime";return 1;}
 try{
  owner(c);const auto g=kimi_mixed_r461::geometry(id);kimi_mixed_r461::validate(id,0,nx,nw,g.m*g.n);
  if(!x||!w)throw std::invalid_argument("mixed complete CPU input/weight pointers");
  clear(c);
  auto hx=rt::Tensor::from_blob(const_cast<void*>(x),{g.m,g.k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  auto hw=rt::Tensor::from_blob(const_cast<void*>(w),{g.n,g.k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  c->x=rt::Tensor::empty(*c->r,{g.m,g.k},rt::DataType::kBFloat16);
  c->weight=rt::Tensor::empty(*c->r,{g.n,g.k},rt::DataType::kBFloat16);
  c->backing=rt::Tensor::empty(*c->r,{g.m*g.n+32},rt::DataType::kBFloat16);
  c->y=c->backing->narrow({{0,16,g.m*g.n}})->view({g.m,g.n});
  c->x->copy_from(*c->r,hx);c->weight->copy_from(*c->r,hw);c->r->syncStream();c->id=id;return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
// Driver observations are not instantaneous CUDA/workspace or pool highwater.
// metrics: dispatch+drain, D2H+canary, entirecall; memory points before,
// afterdrain with F32sum live, aftersum release. Returned rawbytes are BF16.
extern "C" int kimi_mixed_run_r461(void*raw,int mode,void*out,std::size_t ny,double*metrics,std::size_t nm,std::uint64_t*mem,std::size_t nb){
 auto*c=static_cast<Context*>(raw);const auto start=Clock::now();
 if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong owner before Runtime";return 1;}
 try{
  owner(c);const auto g=kimi_mixed_r461::geometry(c->id);kimi_mixed_r461::validate(c->id,mode,g.m*g.k,g.n*g.k,ny);
  if(!out||!metrics||nm!=3||!mem||nb!=6)throw std::invalid_argument("mixed output/metrics schema");
  {Work w{c};w.guard.assign(ny+32,0x3f25);
  w.host=rt::Tensor::from_blob(w.guard.data(),{ny+32},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  c->backing->copy_from(*c->r,w.host);c->r->syncStream();memory(mem);
  const auto dispatch=Clock::now();auto weight=c->weight->permute({1,0});
  w.sum=mode?kimi_mixed_prefill_r461::project(*c->r,c->x,weight,c->y,1.0f)
               :infinilm::layers::linear::dense_prefill_f32_r2475(*c->r,c->x,weight,c->y,1.0f);
  c->r->syncStream();metrics[0]=std::chrono::duration<double>(Clock::now()-dispatch).count();memory(mem+2);
  const auto read=Clock::now();w.host->copy_from(*c->r,c->backing);c->r->syncStream();
  for(std::size_t i=0;i<16;++i)if(w.guard[i]!=0x3f25||w.guard[ny+16+i]!=0x3f25)throw std::runtime_error("mixed output canary");
  std::memcpy(out,w.guard.data()+16,ny*2);metrics[1]=std::chrono::duration<double>(Clock::now()-read).count();
  w.sum={};drain(c);memory(mem+4);++c->counts[c->id*2+mode];}
  metrics[2]=std::chrono::duration<double>(Clock::now()-start).count();return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
// Only the component cold/cache boundary; never injected into Model/reset.
extern "C" int kimi_mixed_trim_r461(void*raw,std::uint64_t*out,std::size_t n){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong owner before Runtime";return 1;}try{owner(c);if(!out||n!=4)throw std::invalid_argument("mixed trim schema");
  drain(c);memory(out);c->r->trimMemory();memory(out+2);return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_mixed_counts_r461(void*raw,std::uint64_t*out,std::size_t n){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong owner before Runtime";return 1;}try{owner(c);if(!out||n!=4)throw std::invalid_argument("mixed count schema");std::memcpy(out,c->counts,sizeof(c->counts));return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int kimi_mixed_destroy_r461(void*raw){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong destroy owner";return 1;}
 clear(c);c->r.reset();delete c;return 0;
}
