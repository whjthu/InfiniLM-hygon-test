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
#include <sys/syscall.h>
#include <unistd.h>
namespace rt=infinilm::runtime;
namespace {
using Clock=std::chrono::steady_clock;
thread_local std::string error;
struct Context {
 std::shared_ptr<rt::Runtime> r;
 std::thread::id owner;
 rt::Tensor x,weight,x_backing,w_backing,backing,y;
 std::vector<std::uint16_t>seed_x,seed_w;
 long owner_tid=0;
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
void clear(Context*c){drain(c);c->x={};c->weight={};c->y={};c->backing={};c->x_backing={};c->w_backing={};drain(c);c->seed_x.clear();c->seed_w.clear();}
struct Work {Context*c;rt::Tensor sum,host;std::vector<std::uint16_t>guard;~Work(){drain(c);}};
}
extern "C" const char*kimi_mixed_error_r461(){return error.c_str();}
extern "C" void*kimi_mixed_create_r461(int device){
 std::unique_ptr<Context>c;
 try{
  if(device!=0)throw std::invalid_argument("mixed one visible Hygon device0");
  check(cudaSetDevice(device));c=std::make_unique<Context>();c->owner=std::this_thread::get_id();c->owner_tid=syscall(SYS_gettid);
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
  c->seed_x.assign(nx+32,0x3f25);c->seed_w.assign(nw+32,0x3f25);
  std::memcpy(c->seed_x.data()+16,x,nx*2);std::memcpy(c->seed_w.data()+16,w,nw*2);
  auto hx=rt::Tensor::from_blob(c->seed_x.data(),{nx+32},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  auto hw=rt::Tensor::from_blob(c->seed_w.data(),{nw+32},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  c->x_backing=rt::Tensor::empty(*c->r,{nx+32},rt::DataType::kBFloat16);
  c->w_backing=rt::Tensor::empty(*c->r,{nw+32},rt::DataType::kBFloat16);
  c->x=c->x_backing->narrow({{0,16,nx}})->view({g.m,g.k});
  c->weight=c->w_backing->narrow({{0,16,nw}})->view({g.n,g.k});
  c->backing=rt::Tensor::empty(*c->r,{g.m*g.n+32},rt::DataType::kBFloat16);
  c->y=c->backing->narrow({{0,16,g.m*g.n}})->view({g.m,g.n});
  c->x_backing->copy_from(*c->r,hx);c->w_backing->copy_from(*c->r,hw);c->r->syncStream();c->id=id;return 0;
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
// Readback original full guarded CPU words, no extra compute or weight conversion.
extern "C" int kimi_mixed_inspect_r461(void*raw,std::uint64_t*out,std::size_t n){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong inspect owner";return 1;}
 struct Hold {Context*c;std::vector<std::uint16_t>x,w;rt::Tensor hx,hw;~Hold(){drain(c);}};
 try{owner(c);if(!out||n!=25||c->id<0)throw std::invalid_argument("mixed inspect schema");
  Hold h{c};h.x.resize(c->seed_x.size());h.w.resize(c->seed_w.size());
  h.hx=rt::Tensor::from_blob(h.x.data(),{h.x.size()},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  h.hw=rt::Tensor::from_blob(h.w.data(),{h.w.size()},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  h.hx->copy_from(*c->r,c->x_backing);h.hw->copy_from(*c->r,c->w_backing);c->r->syncStream();
  auto weight=c->weight->permute({1,0});
  const std::uint64_t values[]={std::uint64_t(c->r->device().index()),std::uint64_t(c->owner_tid),std::uint64_t(syscall(SYS_gettid)==c->owner_tid),std::uint64_t(c->id),
   std::uint64_t(c->x->size(0)),std::uint64_t(c->x->size(1)),std::uint64_t(c->x->stride(0)),std::uint64_t(c->x->stride(1)),
   std::uint64_t(c->weight->size(0)),std::uint64_t(c->weight->size(1)),std::uint64_t(c->weight->stride(0)),std::uint64_t(c->weight->stride(1)),
   std::uint64_t(weight->size(0)),std::uint64_t(weight->size(1)),std::uint64_t(weight->stride(0)),std::uint64_t(weight->stride(1)),
   reinterpret_cast<std::uintptr_t>(c->x->data()),reinterpret_cast<std::uintptr_t>(c->weight->data()),reinterpret_cast<std::uintptr_t>(c->y->data()),
   std::uint64_t(c->x->dtype()==rt::DataType::kBFloat16),std::uint64_t(c->weight->dtype()==rt::DataType::kBFloat16),std::uint64_t(c->y->dtype()==rt::DataType::kBFloat16),
   std::uint64_t(h.x==c->seed_x),std::uint64_t(h.w==c->seed_w),std::uint64_t(weight->stride(0)==1&&weight->stride(1)==std::ptrdiff_t(std::uint64_t(weight->size(0))))};
  std::memcpy(out,values,sizeof(values));
  if(!out[2]||!out[19]||!out[20]||!out[21]||!out[22]||!out[23]||!out[24])throw std::runtime_error("mixed original input/weight bits, canary, owner or transpose changed");
  return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_mixed_destroy_r461(void*raw){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()){error="mixed wrong destroy owner";return 1;}
 clear(c);c->r.reset();delete c;return 0;
}
