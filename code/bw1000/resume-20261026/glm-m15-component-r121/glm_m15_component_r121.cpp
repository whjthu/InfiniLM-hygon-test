#include "glm_fp8_cache_m15_r119.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
thread_local std::string error;
constexpr size_t M=15,N=6144,K=2048,SR=48,SC=128;
struct Context{std::shared_ptr<rt::Runtime>r;void*cache{};int rank{};bool enabled=false,busy=false;std::array<uint64_t,5>counts{};};
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;bool pending=true;~Work(){if(pending)try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto t=rt::Tensor::empty(r,shape,type);held.push_back(t);return t;}void finish(){r.syncStream();pending=false;}};
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||!c->r)throw std::invalid_argument("r121 null context");c->r->assertOwnerThread();if(c->r->isGraphRecording())throw std::logic_error("r121 eager only");int d=-1;if(cudaGetDevice(&d)!=cudaSuccess||d!=c->r->device().index())throw std::logic_error("r121 caller device");}
void pointer(Context*c,const void*p){if(!p)throw std::invalid_argument("r121 null pointer");cudaPointerAttributes a{};if(cudaPointerGetAttributes(&a,p)!=cudaSuccess||a.type!=cudaMemoryTypeDevice||a.device!=c->r->device().index())throw std::invalid_argument("r121 pointer owner");}
}
extern "C" const char*glm_m15_component_error_r121(){return error.c_str();}
extern "C" void*glm_m15_component_create_r121(int device,int logical_rank){std::unique_ptr<Context>c;try{
 if(device<0||device>7||logical_rank<0||logical_rank>7)throw std::invalid_argument("r121 device/rank");
 infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};c=std::make_unique<Context>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();c->rank=logical_rank;return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" int glm_m15_component_policy_r121(void*raw,int enabled){auto*c=static_cast<Context*>(raw);try{owner(c);if(c->busy||(enabled!=0&&enabled!=1))throw std::logic_error("r121 idle strict policy");c->r->syncStream();if(c->enabled!=bool(enabled)){c->enabled=bool(enabled);++c->counts[3];}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
// Both arms include original BF16->FP8 quantization, global K128 scale placement,
// pointer/Tensor metadata, provider call and original Runtime final drain.
extern "C" int glm_m15_component_forward_r121(void*raw,const void*input,const void*weight,const void*weight_scale,void*output,void*producer_stream,double*wall_ms){auto*c=static_cast<Context*>(raw);bool entered=false;try{
 owner(c);if(c->busy||!wall_ms)throw std::logic_error("r121 pending forward");for(auto*p:{input,weight,weight_scale,static_cast<const void*>(output)})pointer(c,p);
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer_stream))!=cudaSuccess)throw std::runtime_error("r121 producer drain");
 c->busy=true;entered=true;auto start=std::chrono::steady_clock::now();auto&r=*c->r;Work work{r};const size_t co=size_t(c->rank)*K;
 auto x=rt::Tensor::from_blob(const_cast<void*>(input),{M,K},rt::DataType::kBFloat16,r.device());
 auto w=rt::Tensor::from_blob(const_cast<void*>(weight),{N,K},rt::DataType::kUInt8,r.device());
 auto ws=rt::Tensor::from_blob(const_cast<void*>(weight_scale),{SR,SC},rt::DataType::kFloat32,r.device());
 auto y=rt::Tensor::from_blob(output,{M,N},rt::DataType::kBFloat16,r.device());
 auto packed=work.alloc({M,K},rt::DataType::kUInt8),local=work.alloc({M,K/128},rt::DataType::kFloat32),global=work.alloc({M,SC},rt::DataType::kFloat32);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),x,packed,local);
 r.setDeviceMemory(global->data(),0xff,global->nbytes());global->narrow({{1,co/128,K/128}})->copy_from(r,local);
 if(c->enabled){
  if(!c->cache){c->cache=glm_fp8_cache_create_r119(r.device().index(),r.stream(),&r);if(!c->cache)throw std::runtime_error(glm_fp8_cache_error_r119());}
  if(glm_fp8_cache_run_r119(c->cache,&r,r.stream(),1,1,M,N,K,SR,SC,0,co,output,packed->data(),global->data(),weight,weight_scale))throw std::runtime_error(glm_fp8_cache_error_r119());++c->counts[2];
 }else{ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),y,packed,global,w,ws,size_t(0),co);++c->counts[1];}
 work.finish();*wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();++c->counts[0];c->busy=false;return 0;
 }catch(const std::exception&e){error=e.what();if(c&&entered){++c->counts[4];drain(c);c->busy=false;}return 1;}}
extern "C" int glm_m15_component_stats_r121(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);try{owner(c);if(!out||n!=16||c->busy)throw std::invalid_argument("r121 stats");std::copy(c->counts.begin(),c->counts.end(),out);out[5]=c->enabled;if(c->cache)return glm_fp8_cache_stats_r119(c->cache,out+6,10);std::fill(out+6,out+16,0);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_m15_component_clear_r121(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);if(c->busy)throw std::logic_error("r121 busy clear");c->r->syncStream();if(c->cache){if(glm_fp8_cache_clear_r119(c->cache,c->r.get(),c->r->stream()))throw std::runtime_error(glm_fp8_cache_error_r119());uint64_t v[10]{};if(glm_fp8_cache_stats_r119(c->cache,v,10)||v[0]!=v[1]||v[5]||v[6])throw std::logic_error("r121 private cache not empty");}return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int glm_m15_component_destroy_r121(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);if(glm_m15_component_clear_r121(raw))throw std::runtime_error(error);if(c->cache&&glm_fp8_cache_destroy_r119(c->cache,c->r.get(),c->r->stream()))throw std::runtime_error(glm_fp8_cache_error_r119());c->cache=nullptr;c->r->trimMemory();c->r.reset();delete c;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
