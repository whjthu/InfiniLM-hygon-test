#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <optional>
#include <string>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{thread_local std::shared_ptr<rt::Runtime> owner;thread_local std::string error;
rt::Tensor view(rt::Runtime&r,void*p,rt::Shape s,rt::DataType d){if(!p)throw std::invalid_argument("null");return rt::Tensor::from_blob(p,s,d,r.device());}}
extern "C" const char*glm_i8_error_r9127(){return error.c_str();}
extern "C" int glm_i8_run_r9127(size_t m,size_t n,size_t k,void*a,void*qa,void*as,void*w,void*ws,void*out,void*producer){try{
 if(!(m==1||m==4||m==8||m==16||m==128)||!((k==6144&&n==512)||(k==256&&n==6144)))throw std::invalid_argument("shape");
 infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,0,true};
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("producer sync");
 if(!owner){owner=rt::Runtime::create({rt::Device::Type::kHygon,0});owner->enableOwnerDrainFailStop();}auto&r=*owner;r.assertOwnerThread();
 auto x=view(r,a,{m,k},rt::DataType::kBFloat16),packed=view(r,qa,{m,k},rt::DataType::kInt8),xs=view(r,as,{m,1},rt::DataType::kFloat32),weight=view(r,w,{n,k},rt::DataType::kInt8),scale=view(r,ws,{n,1},rt::DataType::kFloat32),y=view(r,out,{m,n},rt::DataType::kBFloat16);
 auto cfg=infini::ops::PerChannelQuantI8::DefaultConfig(r.device().type());cfg.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,cfg,x,packed,xs);
 auto gc=infini::ops::I8Gemm::DefaultConfig(r.device().type());gc.set_implementation_index(0);
 ex::submit<infini::ops::I8Gemm>(r,ex::RecordingPolicy::HostOnly,gc,y,packed,xs,weight->permute({1,0}),scale,std::optional<rt::Tensor>{});r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int glm_i8_shutdown_r9127(){try{if(owner){owner->assertOwnerThread();owner->syncStream();owner.reset();}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
