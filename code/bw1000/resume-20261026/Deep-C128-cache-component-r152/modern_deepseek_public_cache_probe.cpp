#include "execution/submit.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <map>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{thread_local std::string error;std::map<int,std::shared_ptr<rt::Runtime>>&workers(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}std::shared_ptr<rt::Runtime>worker(int d){auto&r=workers()[d];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();}return r;}}
extern "C" const char*modern_deep_public_cache_error(){return error.c_str();}
extern "C" int modern_deep_public_cache(int mode,int device,size_t count,size_t pages,size_t block,size_t stride,void*cache,const void*slots,void*values,void*stream){std::shared_ptr<rt::Runtime>owner;rt::Tensor flag;try{
 if(mode<0||mode>1)throw std::invalid_argument("Invalid public V4 cache mode");if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("Public cache producer stream failed");owner=worker(device);auto&r=*owner;
 auto view=[&](const void*p,rt::Shape shape,rt::DataType type){return rt::Tensor::from_blob(const_cast<void*>(p),shape,type,r.device());};flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);auto raw=view(cache,{pages,stride},rt::DataType::kUInt8),ids=view(slots,{count},rt::DataType::kInt64),data=view(values,{count,512},rt::DataType::kBFloat16);
 ex::submit<infini::ops::DeepseekV4PagedCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4PagedCache::DefaultConfig(r.device().type()),raw,ids,data,flag,block,size_t(mode));int status=0;r.memcpyD2H(&status,flag->data(),4);r.syncStream();if(status){error="V4 cache GPU validation status="+std::to_string(status);return 1;}return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int modern_deep_public_cache_shutdown(){try{for(auto&item:workers()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}workers().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
