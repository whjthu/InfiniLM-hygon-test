#include "execution/submit.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <map>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{thread_local std::string error;std::map<int,std::shared_ptr<rt::Runtime>>&owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}std::shared_ptr<rt::Runtime>worker(int device){auto&r=owners()[device];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,device});r->enableOwnerDrainFailStop();}return r;}}
extern "C" const char*modern_deep_compress_error(){return error.c_str();}
extern "C" int modern_deep_compress(int device,size_t tokens,size_t dim,size_t rows,size_t ratio,int f32,void*out,const void*score,const void*ape,void*state,const void*positions,const void*writes,const void*extras,void*stream){std::shared_ptr<rt::Runtime>owner;rt::Tensor flag;try{
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("Compression producer stream failed");owner=worker(device);auto&r=*owner;auto view=[&](const void*p,rt::Shape shape,rt::DataType type){return rt::Tensor::from_blob(const_cast<void*>(p),shape,type,r.device());};size_t channels=ratio==4?4:2;flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);
 ex::submit<infini::ops::DeepseekV4Compress>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4Compress::DefaultConfig(r.device().type()),view(out,{tokens,dim},rt::DataType::kBFloat16),view(score,{tokens,channels*dim},rt::DataType::kBFloat16),view(ape,{ratio==4?size_t(8):size_t(128),dim},rt::DataType::kFloat32),view(state,{rows,channels*dim},f32?rt::DataType::kFloat32:rt::DataType::kBFloat16),view(positions,{tokens},rt::DataType::kInt64),view(writes,{tokens},rt::DataType::kInt64),view(extras,{tokens},rt::DataType::kInt64),flag,ratio);
 int status=0;r.memcpyD2H(&status,flag->data(),sizeof(status));r.syncStream();if(status){error="Compression GPU validation status="+std::to_string(status);return 1;}return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int modern_deep_compress_shutdown(){try{for(auto&item:owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
