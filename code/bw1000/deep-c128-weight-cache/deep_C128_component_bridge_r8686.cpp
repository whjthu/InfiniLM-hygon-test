#include "deepseek_v4_temporal_compressor.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <map>
#include <set>
namespace rt=infinilm::runtime;namespace model=infinilm::models::deepseek_v4;
namespace {
thread_local std::string error,serialized;
struct Context { std::shared_ptr<rt::Runtime>owner;std::unique_ptr<model::TemporalCompressor>module;std::set<std::string>loaded; };
std::map<int,std::shared_ptr<rt::Runtime>>&owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}
void drain(Context*c){if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}}
void producer(void*stream){if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("Compressor producer stream failed");}
}
extern "C" const char*modern_real_compress_error(){return error.c_str();}
extern "C" void*modern_real_compress_create(int device,size_t layer,size_t dim,size_t rows,size_t capacity,const char*cfg){std::unique_ptr<Context>c;try{
 c=std::make_unique<Context>();auto&worker=owners()[device];if(!worker){worker=rt::Runtime::create({rt::Device::Type::kHygon,device});worker->enableOwnerDrainFailStop();}c->owner=worker;
 infini::rt::DeviceGuard guard{worker->device().type(),device,true};c->module=std::make_unique<model::TemporalCompressor>(*worker,nlohmann::json::parse(cfg),layer,dim,rows,capacity);return c.release();
}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" const char*modern_real_compress_keys(void*raw){serialized=nlohmann::json(static_cast<Context*>(raw)->module->state_dict_keys()).dump();return serialized.c_str();}
extern "C" int modern_real_compress_load(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int f32){auto*c=static_cast<Context*>(raw);try{
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),f32?rt::DataType::kFloat32:rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});c->module->load_parameter(*c->owner,name,x);c->loaded.insert(name);return 0;
}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_real_compress_finish(void*raw){auto*c=static_cast<Context*>(raw);try{
 for(auto&name:c->module->state_dict_keys())if(!c->loaded.count(name))throw std::runtime_error("Missing compressor weight "+name);
 c->module->process_weights_after_loading(*c->owner,false);return 0;
}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_real_compress_forward(void*raw,size_t tokens,size_t count,const size_t*requests,const void*x,const void*positions,void*score,void*pooled,void*rotated,int abort,void*stream){auto*c=static_cast<Context*>(raw);try{
 if(!count||count>128||!requests)throw std::invalid_argument("Invalid compressor request count");producer(stream);auto&r=*c->owner;size_t dim=c->module->dim(),channels=c->module->ratio()==4?4:2;
 std::vector<model::TemporalCompressor::Request>metadata;for(size_t i=0;i<count;++i)metadata.push_back({requests[3*i],requests[3*i+1],requests[3*i+2]});
 auto input=rt::Tensor::from_blob(const_cast<void*>(x),{tokens,4096},rt::DataType::kBFloat16,r.device()),pos=rt::Tensor::from_blob(const_cast<void*>(positions),{tokens},rt::DataType::kInt64,r.device());auto batch=c->module->prepare(r,input,pos,metadata);
 auto scattered=rt::Tensor::empty(r,{tokens,dim},rt::DataType::kBFloat16);r.setDeviceMemory(scattered->data(),0,scattered->nbytes());
 for(size_t i=0;i<batch.boundaries.size();++i)scattered->narrow({{0,size_t(batch.boundaries[i]),1}})->copy_from(r,batch.rotated->narrow({{0,i,1}}));
 if(!abort)c->module->commit(r,batch);
 rt::Tensor::from_blob(score,{tokens,size_t(channels)*dim},rt::DataType::kBFloat16,r.device())->copy_from(r,batch.score);
 rt::Tensor::from_blob(pooled,{tokens,dim},rt::DataType::kBFloat16,r.device())->copy_from(r,batch.pooled);
 rt::Tensor::from_blob(rotated,{tokens,dim},rt::DataType::kBFloat16,r.device())->copy_from(r,scattered);r.syncStream();return 0;
}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" size_t modern_real_compress_bytes(void*raw,int cache){auto*c=static_cast<Context*>(raw);auto source=cache?c->module->raw_cache():c->module->state();return source?source->nbytes():0;}
extern "C" int modern_real_compress_snapshot(void*raw,int cache,void*out,void*stream){auto*c=static_cast<Context*>(raw);try{producer(stream);auto&r=*c->owner;auto source=cache?c->module->raw_cache():c->module->state();if(!source)throw std::invalid_argument("No indexer cache in this module");rt::Tensor::from_blob(out,source->shape(),source->dtype(),r.device())->copy_from(r,source);r.syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_real_compress_gather(void*raw,size_t count,const size_t*entries,void*out,void*stream){auto*c=static_cast<Context*>(raw);try{if(!count||count>65536||!entries)throw std::invalid_argument("Invalid compressed selection count");producer(stream);auto&r=*c->owner;std::vector<std::pair<size_t,size_t>>items;for(size_t i=0;i<count;++i)items.push_back({entries[2*i],entries[2*i+1]});auto result=c->module->gather(r,items);rt::Tensor::from_blob(out,{count,512},rt::DataType::kBFloat16,r.device())->copy_from(r,result);r.syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_real_compress_reset(void*raw){auto*c=static_cast<Context*>(raw);try{c->module->reset(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_real_compress_destroy(void*raw){auto*c=static_cast<Context*>(raw);drain(c);c->module.reset();drain(c);delete c;return 0;}
extern "C" int modern_real_compress_release_workers(){try{for(auto&item:owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
