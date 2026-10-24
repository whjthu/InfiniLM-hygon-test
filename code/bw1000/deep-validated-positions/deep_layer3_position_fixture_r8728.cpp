// Component-only fixed checkpoint layer3 C128. Embedding-derived hidden is NOT full-Model layer3 hidden.
#include "deepseek_v4_native_routed_decoder_r20.hpp"
#include "deep_reduce_wide_r3443.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <map>
#include <set>
namespace rt=infinilm::runtime;
namespace model=infinilm::models::deepseek_v4;
namespace {
thread_local std::string error,serialized;
class Partial:public infinilm::weights::ModelWeights {
public:
 std::unique_ptr<model::NativeDecoderR20>layer;
 std::vector<std::pair<std::string,rt::Tensor>>stages;
 Partial(rt::Runtime&r,const nlohmann::json&cfg,const infinilm::engine::distributed::RankInfo&rank){
  auto bounded=cfg;size_t ratio=cfg.at("compress_ratios").at(3);if(ratio)bounded["native_compressed_capacity"]=(1024+ratio-1)/ratio;
  layer=std::make_unique<model::NativeDecoderR20>(r,bounded,3,rank,32);
  for(auto&v:layer->state_dict())bind_weight("layers.3."+v.first,v.second);
 }
 void finish(rt::Runtime&r){layer->process_weights_after_loading(r,false);r.syncStream();}
 void reset(rt::Runtime&r){stages.clear();layer->reset(r);r.syncStream();}
 void forward(rt::Runtime&r,size_t tokens,size_t count,const size_t*req,const void*x,const void*ids,const void*positions,void*proof){
  auto view=[&](const void*p,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());};
  std::vector<model::NativeDecoderR20::Request>a;
  for(size_t i=0;i<count;++i)a.push_back({req[3*i],req[3*i+1],req[3*i+2]});
  auto residual=view(x,{tokens,4,4096},rt::DataType::kBFloat16),ti=view(ids,{tokens},rt::DataType::kInt64),tp=view(positions,{tokens},rt::DataType::kInt64);
  model::wide_r3443::Scope scope(r,count);
  stages.clear();auto out=layer->forward(r,residual,ti,tp,a);
  stages={{"entry",residual},{"token_ids",ti},{"positions",tp},
    {"hc_attn_collapsed",out.attention_prepared.collapsed},{"hc_attn_post",out.attention_prepared.post},{"hc_attn_comb",out.attention_prepared.comb},
    {"attn_norm",out.normalized_attention},{"attn_value",out.attention_value},{"attn_residual",out.attention_residual},
    {"hc_ffn_collapsed",out.ffn.prepared.hc.collapsed},{"hc_ffn_post",out.ffn.prepared.hc.post},{"hc_ffn_comb",out.ffn.prepared.hc.comb},
    {"ffn_norm",out.ffn.prepared.normalized},{"moe_indices",out.ffn.moe.indices},{"moe_weights",out.ffn.moe.weights},{"moe_value",out.ffn.moe.value},{"hidden",out.hidden}};
  r.syncStream();
 }

};
struct Context{std::shared_ptr<rt::Runtime>owner;infinicclComm_t comm=nullptr;std::unique_ptr<Partial>module;std::set<std::string>loaded;};
void drain(Context*c){if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}}
std::shared_ptr<rt::Runtime>worker(int d){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>owners;auto&r=owners[d];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();}return r;}
}
extern "C" const char*deep_partial_error_r7003(){return error.c_str();}
extern "C" size_t deep_partial_id_size_r7003(){return sizeof(infinicclUniqueId);}
extern "C" int deep_partial_id_r7003(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_partial_create_r7003(int device,const char*cfg,const void*raw_id){std::unique_ptr<Context>c;try{c=std::make_unique<Context>();c->owner=worker(device);auto&r=*c->owner;infini::rt::DeviceGuard guard{r.device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("Partial TP8 init failed");infinilm::engine::distributed::RankInfo rank(r.device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;c->module=std::make_unique<Partial>(r,nlohmann::json::parse(cfg),rank);return c.release();}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" const char*deep_partial_keys_r7003(void*raw){serialized=nlohmann::json(static_cast<Context*>(raw)->module->state_dict_keys()).dump();return serialized.c_str();}
extern "C" int deep_partial_load_r7003(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);try{auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dtype==0?rt::DataType::kInt8:dtype==1?rt::DataType::kFloat32:dtype==2?rt::DataType::kBFloat16:rt::DataType::kInt64,{rt::Device::Type::kCpu,0});c->module->load_parameter(*c->owner,name,x);c->loaded.insert(name);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_partial_finish_r7003(void*raw){auto*c=static_cast<Context*>(raw);try{for(auto&name:c->module->state_dict_keys())if(!c->loaded.count(name))throw std::runtime_error("Missing partial weight "+name);c->module->finish(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_partial_reset_r7003(void*raw){auto*c=static_cast<Context*>(raw);try{c->module->reset(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_partial_forward_r7003(void*raw,size_t tokens,size_t count,const size_t*req,const void*x,const void*ids,const void*positions,void*proof,void*stream){auto*c=static_cast<Context*>(raw);try{if(!tokens||tokens>4096||!count||count>32||!req)throw std::invalid_argument("Partial input count differs");if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("Partial producer sync failed");c->module->forward(*c->owner,tokens,count,req,x,ids,positions,proof);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_partial_destroy_r7003(void*raw){auto*c=static_cast<Context*>(raw);drain(c);c->module.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)return 1;delete c;return 0;}


extern "C" const char*deep_partial_stages_r7003(void*raw){auto*c=static_cast<Context*>(raw);try{nlohmann::json a=nlohmann::json::array();for(auto&v:c->module->stages){auto&t=v.second;if(!t||!t->is_contiguous())throw std::runtime_error("Stage tensor not contiguous");a.push_back({{"name",v.first},{"shape",t->shape()},{"dtype",t->dtype()==rt::DataType::kBFloat16?"BF16":t->dtype()==rt::DataType::kFloat32?"F32":t->dtype()==rt::DataType::kInt64?"I64":t->dtype()==rt::DataType::kInt32?"I32":"unsupported"},{"bytes",t->nbytes()}});}serialized=a.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_partial_read_r7003(void*raw,size_t index,void*host,size_t bytes){auto*c=static_cast<Context*>(raw);try{if(index>=c->module->stages.size())throw std::out_of_range("Stage index");auto t=c->module->stages.at(index).second;if(bytes!=t->nbytes())throw std::invalid_argument("Stage byte count");c->owner->syncStream();c->owner->memcpyD2H(host,t->data(),bytes);c->owner->syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
