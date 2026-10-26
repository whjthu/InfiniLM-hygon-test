#include "deepseek_v4_csa_attention.hpp"
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
 std::unique_ptr<model::CsaAttention>swa;
 Partial(rt::Runtime&r,const nlohmann::json&cfg,const infinilm::engine::distributed::RankInfo&rank){
  auto bind=[&](const std::string&prefix,infinilm::weights::ModelWeights&m){for(auto&v:m.state_dict())bind_weight(prefix+v.first,v.second);};
  auto bounded=cfg;bounded["native_compressed_capacity"]=1024;bounded["max_position_embeddings"]=4096;swa=std::make_unique<model::CsaAttention>(r,bounded,16,rank,32);bind("layers.16.attn.",*swa);
 }
 void finish(rt::Runtime&r){swa->process_weights_after_loading(r,false);r.syncStream();}
 void reset(rt::Runtime&r){swa->reset(r);r.syncStream();}
 void forward(rt::Runtime&r,size_t tokens,size_t count,const size_t*req,const void*x,const void*ids,const void*positions,void*proof){
  auto view=[&](const void*p,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());};
  std::vector<model::CsaAttention::Request>b;
  for(size_t i=0;i<count;++i){b.push_back({req[3*i],req[3*i+1],req[3*i+2]});}
  auto tp=view(positions,{tokens},rt::DataType::kInt64);
  model::wide_r3443::Scope scope(r,count);
  auto normalized=view(x,{tokens,4096},rt::DataType::kBFloat16);
  auto result=swa->forward(r,normalized,tp,b);
  r.syncStream();if(proof){size_t offset=0;for(auto t:{result}){rt::Tensor::from_blob(static_cast<char*>(proof)+offset,t->shape(),t->dtype(),r.device())->copy_from(r,t);offset+=t->nbytes();}r.syncStream();}

 }
};
struct Context{std::shared_ptr<rt::Runtime>owner;infinicclComm_t comm=nullptr;std::unique_ptr<Partial>module;std::set<std::string>loaded;};
void drain(Context*c){if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}}
std::map<int,std::shared_ptr<rt::Runtime>>&runtime_owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>v;return v;}
std::shared_ptr<rt::Runtime>worker(int d){auto&r=runtime_owners()[d];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();}return r;}
}
extern "C" const char*deep_csa_fixture_error_r9242(){return error.c_str();}
extern "C" size_t deep_csa_fixture_id_size_r9242(){return sizeof(infinicclUniqueId);}
extern "C" int deep_csa_fixture_id_r9242(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_csa_fixture_create_r9242(int device,const char*cfg,const void*raw_id){std::unique_ptr<Context>c;try{c=std::make_unique<Context>();c->owner=worker(device);auto&r=*c->owner;infini::rt::DeviceGuard guard{r.device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("Partial TP8 init failed");infinilm::engine::distributed::RankInfo rank(r.device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;c->module=std::make_unique<Partial>(r,nlohmann::json::parse(cfg),rank);return c.release();}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" const char*deep_csa_fixture_keys_r9242(void*raw){serialized=nlohmann::json(static_cast<Context*>(raw)->module->state_required_keys()).dump();return serialized.c_str();}
extern "C" int deep_csa_fixture_load_r9242(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);try{auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dtype==0?rt::DataType::kInt8:dtype==1?rt::DataType::kFloat32:dtype==2?rt::DataType::kBFloat16:rt::DataType::kInt64,{rt::Device::Type::kCpu,0});c->module->load_parameter(*c->owner,name,x);c->loaded.insert(name);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_csa_fixture_finish_r9242(void*raw){auto*c=static_cast<Context*>(raw);try{for(auto&name:c->module->state_required_keys())if(!c->loaded.count(name))throw std::runtime_error("Missing partial weight "+name);c->module->finish(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_csa_fixture_reset_r9242(void*raw){auto*c=static_cast<Context*>(raw);try{c->module->reset(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_csa_fixture_forward_r9242(void*raw,size_t tokens,size_t count,const size_t*req,const void*x,const void*ids,const void*positions,void*proof,void*stream){auto*c=static_cast<Context*>(raw);try{if(!tokens||tokens>4096||!count||count>32||!req)throw std::invalid_argument("Partial input count differs");if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("Partial producer sync failed");c->module->forward(*c->owner,tokens,count,req,x,ids,positions,proof);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int deep_csa_fixture_destroy_r9242(void*raw){auto*c=static_cast<Context*>(raw);drain(c);c->module.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)return 1;delete c;return 0;}


extern "C" int deep_csa_fixture_shutdown_r9242(){try{for(auto&v:runtime_owners()){infini::rt::DeviceGuard g{rt::Device::Type::kHygon,v.first,true};v.second->syncStream();v.second->trimMemory();v.second.reset();}runtime_owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
