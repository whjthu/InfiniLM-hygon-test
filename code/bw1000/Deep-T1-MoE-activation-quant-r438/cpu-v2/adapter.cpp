#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include "private_moe.hpp"
#include "activation_quant.hpp"
#include <fstream>
#include <cstring>
#include <set>
#include <chrono>
#include <cstdio>
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4::actquant_r438;
namespace {
thread_local std::string error,serialized;
thread_local std::shared_ptr<rt::Runtime>owner;
thread_local std::weak_ptr<rt::Runtime>last_owner;
constexpr const char*prefix="layers.16.ffn.";
struct Context{
 std::shared_ptr<rt::Runtime>runtime;std::unique_ptr<md::DeepseekV4MoERoutedR20>module;infinicclComm_t comm=nullptr;
 std::thread::id thread=std::this_thread::get_id();bool ready=false,poisoned=false;std::set<std::string>loaded;
 md::DeepseekV4MoERoutedR20::Output output;deep438::Diagnostic diagnostic;
 deep438::State stats{};double returned_s=0,complete_s=0;
 void check(){if(!runtime||poisoned||thread!=std::this_thread::get_id())throw std::logic_error("r438 owner/poison differs");runtime->assertOwnerThread();}
};
void drain(Context*c){if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
template<class F>int invoke(Context*c,F f){try{c->check();f();return 0;}catch(const std::exception&e){error=e.what();c->poisoned=true;drain(c);return 1;}}
void finalize(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&child:m.children())finalize(r,*child.second);m.process_weights_after_loading(r,false);}
rt::Tensor wrap(Context*c,const void*p,size_t tokens){if(!p||!tokens||tokens>64)throw std::invalid_argument("r438 input geometry differs");cudaPointerAttributes attributes{};if(cudaPointerGetAttributes(&attributes,p)!=cudaSuccess||attributes.type!=cudaMemoryTypeDevice||attributes.device!=c->runtime->device().index())throw std::invalid_argument("r438 external input pointer owner differs");return rt::Tensor::from_blob(const_cast<void*>(p),{tokens,4096},rt::DataType::kBFloat16,c->runtime->device());}
rt::Tensor proof(Context*c,int which){
 if(which==0)return c->output.value;if(which==1)return c->output.weights;if(which==2)return c->output.indices;
 if(which<3||which>=17)throw std::invalid_argument("r438 proof field differs");
 auto&v=c->diagnostic.roles[(which-3)/7];switch((which-3)%7){case 0:return v.gu;case 1:return v.activation;case 2:return v.packed;case 3:return v.scale;case 4:return v.candidate_activation;case 5:return v.candidate_packed;case 6:return v.candidate_scale;}return {};
}
}
extern "C" const char*deep_actquant_error_r438(){return error.c_str();}
extern "C" size_t deep_actquant_id_size_r438(){return sizeof(infinicclUniqueId);}
extern "C" int deep_actquant_id_r438(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_actquant_create_r438(int device,const char*rawcfg,const void*uid){std::unique_ptr<Context>c;try{
 if(owner)throw std::logic_error("r438 oneRuntime perowner");c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
 infini::rt::DeviceGuard guard{owner->device().type(),device,true};auto cfg=nlohmann::json::parse(rawcfg);
 if(cfg.at("hidden_size")!=4096||cfg.at("n_routed_experts")!=256||cfg.at("num_experts_per_tok")!=6||cfg.at("moe_intermediate_size")!=2048||cfg.at("n_shared_experts")!=1||cfg.at("num_hash_layers")!=3||cfg.at("scoring_func")!="sqrtsoftplus"||!cfg.at("norm_topk_prob").get<bool>()||cfg.at("routed_scaling_factor")!=1.5)throw std::invalid_argument("r438 checkpoint geometry differs");
 infinicclUniqueId id{};std::memcpy(&id,uid,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("r438 TP8 init failed");
 infinilm::engine::distributed::RankInfo rank(owner->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
 c->module=std::make_unique<md::DeepseekV4MoERoutedR20>(*owner,4096,2048,2048,256,6,size_t(cfg.at("vocab_size")),false,1.5,rank);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->module.reset();drain(c.get());if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();}owner.reset();return nullptr;}}
extern "C" const char*deep_actquant_keys_r438(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>names;for(auto&k:c->module->state_required_keys())names.push_back(std::string(prefix)+k);serialized=nlohmann::json(names).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_actquant_load_r438(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("r438 weightstate differs");rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("r438 weightdtype differs");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});auto key=std::string(name).substr(std::strlen(prefix));c->module->load_parameter(*c->runtime,key,x);c->loaded.insert(key);
 });}
extern "C" int deep_actquant_finish_r438(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("r438 already finalized");for(auto&k:c->module->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("r438 missingweight "+k);finalize(*c->runtime,*c->module);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_actquant_forward_r438(void*raw,size_t tokens,const void*x,void*producer,int mode,int capture){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||mode<0||mode>1||capture<0||capture>1)throw std::invalid_argument("r438 forwardstate/mode differs");auto&r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("r438 producer stream failed");c->diagnostic={};auto input=wrap(c,x,tokens);
 auto begin=std::chrono::steady_clock::now();c->stats=deep438::owner_phase(mode,tokens==1,capture?&c->diagnostic:nullptr,[&]{c->output=c->module->forward(r,input,rt::Tensor{});});auto returned=std::chrono::steady_clock::now();r.syncStream();auto end=std::chrono::steady_clock::now();
 c->returned_s=std::chrono::duration<double>(returned-begin).count();c->complete_s=std::chrono::duration<double>(end-begin).count();
 });}
// Untimed same-GU diagnostic: original SDK outputs retained by the OFF call,
// candidate takes those exact immutable Tensor leases/stream/owner. Poisons
// below exist only in this diagnostic; no new memset/fence in ordinary forward.
extern "C" int deep_actquant_compare_gu_r438(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready)throw std::logic_error("r438 notready");auto&r=*c->runtime;
 for(int role=0;role<2;++role){auto&v=c->diagnostic.roles[role];if(!v.gu||!v.activation||!v.packed||!v.scale)throw std::logic_error("r438 missing original OFF same-GU lease");
  if(v.gu->shape()!=rt::Shape{role==0?size_t(6):size_t(1),size_t(512)})throw std::logic_error("r438 diagnostic geometry differs");
  v.candidate_packed=rt::Tensor::empty(r,v.packed->shape(),rt::DataType::kInt8);v.candidate_scale=rt::Tensor::empty(r,v.scale->shape(),rt::DataType::kFloat32);v.candidate_activation=rt::Tensor::empty(r,v.activation->shape(),rt::DataType::kBFloat16);
  r.setDeviceMemory(deep438::write_data(v.candidate_packed),0xa5,v.candidate_packed->nbytes());r.setDeviceMemory(deep438::write_data(v.candidate_scale),0xff,v.candidate_scale->nbytes());r.setDeviceMemory(deep438::write_data(v.candidate_activation),0xff,v.candidate_activation->nbytes());
  deep438::apply(r,v.gu,v.candidate_packed,v.candidate_scale,v.candidate_activation);
 }r.syncStream();
 });}
extern "C" int deep_actquant_proof_info_r438(void*raw,int which,size_t*shape,size_t*nbytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=proof(c,which);if(!t){*nbytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("r438 proofdim");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*nbytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_actquant_proof_read_r438(void*raw,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=proof(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("r438 proofbytes");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_actquant_stats_r438(void*raw,double*seconds,uint64_t*out){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{seconds[0]=c->returned_s;seconds[1]=c->complete_s;for(int role=0;role<2;++role){out[role*3]=c->stats.calls[role];out[role*3+1]=c->stats.hits[role];out[role*3+2]=c->stats.fallbacks[role];}out[6]=std::hash<std::thread::id>{}(std::this_thread::get_id());});}
extern "C" int deep_actquant_dtype_r438(int i){return i==0?int(rt::DataType::kFloat32):i==1?int(rt::DataType::kBFloat16):i==2?int(rt::DataType::kInt64):i==3?int(rt::DataType::kInt8):-1;}
extern "C" int deep_actquant_destroy_r438(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("r438 wrongdestroythread");drain(c);c->output={};c->diagnostic={};c->module.reset();drain(c);if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("r438 commdestroyfailed");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_actquant_shutdown_r438(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
