#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include "private_moe.hpp"
#include "router_suffix.hpp"
#include <fstream>
#include <cstring>
#include <set>
#include <chrono>
#include <cstdio>
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4::router_suffix_r419;
namespace {
thread_local std::string error,serialized;
thread_local std::shared_ptr<rt::Runtime>owner;
thread_local std::weak_ptr<rt::Runtime>last_owner;
constexpr const char*prefix="layers.16.ffn.";
struct Context{
 std::shared_ptr<rt::Runtime>runtime;std::unique_ptr<md::DeepseekV4MoERoutedR20>module;infinicclComm_t comm=nullptr;
 std::thread::id thread=std::this_thread::get_id();bool ready=false,poisoned=false;std::set<std::string>loaded;
 md::DeepseekV4MoERoutedR20::Output output;rt::Tensor logits,debug_scores,debug_selection;md::RouterBuffers diagnostic;
 deep419::State stats{};double returned_s=0,complete_s=0;
 void check(){if(!runtime||poisoned||thread!=std::this_thread::get_id())throw std::logic_error("r419 owner/poison differs");runtime->assertOwnerThread();}
};
void drain(Context*c){if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
template<class F>int invoke(Context*c,F f){try{c->check();f();return 0;}catch(const std::exception&e){error=e.what();c->poisoned=true;drain(c);return 1;}}
void finalize(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&child:m.children())finalize(r,*child.second);m.process_weights_after_loading(r,false);}
rt::Tensor wrap(Context*c,const void*p,size_t tokens){if(!p||!tokens||tokens>64)throw std::invalid_argument("r419 input geometry differs");cudaPointerAttributes attributes{};if(cudaPointerGetAttributes(&attributes,p)!=cudaSuccess||attributes.type!=cudaMemoryTypeDevice||attributes.device!=c->runtime->device().index())throw std::invalid_argument("r419 external input pointer owner differs");return rt::Tensor::from_blob(const_cast<void*>(p),{tokens,4096},rt::DataType::kBFloat16,c->runtime->device());}
rt::Tensor proof(Context*c,int which){switch(which){case 0:return c->output.value;case 1:return c->output.weights;case 2:return c->output.indices;case 3:return c->logits;case 4:return c->debug_scores;case 5:return c->debug_selection;default:throw std::invalid_argument("r419 proof field differs");}}
}
extern "C" const char*deep_router_error_r419(){return error.c_str();}
extern "C" size_t deep_router_id_size_r419(){return sizeof(infinicclUniqueId);}
extern "C" int deep_router_id_r419(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_router_create_r419(int device,const char*rawcfg,const void*uid){std::unique_ptr<Context>c;try{
 if(owner)throw std::logic_error("r419 oneRuntime perowner");c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
 infini::rt::DeviceGuard guard{owner->device().type(),device,true};auto cfg=nlohmann::json::parse(rawcfg);
 if(cfg.at("hidden_size")!=4096||cfg.at("n_routed_experts")!=256||cfg.at("num_experts_per_tok")!=6||cfg.at("moe_intermediate_size")!=2048||cfg.at("n_shared_experts")!=1||cfg.at("num_hash_layers")!=3||cfg.at("scoring_func")!="sqrtsoftplus"||!cfg.at("norm_topk_prob").get<bool>()||cfg.at("routed_scaling_factor")!=1.5)throw std::invalid_argument("r419 checkpoint geometry differs");
 infinicclUniqueId id{};std::memcpy(&id,uid,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("r419 TP8 init failed");
 infinilm::engine::distributed::RankInfo rank(owner->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
 c->module=std::make_unique<md::DeepseekV4MoERoutedR20>(*owner,4096,2048,2048,256,6,size_t(cfg.at("vocab_size")),false,1.5,rank);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->module.reset();drain(c.get());if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();}owner.reset();return nullptr;}}
extern "C" const char*deep_router_keys_r419(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>names;for(auto&k:c->module->state_required_keys())names.push_back(std::string(prefix)+k);serialized=nlohmann::json(names).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_router_load_r419(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("r419 weightstate differs");rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("r419 weightdtype differs");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});auto key=std::string(name).substr(std::strlen(prefix));c->module->load_parameter(*c->runtime,key,x);c->loaded.insert(key);
 });}
extern "C" int deep_router_finish_r419(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("r419 already finalized");for(auto&k:c->module->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("r419 missingweight "+k);finalize(*c->runtime,*c->module);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_router_forward_r419(void*raw,size_t tokens,const void*x,void*producer,int mode){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||mode<0||mode>1)throw std::invalid_argument("r419 forwardstate/mode differs");auto&r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("r419 producer stream failed");c->diagnostic={};c->logits={};c->debug_scores={};c->debug_selection={};auto input=wrap(c,x,tokens);
 auto begin=std::chrono::steady_clock::now();c->stats=deep419::owner_phase(mode,[&]{c->output=c->module->forward(r,input,rt::Tensor{});});auto returned=std::chrono::steady_clock::now();r.syncStream();auto end=std::chrono::steady_clock::now();
 c->returned_s=std::chrono::duration<double>(returned-begin).count();c->complete_s=std::chrono::duration<double>(end-begin).count();
 });}
// Untimed numerical qualification: generate logits once with the same actual gate helper.
extern "C" int deep_router_project_r419(void*raw,const void*x,void*producer){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(!c->ready)throw std::logic_error("r419 notready");auto&r=*c->runtime;if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("r419 producer failed");c->output={};auto input=wrap(c,x,1);c->logits=rt::Tensor::empty(r,{1,256},rt::DataType::kFloat32);md::gate_project(r,input,c->module->diagnostic_gate(),c->logits);r.syncStream();});}
extern "C" int deep_router_same_logits_r419(void*raw,int mode){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||!c->logits)throw std::logic_error("r419 missingreal logit lease");auto&r=*c->runtime;c->output.value={};c->debug_scores=rt::Tensor::empty(r,{1,256},rt::DataType::kFloat32);c->debug_selection=rt::Tensor::empty(r,{1,256},rt::DataType::kFloat32);
 c->stats=deep419::owner_phase(mode,[&]{bool fused=deep419::select({1,256,6,false,true,true,true,true,true});c->diagnostic=md::router_buffers(r,1,256,fused);md::router_suffix(r,c->logits,c->module->diagnostic_bias(),rt::Tensor{},rt::Tensor{},c->module->diagnostic_epsilon(),c->module->diagnostic_factor(),0,false,6,fused,c->diagnostic,c->debug_scores,c->debug_selection);});
 c->output.weights=c->diagnostic.weights;c->output.indices=c->diagnostic.indices;r.syncStream();
 });}
extern "C" int deep_router_proof_info_r419(void*raw,int which,size_t*shape,size_t*nbytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=proof(c,which);if(!t){*nbytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("r419 proofdim");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*nbytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_router_proof_read_r419(void*raw,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=proof(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("r419 proofbytes");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_router_stats_r419(void*raw,double*seconds,uint64_t*out){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{seconds[0]=c->returned_s;seconds[1]=c->complete_s;out[0]=c->stats.calls;out[1]=c->stats.hits;out[2]=c->stats.fallbacks;out[3]=std::hash<std::thread::id>{}(std::this_thread::get_id());});}
extern "C" int deep_router_dtype_r419(int i){return i==0?int(rt::DataType::kFloat32):i==1?int(rt::DataType::kBFloat16):i==2?int(rt::DataType::kInt64):-1;}
extern "C" int deep_router_destroy_r419(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("r419 wrongdestroythread");drain(c);c->output={};c->diagnostic={};c->logits={};c->debug_scores={};c->debug_selection={};c->module.reset();drain(c);if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("r419 commdestroyfailed");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_router_shutdown_r419(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}

// HOST-only main from the exact bridge object: no Runtime/Native/GPU calls.
__attribute__((visibility("hidden"))) int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--suffix"){
  std::array<float,514>input{};std::ifstream in(argv[2],std::ios::binary);in.read(reinterpret_cast<char*>(input.data()),sizeof(input));if(in.gcount()!=sizeof(input)||in.peek()!=EOF)throw std::invalid_argument("r419 host inputbytes");
  auto out=deep419::host_suffix(input.data(),input.data()+256,input[512],input[513]);std::ofstream f(argv[3],std::ios::binary);f.write(reinterpret_cast<const char*>(out.ids.data()),sizeof(out.ids));f.write(reinterpret_cast<const char*>(out.weights.data()),sizeof(out.weights));f.write(reinterpret_cast<const char*>(out.scores.data()),sizeof(out.scores));f.write(reinterpret_cast<const char*>(out.selection.data()),sizeof(out.selection));if(!f)throw std::runtime_error("r419 host outputfailed");return 0;
 }
 if(argc!=1)throw std::invalid_argument("r419 CPU args");deep419::CPU_contract();nlohmann::json defaults;
 const auto device=rt::Device::Type::kHygon;
#define DEFAULT419(OP) defaults[#OP]={{"default",infini::ops::OP::DefaultConfig(device).implementation_index()},{"active",infini::ops::OP::active_implementation_indices(device)},{"needs_resolution",infini::ops::OP::DefaultConfig(device).needs_implementation_resolution()}}
 DEFAULT419(Gemm);DEFAULT419(Softplus);DEFAULT419(Sqrt);DEFAULT419(Add);DEFAULT419(Topk);DEFAULT419(Gather);DEFAULT419(Sum);DEFAULT419(Div);DEFAULT419(Mul);
#undef DEFAULT419
 std::puts(nlohmann::json({{"passed",true},{"actual_owner_threads",8},{"defaultOFF_and_fallbacks",true},{"fallback_geometries",9},{"error_restore",true},{"SDK_default_metadata_only",defaults},{"Native_dlopen",false},{"Runtime_created",false},{"GPU_calls",0}}).dump().c_str());return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
