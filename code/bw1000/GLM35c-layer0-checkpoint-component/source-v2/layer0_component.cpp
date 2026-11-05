// Genuine checkpoint Layer0 attention component. No TextModel/Decoder/FFN/LM head.
#include <infini/rt/device_guard.h>
#include "component_scope.hpp"
#include "policy_r403.hpp"
#include "glm_rope_bound_r4279.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "execution/submit.hpp"
#include <cuda_runtime_api.h>
#include <set>
#include <optional>
#include <cmath>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace m=infinilm::models::glm_moe_dsa;namespace lc=m::layer0_component;
namespace infinilm::models::glm_moe_dsa {void glm_indexer_policy_r998(rt::Runtime&,bool);void glm_indexer_policy_clear_r998(rt::Runtime&);}
namespace {
thread_local std::string error,serialized;
void cuda_check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
struct Layer0 final:infinilm::weights::ModelWeights {
 rt::Runtime&r;rt::Tensor embedding,norm,ones;std::unique_ptr<m::NativeAttentionProjectedR22>attention;float eps;
 Layer0(rt::Runtime&owner,const nlohmann::json&cfg,const infinilm::engine::distributed::RankInfo&rank):r(owner),eps(cfg.at("rms_norm_eps")){
  if(cfg.at("vocab_size")!=154880||cfg.at("hidden_size")!=6144||cfg.at("num_hidden_layers")!=78||cfg.at("indexer_types").at(0)!="full"||!std::isfinite(eps)||eps<=0)throw std::invalid_argument("actual GLM checkpoint geometry");
  // Same full BF16 embedding and GPU IndexSelect as the original first stage.
  embedding=rt::Tensor::empty_parameter(r,{154880,6144},rt::DataType::kBFloat16);bind_weight("model.embed_tokens.weight",infinilm::weights::Weight(embedding));
  norm=rt::Tensor::empty_parameter(r,{6144},rt::DataType::kBFloat16);bind_weight("model.layers.0.input_layernorm.weight",infinilm::weights::Weight(norm));
  {lc::Construction scope(r);attention=std::make_unique<m::NativeAttentionProjectedR22>(r,cfg,0,rank,1,32768,2048);}
  for(auto&v:attention->state_dict())bind_weight("model.layers.0.self_attn."+v.first,v.second);
  std::vector<float>host(6144,1.f);auto f=rt::Tensor::empty(r,{6144},rt::DataType::kFloat32);ones=rt::Tensor::empty(r,{6144},rt::DataType::kBFloat16);r.memcpyH2D(f->data(),host.data(),host.size()*4,false);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),f,ones);r.syncStream();
 }
 void process_weights_after_loading(rt::Runtime&owner,bool transpose)override{if(&owner!=&r)throw std::invalid_argument("Layer0 postload owner");attention->process_weights_after_loading(r,transpose);r.syncStream();}
};
struct Context {
 std::shared_ptr<rt::Runtime>r;std::unique_ptr<Layer0>layer;infinicclComm_t tp=nullptr;std::set<std::string>loaded;
 std::unique_ptr<lc::Capture>capture;size_t past=0;int device=-1;bool ready=false,poison=false,closing=false;
};
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||!c->r||c->closing)throw std::invalid_argument("Layer0 component context");c->r->assertOwnerThread();int d=-1;cuda_check(cudaGetDevice(&d));cudaStreamCaptureStatus s;cuda_check(cudaStreamIsCapturing(static_cast<cudaStream_t>(c->r->stream()),&s));if(d!=c->device||c->r->isGraphRecording()||s!=cudaStreamCaptureStatusNone)throw std::invalid_argument("Layer0 component owner/device/eager");}
struct Work {rt::Runtime&r;std::vector<rt::Tensor>held;std::optional<m::NativeAttentionProjectedR22::Output>out;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto t=rt::Tensor::empty(r,s,d);held.push_back(t);return t;}};
void release(Context*c){drain(c);c->capture.reset();if(c->r&&modern_glm_fp8_runtime_clear_r809(c->r.get()))throw std::runtime_error("component FP8 cache clear");if(c->r)m::glm_indexer_policy_clear_r998(*c->r);c->layer.reset();drain(c);if(c->tp&&infinicclCommDestroy(c->tp)!=infinicclSuccess)throw std::runtime_error("component TP destroy");c->tp=nullptr;}
void floor_bytes(size_t minimum){size_t free=0,total=0;cuda_check(cudaMemGetInfo(&free,&total));if(free<minimum)throw std::runtime_error("original GLM memory floor");}
}
extern "C" const char*glm_l0_error(){return error.c_str();}
extern "C" size_t glm_l0_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int glm_l0_id(void*out){if(!out)return 1;infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*glm_l0_create(int enabled,int device,const char*config,const void*uid,int bounded_rope){
 std::unique_ptr<Context>c;try{
  if(enabled!=1||device<0||device>=8||!config||!uid||(bounded_rope!=0&&bounded_rope!=1)||!m::v_on_demand_r403::requested())throw std::invalid_argument("explicit Layer0 diagnostic/NoV required");
  c=std::make_unique<Context>();c->device=device;infini::rt::DeviceGuard g{rt::Device::Type::kHygon,device,true};c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();infinicclUniqueId id{};std::memcpy(&id,uid,sizeof(id));if(infinicclCommInitRank(&c->tp,8,id,device)!=infinicclSuccess)throw std::runtime_error("component genuine TP8 init");
  infinilm::engine::distributed::RankInfo rank(c->r->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->tp;rank.world_comm=c->tp;rank.pp_size=1;rank.pp_stage=0;
  m::rope_bound_r4279::Scope bounded(*c->r,bounded_rope!=0,32768);c->layer=std::make_unique<Layer0>(*c->r,nlohmann::json::parse(config),rank);return c.release();
 }catch(const std::exception&e){error=e.what();if(c){try{release(c.get());}catch(...){std::terminate();}}return nullptr;}}
extern "C" const char*glm_l0_keys(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);serialized=nlohmann::json(c->layer->state_dict_keys()).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_l0_load(void*raw,const char*descriptors){auto*c=static_cast<Context*>(raw);bool owned=false;try{
 owner(c);owned=true;if(c->ready||c->poison||!descriptors)throw std::invalid_argument("component load state");
 struct Sources{rt::Runtime&r;std::unordered_map<std::string,rt::Tensor>map;~Sources(){try{r.syncStream();}catch(...){std::terminate();}}}sources{*c->r};
 for(auto&v:nlohmann::json::parse(descriptors)){auto name=v.at("name").get<std::string>();int d=v.at("dtype");auto p=reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>());if(!p||d<0||d>2||sources.map.count(name)||c->loaded.count(name))throw std::invalid_argument("component checkpoint descriptor");sources.map.emplace(name,rt::Tensor::from_blob(p,v.at("shape").get<rt::Shape>(),d==0?rt::DataType::kFloat32:d==1?rt::DataType::kBFloat16:rt::DataType::kUInt8,{rt::Device::Type::kCpu,0}));}
 c->layer->load_parameters_no_sync(*c->r,sources.map,true);c->r->syncStream();for(auto&v:sources.map)c->loaded.insert(v.first);return 0;
 }catch(const std::exception&e){error=e.what();if(owned){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_l0_finish(void*raw){auto*c=static_cast<Context*>(raw);bool owned=false;try{owner(c);owned=true;if(c->ready||c->poison)throw std::invalid_argument("component finish state");for(auto&k:c->layer->state_dict_keys())if(!c->loaded.count(k))throw std::invalid_argument("missing actual checkpoint key: "+k);c->layer->process_weights_after_loading(*c->r,false);m::glm_indexer_policy_r998(*c->r,false);floor_bytes(4ull*1024*1024*1024+1);c->ready=true;return 0;}catch(const std::exception&e){error=e.what();if(owned){drain(c);c->poison=true;}return 1;}}
// IDs are owned CPU input from the exact frozen above plan, not GPU hidden input.
extern "C" int glm_l0_step(void*raw,size_t past,size_t n,const int64_t*input,int capture){auto*c=static_cast<Context*>(raw);bool owned=false,mutating=false;try{
 owner(c);owned=true;if(!c->ready||c->poison||!input||!n||n>512||past!=c->past||past>32768||n>32768-past||(capture!=0&&capture!=1)||c->capture)throw std::invalid_argument("component step geometry/state");
 if(capture&&(past!=2949||(n!=1&&n!=8)))throw std::invalid_argument("component capture only original first query");
 std::vector<int64_t>ids(input,input+n),positions(n);for(size_t i=0;i<n;++i){if(ids[i]<0||ids[i]>=154880)throw std::invalid_argument("component frozen token range");positions[i]=int64_t(past+i);}
 Work w{*c->r};auto&r=*c->r;auto in=w.alloc({n},rt::DataType::kInt64),pos=w.alloc({n},rt::DataType::kInt64);r.memcpyH2D(in->data(),ids.data(),n*8,false);r.memcpyH2D(pos->data(),positions.data(),n*8,false);
 if(capture){c->capture=std::make_unique<lc::Capture>(r);auto&a=*c->layer->attention;lc::keep("history.latent",a.latent_cache()->narrow({{0,0,1},{1,0,past}})->view({past,512}));lc::keep("history.rotary",a.rotary_cache()->narrow({{0,0,1},{1,0,past}})->view({past,64}));lc::keep("history.indexer",a.index_cache()->narrow({{0,0,1},{1,0,past}})->view({past,128}));lc::keep("history.projected_K",a.projected_key_cache()->narrow({{0,0,1},{1,0,past}})->view({past,8,256}));}
 auto hidden=w.alloc({n,6144},rt::DataType::kBFloat16),normalized=w.alloc({n,6144},rt::DataType::kBFloat16),weighted=w.alloc({n,6144},rt::DataType::kBFloat16);
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),c->layer->embedding,in,int64_t{0},hidden);
 ex::submit<infini::ops::RmsNorm>(r,infini::ops::RmsNorm::DefaultConfig(r.device().type()),hidden,c->layer->ones,c->layer->eps,normalized);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),normalized,c->layer->norm,weighted);
 lc::keep("input.embedding",hidden);lc::keep("input.normalized_attention",weighted);mutating=true;w.out=c->layer->attention->forward(r,weighted,pos,{{0,past,n}});r.syncStream();
 if(capture){auto&o=*w.out;lc::keep("prepared.Q",o.prepared.query);lc::keep("prepared.q_resid",o.prepared.q_resid);lc::keep("prepared.latent",o.prepared.kv_latent);lc::keep("prepared.rotary",o.prepared.k_rot);if(o.selection.size()!=1)throw std::logic_error("component selection rows");lc::keep("selection.IDs",o.selection[0]);lc::keep("attention.NoV_output",o.attention);lc::keep("attention.Oproj_TP_output",o.hidden);c->capture->finish();}
 floor_bytes(512ull*1024*1024);c->past=past+n;return 0;
 }catch(const std::exception&e){error=e.what();if(owned){drain(c);c->capture.reset();if(mutating)c->poison=true;}return 1;}}
extern "C" const char*glm_l0_snapshot(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);if(!c->capture||!c->capture->complete)throw std::invalid_argument("no completed snapshot");auto a=nlohmann::json::array();for(auto&f:c->capture->fields){int type=f.dtype==rt::DataType::kFloat32?0:f.dtype==rt::DataType::kBFloat16?1:f.dtype==rt::DataType::kInt64?3:-1;if(type<0||f.tensor)throw std::logic_error("snapshot CPU-only field");a.push_back({{"name",f.name},{"dtype",type},{"shape",f.shape},{"bytes",f.bytes.size()}});}serialized=nlohmann::json{{"component_only",true},{"layer",0},{"TP",8},{"PP",1},{"world",8},{"past_after",c->past},{"fields",a}}.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_l0_read(void*raw,size_t index,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);try{owner(c);if(!c->capture||!c->capture->complete||!out||index>=c->capture->fields.size()||bytes!=c->capture->fields[index].bytes.size())throw std::invalid_argument("snapshot exact CPU buffer");std::memcpy(out,c->capture->fields[index].bytes.data(),bytes);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_l0_reset(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);drain(c);c->capture.reset();c->layer->attention->reset(*c->r);c->past=0;c->poison=false;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int glm_l0_destroy(void*raw,int*expired){auto*c=static_cast<Context*>(raw);try{owner(c);if(!expired)throw std::invalid_argument("destroy output");std::weak_ptr<rt::Runtime>w=c->r;release(c);c->closing=true;c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
