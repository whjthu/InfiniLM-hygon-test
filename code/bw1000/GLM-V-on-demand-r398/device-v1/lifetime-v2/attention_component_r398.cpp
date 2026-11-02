// Private checkpoint attention consumer; no FFN, indexer or Model activation.
// Selection is an explicit synthetic geometry input, not recorded Model routing.
#include "selected_values_r398.hpp"
#include "glm_moe_dsa_attention_output.hpp"
#include "glm_selection_map_r302.hpp"
#include "glm_moe_dsa_sparse_attention.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <set>

namespace rt=infinilm::runtime;
namespace m=infinilm::models::glm_moe_dsa;
namespace {
thread_local std::string err398,report398;
using Clock=std::chrono::steady_clock;
void ck(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
struct Context {
 std::shared_ptr<rt::Runtime> r;std::unique_ptr<m::AttentionPrepare> prep;
 std::unique_ptr<m::AttentionOutput> out;infinicclComm_t comm=nullptr;
 rt::Tensor latent,rotary,key,value;std::set<std::string>loaded;
 size_t seeded=0;int device=-1;bool ready=false,poison=false,closing=false;
};
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||!c->r||c->closing)throw std::invalid_argument("r398 context");c->r->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device||c->r->isGraphRecording())throw std::invalid_argument("r398 owner/device/eager");}
void input(Context*c,const void*p){if(!p)throw std::invalid_argument("r398 null");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("r398 input device");}
struct Work {rt::Runtime&r;std::vector<rt::Tensor>held;m::SelectionMapR302 host_selection;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType t){auto x=rt::Tensor::empty(r,s,t);held.push_back(x);return x;}};
rt::Tensor blob(Context*c,const void*p,rt::Shape shape,rt::DataType t){input(c,p);return rt::Tensor::from_blob(const_cast<void*>(p),shape,t,c->r->device());}
std::vector<std::string>keys(Context*c){auto a=c->prep->state_dict_keys(),b=c->out->state_dict_keys();a.insert(a.end(),b.begin(),b.end());std::set<std::string>s(a.begin(),a.end());if(s.size()!=a.size())throw std::logic_error("r398 duplicate weights");return a;}
void clear_cache(Context*c){drain(c);c->latent.reset();c->rotary.reset();c->key.reset();c->value.reset();c->seeded=0;}
}
namespace infinilm::models::glm_moe_dsa {
runtime::Tensor sparse_attention_all_selected_decode_r601(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,float);
}
extern "C" const char*glm_attention_error_r398(){return err398.c_str();}
extern "C" const char*glm_attention_report_r398(){return report398.c_str();}
extern "C" void*glm_attention_create_r398(int device,const char*config,const void*uid){
 std::unique_ptr<Context>c;
 try {
  if(device<0||device>=8||!config||!uid)throw std::invalid_argument("r398 create");
  auto cfg=nlohmann::json::parse(config);if(cfg.at("max_position_embeddings")<32768)throw std::invalid_argument("r398 full context");
  c=std::make_unique<Context>();c->device=device;infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};
  c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
  infinicclUniqueId id{};std::memcpy(&id,uid,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("r398 TP8");
  infinilm::engine::distributed::RankInfo rank(c->r->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
  c->prep=std::make_unique<m::AttentionPrepare>(*c->r,cfg,rank);c->out=std::make_unique<m::AttentionOutput>(*c->r,cfg,rank);return c.release();
 }catch(const std::exception&e){err398=e.what();if(c){drain(c.get());c->prep.reset();c->out.reset();if(c->comm)infinicclCommDestroy(c->comm);}return nullptr;}
}
extern "C" const char*glm_attention_keys_r398(void*raw){try{auto*c=static_cast<Context*>(raw);owner(c);report398=nlohmann::json(keys(c)).dump();return report398.c_str();}catch(const std::exception&e){err398=e.what();return nullptr;}}
extern "C" int glm_attention_load_r398(void*raw,const char*json){auto*c=static_cast<Context*>(raw);bool own=false;try{
 owner(c);own=true;if(c->ready||c->poison||!json)throw std::invalid_argument("r398 load state");
 std::unordered_map<std::string,rt::Tensor>a,b;std::set<std::string>pk,ok,seen;for(auto&k:c->prep->state_dict_keys())pk.insert(k);for(auto&k:c->out->state_dict_keys())ok.insert(k);
 for(auto&v:nlohmann::json::parse(json)){std::string name=v.at("name");int d=v.at("dtype");auto p=reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>());if(!p||d<0||d>2||c->loaded.count(name)||!seen.insert(name).second)throw std::invalid_argument("r398 weight descriptor");auto x=rt::Tensor::from_blob(p,v.at("shape").get<rt::Shape>(),d==0?rt::DataType::kFloat32:d==1?rt::DataType::kBFloat16:rt::DataType::kUInt8,{rt::Device::Type::kCpu,0});if(pk.count(name))a.emplace(name,x);else if(ok.count(name))b.emplace(name,x);else throw std::invalid_argument("r398 unknown weight");}
 if(!a.empty())c->prep->load_parameters_no_sync(*c->r,a,true);if(!b.empty())c->out->load_parameters_no_sync(*c->r,b,true);c->r->syncStream();for(auto&v:a)c->loaded.insert(v.first);for(auto&v:b)c->loaded.insert(v.first);return 0;
 }catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_attention_finish_r398(void*raw){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;if(c->ready||c->poison)throw std::invalid_argument("r398 finish state");for(auto&k:keys(c))if(!c->loaded.count(k))throw std::invalid_argument("r398 missing weight");c->prep->process_weights_after_loading(*c->r,false);c->out->process_weights_after_loading(*c->r,false);c->r->syncStream();c->ready=true;return 0;}catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_attention_begin_r398(void*raw,int mode){auto*c=static_cast<Context*>(raw);bool own=false;try{
 owner(c);own=true;if(!c->ready||c->poison||(mode!=0&&mode!=1))throw std::invalid_argument("r398 begin");clear_cache(c);
 c->r->trimMemory();c->latent=rt::Tensor::empty(*c->r,{32768,512},rt::DataType::kBFloat16);c->rotary=rt::Tensor::empty(*c->r,{32768,64},rt::DataType::kBFloat16);c->key=rt::Tensor::empty(*c->r,{32768,8,256},rt::DataType::kBFloat16);if(!mode)c->value=rt::Tensor::empty(*c->r,{32768,8,256},rt::DataType::kBFloat16);c->r->syncStream();return 0;
 }catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_attention_seed_r398(void*raw,size_t past,size_t count,const void*hidden,const void*position){auto*c=static_cast<Context*>(raw);bool own=false;try{
 owner(c);own=true;if(!c->ready||c->poison||!c->key||past!=c->seeded||!count||count>512||past>32768||count>32768-past)throw std::invalid_argument("r398 seed boundary");
 Work w{*c->r};auto x=blob(c,hidden,{count,6144},rt::DataType::kBFloat16),p=blob(c,position,{count},rt::DataType::kInt64);std::vector<int64_t>cpu(count);c->r->memcpyD2H(cpu.data(),position,count*8);for(size_t j=0;j<count;++j)if(cpu[j]!=int64_t(past+j))throw std::invalid_argument("r398 seed absolute position");
 auto a=c->prep->forward_latent(*c->r,x,p);w.held.insert(w.held.end(),{a.query,a.q_resid,a.kv_latent,a.k_rot});auto kv=c->prep->expand_cached(*c->r,a.kv_latent,a.k_rot);w.held.insert(w.held.end(),{kv.first,kv.second});
 c->latent->narrow({{0,past,count}})->copy_from(*c->r,a.kv_latent);c->rotary->narrow({{0,past,count}})->copy_from(*c->r,a.k_rot->view({count,64}));c->key->narrow({{0,past,count}})->copy_from(*c->r,kv.first);if(c->value)c->value->narrow({{0,past,count}})->copy_from(*c->r,kv.second);c->r->syncStream();c->seeded=past+count;return 0;
 }catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
// Immutable-prefix component repetition, not a production append lifecycle.
// Each timed query recomputes original fresh Q/KV, writes the same final rows,
// gathers selected K and cached/recomputed V, executes original attention+O+TP.
extern "C" int glm_attention_query_r398(void*raw,size_t past,size_t count,const void*hidden,const void*position,const void*selected,void*out,void*attention_out){auto*c=static_cast<Context*>(raw);bool own=false;try{
 owner(c);own=true;if(!c->ready||c->poison||!c->key||(count!=1&&count!=512)||past>32768||count>32768-past||past+count!=c->seeded)throw std::invalid_argument("r398 query boundary");
 auto start=Clock::now();size_t free0=0,total=0;ck(cudaMemGetInfo(&free0,&total));size_t U=0,free1=0,free2=0;
 {
 Work w{*c->r};auto x=blob(c,hidden,{count,6144},rt::DataType::kBFloat16),p=blob(c,position,{count},rt::DataType::kInt64),sel=blob(c,selected,{count,2048},rt::DataType::kInt64),y=blob(c,out,{count,6144},rt::DataType::kBFloat16),z=blob(c,attention_out,{count,8,256},rt::DataType::kBFloat16);
 std::vector<int64_t>positions(count),host(count*2048);c->r->memcpyD2H(positions.data(),position,count*8);c->r->memcpyD2H(host.data(),selected,host.size()*8);
 for(size_t i=0;i<count;++i){if(positions[i]!=int64_t(past+i))throw std::invalid_argument("r398 query position");bool causal=false;for(size_t j=0;j<2048;++j){auto k=host[i*2048+j];if(k<0||size_t(k)>=c->seeded)throw std::invalid_argument("r398 selected prefix");causal|=size_t(k)<=past+i;}if(!causal)throw std::invalid_argument("r398 no causal key");}
 auto&remapped=w.host_selection;remapped=m::selection_map_r302(host,count,2048,past);U=remapped.logical.size();auto ids=w.alloc({U},rt::DataType::kInt64);c->r->memcpyH2D(ids->data(),remapped.logical.data(),U*8,false);
 auto a=c->prep->forward_latent(*c->r,x,p);w.held.insert(w.held.end(),{a.query,a.q_resid,a.kv_latent,a.k_rot});auto fresh=c->prep->expand_cached(*c->r,a.kv_latent,a.k_rot);w.held.insert(w.held.end(),{fresh.first,fresh.second});
 c->latent->narrow({{0,past,count}})->copy_from(*c->r,a.kv_latent);c->rotary->narrow({{0,past,count}})->copy_from(*c->r,a.k_rot->view({count,64}));c->key->narrow({{0,past,count}})->copy_from(*c->r,fresh.first);if(c->value)c->value->narrow({{0,past,count}})->copy_from(*c->r,fresh.second);
 bool complete=U==c->seeded;for(size_t i=0;complete&&i<U;++i)complete=remapped.logical[i]==int64_t(i);
 rt::Tensor k,v;
 if(complete){k=c->key->narrow({{0,0,U}});if(c->value)v=c->value->narrow({{0,0,U}});}
 else{k=w.alloc({U,8,256},rt::DataType::kBFloat16);infinilm::execution::submit<infini::ops::IndexSelect>(*c->r,infini::ops::IndexSelect::DefaultConfig(c->r->device().type()),c->key,ids,int64_t{0},k);if(c->value){v=w.alloc({U,8,256},rt::DataType::kBFloat16);infinilm::execution::submit<infini::ops::IndexSelect>(*c->r,infini::ops::IndexSelect::DefaultConfig(c->r->device().type()),c->value,ids,int64_t{0},v);}}
 if(!c->value)v=m::v_on_demand_r398::selected_values(*c->prep,*c->r,c->latent,c->rotary,ids,w.held);w.held.insert(w.held.end(),{k,v});c->r->syncStream();ck(cudaMemGetInfo(&free1,&total));
 rt::Tensor attn;
 if(count==1)attn=m::sparse_attention_all_selected_decode_r601(*c->r,a.query,k,v,1.f/16);
 else{auto mapped=w.alloc({count,2048},rt::DataType::kInt64);c->r->memcpyH2D(mapped->data(),remapped.mapped.data(),remapped.mapped.size()*8,false);attn=m::sparse_attention_mapped_bounded(*c->r,a.query,k,v,mapped,ids,past,1.f/16);}
 w.held.push_back(attn);auto result=c->out->forward(*c->r,attn);w.held.insert(w.held.end(),{result.local,result.hidden});y->copy_from(*c->r,result.hidden);z->copy_from(*c->r,attn);c->r->syncStream();ck(cudaMemGetInfo(&free2,&total));
 }
 auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();report398=nlohmann::json({{"complete_attention_CAPI_ms",ms},{"mode",!bool(c->value)},{"past",past},{"count",count},{"union_U",U},{"seeded",c->seeded},{"persistent_V_bytes",c->value?c->value->nbytes():0},{"logical_persistent_bytes",c->latent->nbytes()+c->rotary->nbytes()+c->key->nbytes()+(c->value?c->value->nbytes():0)},{"free_entry",free0},{"free_after_selected",free1},{"free_after_output",free2},{"actual_allocator_peak","UNKNOWN; snapshots only"},{"indexer_included",false},{"profile",0}}).dump();return 0;
 }catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_attention_destroy_r398(void*raw,int*expired){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;if(!expired)throw std::invalid_argument("r398 destroy output");drain(c);std::weak_ptr<rt::Runtime>weak=c->r;clear_cache(c);if(modern_glm_fp8_runtime_clear_r809(c->r.get()))throw std::runtime_error("r398 FP8 clear");c->closing=true;c->prep.reset();c->out.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("r398 CCL destroy");c->comm=nullptr;c->r->trimMemory();c->r.reset();*expired=weak.expired();delete c;return *expired?0:1;}catch(const std::exception&e){err398=e.what();if(own)drain(c);return 1;}}

// Diagnostic only, once per geometry outside every benchmark interval. Uses
// the same selected_values producer as production candidate and actual parent
// cached-V baseline. Streamed host metrics avoid retaining giant raw payloads.
#include "bf16_metrics_r398.hpp"
extern "C" int glm_attention_compare_v_r398(void*raw,size_t past,size_t count,const void*selected){
 auto*c=static_cast<Context*>(raw);bool own=false;try{
  owner(c);own=true;if(!c->ready||c->poison||!c->value||(count!=1&&count!=512)||past+count!=c->seeded)throw std::invalid_argument("r398 V diagnostic requires seeded OFF cache");
  auto start=Clock::now();r398_metrics::Accumulator metrics;size_t U=0;
  {
   Work w{*c->r};auto sel=blob(c,selected,{count,2048},rt::DataType::kInt64);std::vector<int64_t>host(count*2048);c->r->memcpyD2H(host.data(),selected,host.size()*8);
   auto&map=w.host_selection;map=m::selection_map_r302(host,count,2048,past);U=map.logical.size();auto ids=w.alloc({U},rt::DataType::kInt64);c->r->memcpyH2D(ids->data(),map.logical.data(),U*8,false);
   auto cached=w.alloc({U,8,256},rt::DataType::kBFloat16);infinilm::execution::submit<infini::ops::IndexSelect>(*c->r,infini::ops::IndexSelect::DefaultConfig(c->r->device().type()),c->value,ids,int64_t{0},cached);
   auto computed=m::v_on_demand_r398::selected_values(*c->prep,*c->r,c->latent,c->rotary,ids,w.held);c->r->syncStream();
   const size_t chunk=524288;std::vector<uint16_t>a(chunk),b(chunk);
   for(size_t off=0;off<cached->numel();off+=chunk){size_t n=std::min(chunk,cached->numel()-off);c->r->memcpyD2H(a.data(),cached->data()+off*2,n*2);c->r->memcpyD2H(b.data(),computed->data()+off*2,n*2);c->r->syncStream();metrics.add(a.data(),b.data(),n);}
  }
  report398=nlohmann::json({{"diagnostic_only",true},{"timed_query",false},{"past",past},{"count",count},{"union_U",U},{"elements",metrics.elements},{"finite",metrics.finite},{"NRMSE",metrics.NRMSE()},{"reference_RMS",metrics.reference_RMS()},{"maxabs",metrics.maxabs},{"different_BF16_values_diagnostic_only",metrics.different},{"source","original cachedV vs unchanged expand_cached(selectedlatent)"},{"D2H_bytes",metrics.elements*4},{"host_metric_buffer_bytes",2097152},{"diagnostic_wall_s",std::chrono::duration<double>(Clock::now()-start).count()},{"no_raw_values_saved",true}}).dump();return 0;
 }catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}

extern "C" int glm_attention_control_r398(void*raw,void*values){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;Work w{*c->r};auto v=blob(c,values,{1},rt::DataType::kFloat32);w.held.push_back(v);infinilm::engine::distributed::allreduce_(*c->r,v,v,infinicclSum,c->comm);c->r->syncStream();return 0;}catch(const std::exception&e){err398=e.what();if(own){drain(c);c->poison=true;}return 1;}}
