// Source-only genuine C4 consumer. No fake RankInfo, cache seeding or bypassed compressor.
#include "deepseek_v4_c4_indexer.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <stdexcept>
#include <cstdint>
extern "C" int deep_constant_slot_metadata_set_r482(int);
extern "C" int modern_deep_allselected_set_r813(int);
extern "C" int modern_deep_c4_smallt_mode_r9252(int);
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4;
namespace ds=infinilm::engine::distributed;
namespace {
using JSON=nlohmann::json;
using Clock=std::chrono::steady_clock;
thread_local std::string error,reply;
std::atomic<bool> ever_created{false};
struct Context {
 std::unique_ptr<ds::CommunicationGroup> group;
 std::shared_ptr<rt::Runtime> r;
 std::unique_ptr<md::C4Indexer> index;
 std::unique_ptr<md::C4Indexer::Pending> pending;
 std::thread::id thread=std::this_thread::get_id();
 std::set<std::string> required,loaded;
 size_t past=0,prepared_tokens=0,prepares=0,commits=0,aborts=0,resets=0;
 bool ready=false,poisoned=false;
};
thread_local Context*live=nullptr;
void cuda_check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void memory(uint64_t*out){size_t f=0,t=0;cuda_check(cudaMemGetInfo(&f,&t));out[0]=f;out[1]=t;}
void floor(){uint64_t m[2];memory(m);if(m[0]<uint64_t(16)*1024*1024*1024)throw std::runtime_error("retained16GiB observed boundary floor");}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
Context*owner(void*p,bool allow_poison=false){
 if(!p||p!=live)throw std::logic_error("unknown owner handle");auto*c=live;
 if(c->thread!=std::this_thread::get_id()||!c->r||(!allow_poison&&c->poisoned))throw std::logic_error("wrong/poisoned owner");
 c->r->assertOwnerThread();int d=-1;cuda_check(cudaGetDevice(&d));if(d!=0)throw std::logic_error("one visible local device0");return c;
}
template<class F>int invoke(void*p,F f){Context*c=nullptr;try{c=owner(p);floor();f(*c);floor();return 0;}catch(const std::exception&e){error=e.what();if(c)c->poisoned=true;drain(c);return 1;}}
struct Inputs {
 rt::Runtime&r;std::vector<std::vector<uint8_t>> host;std::vector<rt::Tensor> held;
 ~Inputs(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor upload(const void*p,size_t bytes,const rt::Shape&shape,rt::DataType dtype){
  if(!p||!bytes)throw std::invalid_argument("missing actual input bytes");auto t=rt::Tensor::empty(r,shape,dtype);
  if(t->nbytes()!=bytes)throw std::invalid_argument("actual tensor byte count differs");
  host.emplace_back(bytes);std::memcpy(host.back().data(),p,bytes);held.push_back(t);r.memcpyH2D(t->data(),host.back().data(),bytes,false);return t;
 }
};
JSON info(const rt::Tensor&t){if(!t)return nullptr;return JSON{{"shape",t->shape()},{"dtype",int(t->dtype())},{"bytes",t->nbytes()},{"contiguous",t->is_contiguous()}};}
rt::Tensor tensor(Context&c,const std::string&name,size_t n){
 if(name=="state")return c.index->state();if(name=="raw_cache")return c.index->raw_cache();
 if(!c.pending)throw std::logic_error("no active prepare transaction");auto&p=*c.pending;
 if(name=="scores"){if(n>=p.scores.size())throw std::out_of_range("scores row");return p.scores[n];}
 if(name=="indices"){if(n>=p.indices.size())throw std::out_of_range("indices row");return p.indices[n];}
 if(n)throw std::out_of_range("scalar tensor field");
 if(name=="query")return p.query;if(name=="query_scales")return p.query_scales;
 if(name=="merged_weights")return p.merged_weights;if(name=="raw_query")return p.raw_query;
 if(name=="head_weights")return p.head_weights;if(name=="projected_score")return p.projected_score;
 if(name=="pooled")return p.pooled;if(name=="new_key_values")return p.new_keys.values;
 if(name=="new_key_scales")return p.new_keys.scales;throw std::invalid_argument("unknown output field");
}
}
extern "C" const char*deep_c4_probe_error_r483(){return error.c_str();}
extern "C" void*deep_c4_probe_create_r483(const char*config,size_t layer,int mode){
 std::unique_ptr<Context>c;
 try{
  if(!config||mode<0||mode>1||ever_created.exchange(true))throw std::logic_error("one fixed startup phase per process");
  auto cfg=JSON::parse(config);if(cfg.at("max_position_embeddings")!=32768||cfg.at("compress_ratios").at(layer)!=4)throw std::invalid_argument("full32768 actual C4 layer required");
  int n=0;cuda_check(cudaGetDeviceCount(&n));if(n!=1)throw std::logic_error("one physical GPU visible");cuda_check(cudaSetDevice(0));floor();
  if(deep_constant_slot_metadata_set_r482(mode)||modern_deep_allselected_set_r813(0)||modern_deep_c4_smallt_mode_r9252(0))throw std::logic_error("startup policy not available");
  c=std::make_unique<Context>();
  c->group=std::make_unique<ds::CommunicationGroup>(ds::DistConfig(std::vector<int>{0}),rt::Device::Type::kHygon);
  auto rank=c->group->get_rank_info(0);
  if(c->group->tp_size()!=1||rank.tp_size!=1||rank.tp_rank||rank.pp_size!=1||rank.pp_stage||rank.world_size!=1||rank.world_rank||rank.comm||rank.world_comm||rank.device!=rt::Device(rt::Device::Type::kHygon,0))throw std::logic_error("actual factory topology differs");
  c->r=rt::Runtime::create(rank.device);c->r->enableOwnerDrainFailStop();
  c->index=std::make_unique<md::C4Indexer>(*c->r,cfg,layer,rank,1,8192,512);
  for(auto&k:c->index->state_required_keys())c->required.insert(k);
  if(c->required.empty()||c->index->local_heads()!=64)throw std::logic_error("real TP1 parameter tree missing");
  c->r->syncStream();floor();if(deep_constant_slot_metadata_set_r482(mode)!=2)throw std::logic_error("firstcache policy must freeze");live=c.release();return live;
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->pending.reset();c->index.reset();drain(c.get());c->group.reset();}return nullptr;}
}
extern "C" const char*deep_c4_probe_parameters_r483(void*p){
 int rc=invoke(p,[&](Context&c){JSON j=JSON::array();auto all=c.index->state_dict();for(auto&k:c.required){auto t=all.at(k).tensor();auto v=info(t);v["name"]=k;v["loaded"]=bool(c.loaded.count(k));j.push_back(v);}reply=j.dump();});return rc?nullptr:reply.c_str();
}
extern "C" int deep_c4_probe_load_r483(void*p,const char*name,const void*data,size_t bytes){return invoke(p,[&](Context&c){
 if(c.ready||!name||!c.required.count(name)||c.loaded.count(name))throw std::invalid_argument("required checkpoint key exactly once before postload");
 auto target=c.index->state_dict().at(name).tensor();Inputs work{*c.r};auto src=work.upload(data,bytes,target->shape(),target->dtype());
 c.index->load_parameter(*c.r,name,src);c.r->syncStream();c.loaded.insert(name);
});}
extern "C" int deep_c4_probe_postload_r483(void*p){return invoke(p,[&](Context&c){
 if(c.ready||c.loaded!=c.required)throw std::logic_error("complete actual checkpoint parameter set required");
 c.index->process_weights_after_loading(*c.r,false);c.r->syncStream();c.ready=true;
});}
extern "C" int deep_c4_probe_prepare_r483(void*p,size_t past,size_t tokens,const void*hidden,size_t hidden_bytes,const void*q,size_t q_bytes,const int64_t*positions,size_t npositions,double*seconds){return invoke(p,[&](Context&c){
 if(!c.ready||c.pending||!tokens||tokens>4096||past!=c.past||past>32768||tokens>32768-past||npositions!=tokens||!positions||!seconds)throw std::invalid_argument("one real causal packed request and pending transaction");
 if(hidden_bytes!=tokens*4096*2||q_bytes!=tokens*1024*2)throw std::invalid_argument("actual BF16 hidden/q_lora byte shape");
 for(size_t i=0;i<tokens;++i)if(positions[i]!=int64_t(past+i))throw std::invalid_argument("logical position sequence differs");
 Inputs work{*c.r};auto x=work.upload(hidden,hidden_bytes,{tokens,4096},rt::DataType::kBFloat16);auto z=work.upload(q,q_bytes,{tokens,1024},rt::DataType::kBFloat16);auto pos=work.upload(positions,tokens*8,{tokens},rt::DataType::kInt64);c.r->syncStream();
 auto start=Clock::now();c.pending=std::make_unique<md::C4Indexer::Pending>(c.index->prepare(*c.r,x,z,pos,{{0,past,tokens}}));auto stop=Clock::now();
 *seconds=std::chrono::duration<double>(stop-start).count();c.prepared_tokens=tokens;++c.prepares;
});}
extern "C" const char*deep_c4_probe_tensor_r483(void*p,const char*name,size_t row){int rc=invoke(p,[&](Context&c){if(!name)throw std::invalid_argument("missing field");reply=info(tensor(c,name,row)).dump();});return rc?nullptr:reply.c_str();}
extern "C" int deep_c4_probe_copy_r483(void*p,const char*name,size_t row,void*out,size_t bytes){return invoke(p,[&](Context&c){
 if(!name||!out)throw std::invalid_argument("missing output");auto t=tensor(c,name,row);if(!t||!t->is_contiguous()||t->nbytes()!=bytes)throw std::invalid_argument("actual output shape/stride/bytes required");c.r->memcpyD2H(out,t->data(),bytes);c.r->syncStream();
});}
extern "C" int deep_c4_probe_commit_r483(void*p,double*seconds){return invoke(p,[&](Context&c){if(!c.pending||!seconds)throw std::logic_error("no pending transaction");auto start=Clock::now();c.index->commit(*c.r,*c.pending);auto stop=Clock::now();*seconds=std::chrono::duration<double>(stop-start).count();c.past+=c.prepared_tokens;c.prepared_tokens=0;c.pending.reset();++c.commits;});}
extern "C" int deep_c4_probe_abort_r483(void*p){return invoke(p,[&](Context&c){if(!c.pending)throw std::logic_error("no pending transaction");c.r->syncStream();c.pending.reset();c.prepared_tokens=0;++c.aborts;});}
extern "C" int deep_c4_probe_reset_r483(void*p){return invoke(p,[&](Context&c){if(!c.ready||c.pending)throw std::logic_error("reset requires loaded idle component");c.index->reset(*c.r);c.past=0;++c.resets;});}
extern "C" const char*deep_c4_probe_observe_r483(void*p){int rc=invoke(p,[&](Context&c){uint64_t m[2];memory(m);reply=JSON{{"past",c.past},{"pending",bool(c.pending)},{"ready",c.ready},{"local_heads",64},{"compressed_capacity",8192},{"full_context",32768},{"prepares",c.prepares},{"commits",c.commits},{"aborts",c.aborts},{"resets",c.resets},{"driver_free",m[0]},{"driver_total",m[1]},{"scope","actual TP1 C4Indexer; not TP8 model/attention/quality"}}.dump();});return rc?nullptr:reply.c_str();}
extern "C" int deep_c4_probe_destroy_r483(void*p,uint64_t*out,size_t n){Context*c=nullptr;try{
 c=owner(p,true);if(!out||n!=4)throw std::invalid_argument("close width4");uint64_t m[2];memory(m);out[0]=m[0];drain(c);c->pending.reset();c->index.reset();drain(c);c->group.reset();c->r->syncStream();c->r->trimMemory();std::weak_ptr<rt::Runtime>last=c->r;c->r.reset();delete c;live=nullptr;c=nullptr;memory(m);out[1]=m[0];out[2]=last.expired();out[3]=1;return out[2]?0:1;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
