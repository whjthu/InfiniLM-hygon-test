// No production Native/SDK changes. One original C4 Pending supplies both scorers.
#include "deepseek_v4_c4_indexer.hpp"
#include "deep_reduce_wide_r3443.hpp"
#include "execution/submit.hpp"
#include "lease_contract.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <array>
#include <optional>
#include <set>
#include <thread>
#include <cstring>
extern "C" int deep_paged_scores_r359(void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,void*,size_t,size_t,size_t,size_t,size_t,void*);
namespace rt=infinilm::runtime;namespace md=infinilm::models::deepseek_v4;namespace ex=infinilm::execution;
namespace {
thread_local std::string error,serialized;thread_local std::shared_ptr<rt::Runtime> owner;thread_local std::weak_ptr<rt::Runtime> last_owner;
constexpr const char*prefix="layers.16.attn.indexer.";
struct Context {
 std::shared_ptr<rt::Runtime> runtime;infinicclComm_t comm=nullptr;std::thread::id thread=std::this_thread::get_id();
 std::unique_ptr<md::C4Indexer> indexer;std::optional<md::C4Indexer::Pending> pending;std::set<std::string> loaded;
 same_cache_r390::Lease lease;bool ready=false,poisoned=false;size_t past=0,before=0,after=0;
 void*stream=nullptr;const void*pending_address=nullptr;const void*indexer_address=nullptr;
 rt::Tensor raw_before,raw_frozen,state_before,state_prepared,x,q,pos;
 std::array<rt::Tensor,2>scores,indices;
 void check(){if(thread!=std::this_thread::get_id()||!runtime||poisoned)throw std::logic_error("Wrong owner/poisoned context");runtime->assertOwnerThread();}
 void live(uint64_t token){check();lease.check(token);if(!pending||&*pending!=pending_address||indexer.get()!=indexer_address||runtime->stream()!=stream)throw std::logic_error("Pending/owner/stream identity changed");}
 void drop(){pending.reset();scores={};indices={};raw_before={};raw_frozen={};state_before={};state_prepared={};x={};q={};pos={};pending_address=nullptr;indexer_address=nullptr;}
};
void drain(Context*c){if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
template<class F>int invoke(Context*c,F f){try{c->check();f();return 0;}catch(const std::exception&e){error=e.what();c->poisoned=true;drain(c);return 1;}}
rt::Tensor clone(rt::Runtime&r,const rt::Tensor&t){auto out=rt::Tensor::empty(r,t->shape(),t->dtype());out->copy_from(r,t);r.syncStream();return out;}
rt::Tensor ids(rt::Runtime&r,const std::vector<int64_t>&v){auto t=rt::Tensor::empty(r,{v.size()},rt::DataType::kInt64);r.memcpyH2D(t->data(),v.data(),v.size()*8,false);return t;}
void status(rt::Runtime&r,const rt::Tensor&t){int flag=-1;r.memcpyD2H(&flag,t->data(),4);r.syncStream();if(flag)throw std::runtime_error("Score validation status="+std::to_string(flag));}
rt::Tensor field(Context*c,int which){
 if(!c->pending)throw std::logic_error("No retained Pending");auto&p=*c->pending;
 switch(which){case 0:return p.query;case 1:return p.merged_weights;case 2:return p.new_keys.values;case 3:return p.new_keys.scales;
 case 4:return p.scores.at(0);case 5:return p.indices.at(0);case 6:return c->raw_before;case 7:return c->raw_frozen;
 case 8:return c->state_before;case 9:return c->state_prepared;case 10:return c->indexer->raw_cache();case 11:return c->indexer->state();
 case 12:return c->scores[0];case 13:return c->indices[0];case 14:return c->scores[1];case 15:return c->indices[1];case 16:return c->pos;
 default:throw std::invalid_argument("Unknown lease proof field");}
}
}
extern "C" const char*deep_c4_lease_error_r390(){return error.c_str();}
extern "C" size_t deep_c4_lease_id_size_r390(){return sizeof(infinicclUniqueId);}
extern "C" int deep_c4_lease_id_r390(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_c4_lease_create_r390(int device,const char*cfg,const void*raw_id){std::unique_ptr<Context>c;try{
 if(owner)throw std::logic_error("One Runtime per owner thread");c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
 infini::rt::DeviceGuard guard{owner->device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("TP8 comm init");
 infinilm::engine::distributed::RankInfo rank(owner->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
 c->indexer=std::make_unique<md::C4Indexer>(*owner,nlohmann::json::parse(cfg),16,rank,1,8192,512);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->indexer.reset();if(c->comm)infinicclCommDestroy(c->comm);}owner.reset();return nullptr;}}
extern "C" const char*deep_c4_lease_keys_r390(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>v;for(auto&k:c->indexer->state_required_keys())v.push_back(std::string(prefix)+k);serialized=nlohmann::json(v).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_c4_lease_load_r390(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("Weight state/prefix");rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("Weight dtype");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});auto key=std::string(name).substr(std::strlen(prefix));c->indexer->load_parameter(*c->runtime,key,x);c->loaded.insert(key);});}
extern "C" int deep_c4_lease_finish_r390(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("Already loaded");for(auto&k:c->indexer->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("Missing "+k);c->indexer->process_weights_after_loading(*c->runtime,false);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_c4_lease_reset_r390(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{c->lease.idle();c->drop();c->indexer->reset(*c->runtime);c->lease.reset();});}
extern "C" int deep_c4_lease_prefix_r390(void*raw,size_t past,size_t count,const void*x,const void*q,const void*pos,void*producer){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 c->lease.idle();if(!c->ready||!count||count>1024||past>=8575||count>8575-past)throw std::invalid_argument("Frozen prefix scope");if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("Producer stream");auto&r=*c->runtime;if(md::wide_r3443::selected(r)!=0)throw std::logic_error("Expected original F32 prefix TPsum mode0");
 auto xx=rt::Tensor::from_blob(const_cast<void*>(x),{count,4096},rt::DataType::kBFloat16,r.device()),qq=rt::Tensor::from_blob(const_cast<void*>(q),{count,1024},rt::DataType::kBFloat16,r.device()),pp=rt::Tensor::from_blob(const_cast<void*>(pos),{count},rt::DataType::kInt64,r.device());auto p=c->indexer->prepare(r,xx,qq,pp,{{0,past,count}});c->indexer->commit(r,p);r.syncStream();});}
extern "C" int deep_c4_lease_prepare_r390(void*raw,size_t past,const void*x,const void*q,const void*pos,void*producer,uint64_t*token){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 c->lease.idle();if(!c->ready||!token||(past!=8575&&past!=8576))throw std::invalid_argument("Frozen T1 boundary scope");if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("Producer stream");auto&r=*c->runtime;c->drop();
 if(md::wide_r3443::selected(r)!=0)throw std::logic_error("Expected original F32 C1 TPsum mode0");
 c->past=past;c->before=past/4;c->after=(past+1)/4;c->x=rt::Tensor::from_blob(const_cast<void*>(x),{1,4096},rt::DataType::kBFloat16,r.device());c->q=rt::Tensor::from_blob(const_cast<void*>(q),{1,1024},rt::DataType::kBFloat16,r.device());c->pos=rt::Tensor::from_blob(const_cast<void*>(pos),{1},rt::DataType::kInt64,r.device());
 c->raw_before=clone(r,c->indexer->raw_cache());c->state_before=clone(r,c->indexer->state());c->pending.emplace(c->indexer->prepare(r,c->x,c->q,c->pos,{{0,past,1}}));
 c->raw_frozen=clone(r,c->indexer->raw_cache());c->state_prepared=clone(r,c->indexer->state());c->stream=r.stream();c->pending_address=&*c->pending;c->indexer_address=c->indexer.get();*token=c->lease.begin();});}
extern "C" int deep_c4_lease_score_r390(void*raw,uint64_t token,int mode){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 c->live(token);c->lease.score(token,mode);auto&r=*c->runtime;if(md::wide_r3443::selected(r)!=0)throw std::logic_error("No wide/F64 sum frame allowed");auto&p=*c->pending;size_t n=c->after;auto counts=rt::Tensor::empty(r,{1},rt::DataType::kInt32),flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);int32_t visible=int32_t(n);r.memcpyH2D(counts->data(),&visible,4,false);
 auto out=rt::Tensor::empty(r,{1,n},rt::DataType::kFloat32);r.setDeviceMemory(out->data(),0xff,out->nbytes());
 if(mode==0){
  // Same SDK cache gather and score producer as original C4. Original prepare
  // already certified all historical slots written; no mutation is allowed in lease.
  std::vector<int64_t>slots(c->before);for(size_t i=0;i<c->before;++i)slots[i]=int64_t(i);auto slot=ids(r,slots);
  auto history_key=rt::Tensor::empty(r,{c->before,128},rt::DataType::kUInt8),history_scale=rt::Tensor::empty(r,{c->before,1},rt::DataType::kFloat32);
  auto key=rt::Tensor::empty(r,{n,128},rt::DataType::kUInt8),scale=rt::Tensor::empty(r,{n,1},rt::DataType::kFloat32);
  ex::submit<infini::ops::DeepseekV4IndexerCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4IndexerCache::DefaultConfig(r.device().type()),c->raw_frozen,slot,history_key,history_scale,flag,size_t{1});status(r,flag);
  key->narrow({{0,0,c->before}})->copy_from(r,history_key);scale->narrow({{0,0,c->before}})->copy_from(r,history_scale);
  if(n>c->before){key->narrow({{0,c->before,1}})->copy_from(r,p.new_keys.values);scale->narrow({{0,c->before,1}})->copy_from(r,p.new_keys.scales);}
  ex::submit<infini::ops::DeepseekV4IndexerScores>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4IndexerScores::DefaultConfig(r.device().type()),out,p.query,p.merged_weights,key,scale,counts,flag);status(r,flag);
 }else{
  auto decoded=rt::Tensor::empty(r,{1,8,128},rt::DataType::kFloat32);int code=deep_paged_scores_r359(out->data(),c->raw_frozen->data(),p.query->data(),p.merged_weights->data(),n>c->before?p.new_keys.values->data():nullptr,n>c->before?p.new_keys.scales->data():nullptr,counts->data(),flag->data(),decoded->data(),c->before,n,0,c->raw_frozen->size(0)*64,0,r.stream());if(code)throw std::runtime_error("Old paged score launch="+std::to_string(code));status(r,flag);
 }
 md::wide_r3443::reduce(r,out,c->comm,2);auto values=rt::Tensor::empty(r,{1,512},rt::DataType::kFloat32),indices=rt::Tensor::empty(r,{1,512},rt::DataType::kInt64);ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(r.device().type()),out,int64_t{512},int64_t{1},true,true,values,indices);r.syncStream();c->scores[mode]=out;c->indices[mode]=indices;
 });}
extern "C" int deep_c4_lease_commit_r390(void*raw,uint64_t token){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{c->live(token);if(c->lease.scored!=3)throw std::logic_error("Missing score qualification");c->indexer->commit(*c->runtime,*c->pending);c->runtime->syncStream();c->lease.finish(token);});}
extern "C" const char*deep_c4_lease_identity_r390(void*raw,uint64_t token){auto*c=static_cast<Context*>(raw);try{c->check();c->lease.readable(token);if(!c->pending)throw std::logic_error("Missing retained Pending");auto&p=*c->pending;auto ptr=[](const void*v){return uint64_t(reinterpret_cast<uintptr_t>(v));};
 serialized=nlohmann::json{{"generation",c->lease.generation},{"active",c->lease.active},{"committed",c->lease.committed},{"past",c->past},{"before",c->before},{"after",c->after},{"Runtime",ptr(c->runtime.get())},{"C4owner",ptr(c->indexer.get())},{"stream",ptr(c->runtime->stream())},{"Pending",ptr(&p)},{"frozen_raw",ptr(c->raw_frozen->data())},{"query",ptr(p.query->data())},{"merged_weights",ptr(p.merged_weights->data())},{"tail_values",p.new_keys.values?ptr(p.new_keys.values->data()):0},{"tail_scales",p.new_keys.scales?ptr(p.new_keys.scales->data()):0},{"positions",ptr(c->pos->data())},{"original_commit_validates_private_owner_revision",true}}.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_c4_lease_proof_info_r390(void*raw,uint64_t token,int which,size_t*shape,size_t*bytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{c->lease.readable(token);auto t=field(c,which);if(!t){*bytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("Proof rank");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*bytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_c4_lease_proof_read_r390(void*raw,uint64_t token,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{c->lease.readable(token);auto t=field(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("Proof bytes");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_c4_lease_destroy_r390(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("Destroy owner");drain(c);c->drop();c->indexer.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("Comm destroy");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_c4_lease_shutdown_r390(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
