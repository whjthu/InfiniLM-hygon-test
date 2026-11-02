// Whole C4Indexer transaction. Baseline is the original afc6 C4 class.
// Candidate private class preserves original prepare/commit and enables paged scoring.
#include "deepseek_v4_c4_indexer.hpp"
#include "private_indexer_r359.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <set>
#include <thread>
#include <optional>
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4;
namespace {
thread_local std::string error,serialized;
thread_local std::shared_ptr<rt::Runtime> owner;
thread_local std::weak_ptr<rt::Runtime> last_owner;
constexpr const char*prefix="layers.16.attn.indexer.";
struct Context {
 std::shared_ptr<rt::Runtime> runtime;
 infinicclComm_t comm=nullptr;
 std::thread::id thread=std::this_thread::get_id();
 std::unique_ptr<md::C4Indexer> baseline;
 std::unique_ptr<md::paged_score_r359::C4Indexer> candidate;
 std::optional<md::C4Indexer::Pending> off_pending;
 std::optional<md::paged_score_r359::C4Indexer::Pending> on_pending;
 std::set<std::string> loaded;
 int mode=0,last_mode=-1;
 bool ready=false,poisoned=false;
 double prepare_seconds=0,commit_seconds=0;
 void check(){if(thread!=std::this_thread::get_id()||!runtime||poisoned)throw std::logic_error("C4 fixture owner/poison differs");runtime->assertOwnerThread();}
 void drop(){off_pending.reset();on_pending.reset();last_mode=-1;}
};
void drain(Context*c){if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
template<class F>int invoke(Context*c,F f){try{c->check();f();return 0;}catch(const std::exception&e){error=e.what();c->poisoned=true;drain(c);return 1;}}
rt::Tensor field(Context*c,int which){
 if(c->last_mode<0)throw std::logic_error("No completed transaction");
 bool on=c->last_mode!=0;
 if(which==6)return on?c->candidate->raw_cache():c->baseline->raw_cache();
 if(which==7)return on?c->candidate->state():c->baseline->state();
 auto get=[&](auto&p)->rt::Tensor{switch(which){
 case 0:return p.query;case 1:return p.merged_weights;
 case 2:return p.new_keys.values;case 3:return p.new_keys.scales;
 case 4:return p.scores.at(0);case 5:return p.indices.at(0);
 default:throw std::invalid_argument("Unknown proof field");}};
 return on?get(*c->on_pending):get(*c->off_pending);
}
}
extern "C" const char*deep_c4_fixture_error_r359(){return error.c_str();}
extern "C" size_t deep_c4_fixture_id_size_r359(){return sizeof(infinicclUniqueId);}
extern "C" int deep_c4_fixture_id_r359(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_c4_fixture_create_r359(int device,const char*cfg,const void*raw_id){
 std::unique_ptr<Context>c;
 try{
  if(owner)throw std::logic_error("One C4 fixture per owner thread");
  c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});
  last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
  auto&r=*owner;infini::rt::DeviceGuard guard{r.device().type(),device,true};
  infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));
  if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("C4 TP8 comm init failed");
  infinilm::engine::distributed::RankInfo rank(r.device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
  auto config=nlohmann::json::parse(cfg);
  // Full checkpoint position support; 8192 compressed rows, not old CSA capacity4096.
  if(config.at("max_position_embeddings").get<size_t>()<32768)throw std::invalid_argument("C4 max position too short");
  c->baseline=std::make_unique<md::C4Indexer>(r,config,16,rank,1,8192,512);
  c->candidate=std::make_unique<md::paged_score_r359::C4Indexer>(r,config,16,rank,1,8192,512);
  if(c->baseline->state_required_keys()!=c->candidate->state_required_keys())throw std::logic_error("C4 required keys differ");
  return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->candidate.reset();c->baseline.reset();drain(c.get());if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();}owner.reset();return nullptr;}
}
extern "C" const char*deep_c4_fixture_keys_r359(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>names;for(auto&k:c->baseline->state_required_keys())names.push_back(std::string(prefix)+k);serialized=nlohmann::json(names).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_c4_fixture_load_r359(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("C4 weight prefix/state differs");
 rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("C4 weight dtype differs");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});
 auto key=std::string(name).substr(std::strlen(prefix));c->baseline->load_parameter(*c->runtime,key,x);c->candidate->load_parameter(*c->runtime,key,x);c->loaded.insert(key);
 });}
extern "C" int deep_c4_fixture_finish_r359(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("C4 already loaded");for(auto&k:c->baseline->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("Missing C4 weight "+k);c->baseline->process_weights_after_loading(*c->runtime,false);c->candidate->process_weights_after_loading(*c->runtime,false);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_c4_fixture_reset_r359(void*raw,int mode){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(!c->ready||mode<0||mode>2)throw std::invalid_argument("C4 mode/state differs");c->drop();c->baseline->reset(*c->runtime);c->candidate->reset(*c->runtime);c->mode=mode;c->candidate->score_mode(mode==1);c->runtime->syncStream();});}
extern "C" int deep_c4_fixture_forward_r359(void*raw,size_t past,size_t tokens,const void*hidden,const void*q_lora,const void*positions,void*stream){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||tokens<1||tokens>1024||past>32768||tokens>32768-past)throw std::invalid_argument("C4 transaction geometry differs");
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("C4 producer stream failed");
 c->drop();auto&r=*c->runtime;
 auto wrap=[&](const void*p,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());};
 auto x=wrap(hidden,{tokens,4096},rt::DataType::kBFloat16),q=wrap(q_lora,{tokens,1024},rt::DataType::kBFloat16),pos=wrap(positions,{tokens},rt::DataType::kInt64);
 auto t0=std::chrono::steady_clock::now();
 if(c->mode==0){c->off_pending.emplace(c->baseline->prepare(r,x,q,pos,{{0,past,tokens}}));auto t1=std::chrono::steady_clock::now();c->baseline->commit(r,*c->off_pending);r.syncStream();auto t2=std::chrono::steady_clock::now();c->prepare_seconds=std::chrono::duration<double>(t1-t0).count();c->commit_seconds=std::chrono::duration<double>(t2-t1).count();}
 else{c->on_pending.emplace(c->candidate->prepare(r,x,q,pos,{{0,past,tokens}}));auto t1=std::chrono::steady_clock::now();c->candidate->commit(r,*c->on_pending);r.syncStream();auto t2=std::chrono::steady_clock::now();c->prepare_seconds=std::chrono::duration<double>(t1-t0).count();c->commit_seconds=std::chrono::duration<double>(t2-t1).count();}
 c->last_mode=c->mode;
 });}
extern "C" int deep_c4_fixture_proof_info_r359(void*raw,int which,size_t*shape,size_t*nbytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t){*nbytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("C4 proof ndim differs");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*nbytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_c4_fixture_proof_read_r359(void*raw,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("C4 proof bytes differ");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_c4_fixture_stats_r359(void*raw,double*out,size_t*hits){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{out[0]=c->prepare_seconds;out[1]=c->commit_seconds;*hits=c->candidate->score_hits();});}
extern "C" int deep_c4_fixture_destroy_r359(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("C4 destroy wrong owner");drain(c);c->drop();c->candidate.reset();c->baseline.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("C4 comm destroy failed");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_c4_fixture_shutdown_r359(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
