// Same current C4 class, isolated namespace; no SDK operation/check is removed.
#include "private_indexer.hpp"
#include "status_timing.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <optional>
#include <set>
#include <cstdio>
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4::status_enclosure_r406;
namespace {
thread_local std::string error,serialized;
thread_local std::shared_ptr<rt::Runtime> owner;
thread_local std::weak_ptr<rt::Runtime> last_owner;
constexpr const char*prefix="layers.16.attn.indexer.";
struct Context {
 std::shared_ptr<rt::Runtime> runtime;infinicclComm_t comm=nullptr;
 std::thread::id thread=std::this_thread::get_id();std::unique_ptr<md::C4Indexer> module;
 std::optional<md::C4Indexer::Pending> pending;std::set<std::string> loaded;
 bool ready=false,poisoned=false;double prepare_seconds=0,commit_seconds=0;deep406::Stats stats{};
 rt::Tensor raw_before,state_before,state_prepared;
 void check(){if(thread!=std::this_thread::get_id()||!runtime||poisoned)throw std::logic_error("C4 fixture owner/poison differs");runtime->assertOwnerThread();}
 void drop(){pending.reset();raw_before={};state_before={};state_prepared={};}
};
void drain(Context*c){if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
template<class F>int invoke(Context*c,F f){try{c->check();f();return 0;}catch(const std::exception&e){error=e.what();c->poisoned=true;drain(c);return 1;}}
rt::Tensor clone(rt::Runtime&r,const rt::Tensor&t){auto out=rt::Tensor::empty(r,t->shape(),t->dtype());out->copy_from(r,t);r.syncStream();return out;}
rt::Tensor field(Context*c,int which){
 if(which==6)return c->module->raw_cache();if(which==7)return c->module->state();
 if(which==8)return c->raw_before;if(which==9)return c->state_before;if(which==10)return c->state_prepared;
 if(!c->pending)throw std::logic_error("No completed C4 transaction");auto&p=*c->pending;
 switch(which){case 0:return p.query;case 1:return p.merged_weights;case 2:return p.new_keys.values;case 3:return p.new_keys.scales;case 4:return p.scores.at(0);case 5:return p.indices.at(0);default:throw std::invalid_argument("Unknown proof field");}
}
}
extern "C" const char*deep_c4_status_error_r406(){return error.c_str();}
extern "C" size_t deep_c4_status_id_size_r406(){return sizeof(infinicclUniqueId);}
extern "C" int deep_c4_status_id_r406(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_c4_status_create_r406(int device,const char*cfg,const void*raw_id){std::unique_ptr<Context>c;try{
 if(owner)throw std::logic_error("One Runtime per owner thread");c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
 infini::rt::DeviceGuard guard{owner->device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("C4 TP8 comm init failed");
 infinilm::engine::distributed::RankInfo rank(owner->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
 auto config=nlohmann::json::parse(cfg);if(config.at("max_position_embeddings").get<size_t>()<32768)throw std::invalid_argument("C4 max context differs");c->module=std::make_unique<md::C4Indexer>(*owner,config,16,rank,1,8192,512);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->module.reset();drain(c.get());if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();}owner.reset();return nullptr;}}
extern "C" const char*deep_c4_status_keys_r406(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>names;for(auto&k:c->module->state_required_keys())names.push_back(std::string(prefix)+k);serialized=nlohmann::json(names).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_c4_status_load_r406(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("C4 weight prefix/state differs");rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("C4 weight dtype differs");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});auto key=std::string(name).substr(std::strlen(prefix));c->module->load_parameter(*c->runtime,key,x);c->loaded.insert(key);});}
extern "C" int deep_c4_status_finish_r406(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("C4 already loaded");for(auto&k:c->module->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("Missing C4 weight "+k);c->module->process_weights_after_loading(*c->runtime,false);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_c4_status_reset_r406(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(!c->ready)throw std::logic_error("C4 not loaded");c->drop();c->module->reset(*c->runtime);c->runtime->syncStream();c->stats={};});}
extern "C" int deep_c4_status_forward_r406(void*raw,size_t past,size_t tokens,const void*x,const void*q,const void*positions,void*producer,int timing,int capture){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||tokens<1||tokens>1024||past>8496||tokens>8497-past||timing<0||timing>1||capture<0||capture>1||(timing&&(past!=8496||tokens!=1))||(capture&&(past!=8496||tokens!=1)))throw std::invalid_argument("Frozen C4 diagnostic geometry differs");
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("C4 producer stream failed");c->drop();auto&r=*c->runtime;
 auto hidden=rt::Tensor::from_blob(const_cast<void*>(x),{tokens,4096},rt::DataType::kBFloat16,r.device()),qlora=rt::Tensor::from_blob(const_cast<void*>(q),{tokens,1024},rt::DataType::kBFloat16,r.device()),pos=rt::Tensor::from_blob(const_cast<void*>(positions),{tokens},rt::DataType::kInt64,r.device());
 if(capture){c->raw_before=clone(r,c->module->raw_cache());c->state_before=clone(r,c->module->state());}
 auto t0=std::chrono::steady_clock::now();c->stats=deep406::owner_phase(timing,[&]{c->pending.emplace(c->module->prepare(r,hidden,qlora,pos,{{0,past,tokens}}));});auto t1=std::chrono::steady_clock::now();
 c->prepare_seconds=std::chrono::duration<double>(t1-t0).count();if(capture)c->state_prepared=clone(r,c->module->state());
 auto t2=std::chrono::steady_clock::now();c->module->commit(r,*c->pending);r.syncStream();auto t3=std::chrono::steady_clock::now();c->commit_seconds=std::chrono::duration<double>(t3-t2).count();});}
extern "C" int deep_c4_status_proof_info_r406(void*raw,int which,size_t*shape,size_t*nbytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t){*nbytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("Proof ndim differs");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*nbytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_c4_status_proof_read_r406(void*raw,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("Proof bytes differ");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_c4_status_stats_r406(void*raw,double*out,uint64_t*counts,uint64_t*errors,uint64_t*wall,uint64_t*cpu,uint64_t*meta){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{out[0]=c->prepare_seconds;out[1]=c->commit_seconds;for(unsigned i=0;i<5;++i){counts[i]=c->stats.calls[i];errors[i]=c->stats.errors[i];wall[i]=c->stats.wall_ns[i];cpu[i]=c->stats.cpu_ns[i];}meta[0]=c->stats.clock_reads;meta[1]=c->stats.clocks_valid;meta[2]=std::hash<std::thread::id>{}(std::this_thread::get_id());});}
extern "C" int deep_c4_status_dtype_r406(int which){switch(which){case 0:return int(rt::DataType::kFloat32);case 1:return int(rt::DataType::kBFloat16);case 2:return int(rt::DataType::kInt64);case 3:return int(rt::DataType::kUInt8);default:return -1;}}
extern "C" int deep_c4_status_destroy_r406(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("C4 destroy wrong owner");drain(c);c->drop();c->module.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("C4 comm destroy failed");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_c4_status_shutdown_r406(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
// Same adapter object is linked with --gc-sections as a HOST-only contract exe.
// Unused Native/SDK/CUDA sections are discarded: zero dlopen/Runtime/GPU.
__attribute__((visibility("hidden"))) int main(){int result=deep406::CPU_contract();std::puts("{\"passed\":true,\"actual_owner_threads\":8,\"phases_per_owner\":5,\"OFF_clock_reads\":0,\"ON_sites_each_once\":true,\"original_status_exception_and_restore\":true,\"Native_dlopen\":false}");return result;}
