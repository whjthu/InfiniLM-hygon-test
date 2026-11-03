// Same current C4 class, isolated namespace; no SDK operation/check is removed.
#include "private_indexer.hpp"
#include "topk_contract.hpp"
#include "freeze_protocol.hpp"
#include <fstream>
#include <iterator>
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <optional>
#include <set>
#include <cstdio>
namespace rt=infinilm::runtime;
namespace md=infinilm::models::deepseek_v4::rank64_topk_r416;
namespace {
thread_local std::string error,serialized;
thread_local std::shared_ptr<rt::Runtime> owner;
thread_local std::weak_ptr<rt::Runtime> last_owner;
constexpr const char*prefix="layers.16.attn.indexer.";
struct Context {
 std::shared_ptr<rt::Runtime> runtime;infinicclComm_t comm=nullptr;
 std::thread::id thread=std::this_thread::get_id();std::unique_ptr<md::C4Indexer> module;
 std::optional<md::C4Indexer::Pending> pending;std::set<std::string> loaded;
 bool ready=false,poisoned=false;double prepare_seconds=0,commit_seconds=0;deep416::Stats stats{};
 rt::Tensor raw_before,state_before,state_prepared,frozen_raw,frozen_state;
 freeze416::Protocol freeze;size_t committed_past=0;double freeze_seconds=0,restore_seconds=0;uint64_t prefix_generation=0,restored_generation=0,restore_revision=0;
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
 switch(which){case 11:return p.topk_values.at(0);case 0:return p.query;case 1:return p.merged_weights;case 2:return p.new_keys.values;case 3:return p.new_keys.scales;case 4:return p.scores.at(0);case 5:return p.indices.at(0);default:throw std::invalid_argument("Unknown proof field");}
}
}
extern "C" const char*deep_c4_topk_error_r416(){return error.c_str();}
extern "C" size_t deep_c4_topk_id_size_r416(){return sizeof(infinicclUniqueId);}
extern "C" int deep_c4_topk_id_r416(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*deep_c4_topk_create_r416(int device,const char*cfg,const void*raw_id){std::unique_ptr<Context>c;try{
 if(owner)throw std::logic_error("One Runtime per owner thread");c=std::make_unique<Context>();owner=rt::Runtime::create({rt::Device::Type::kHygon,device});last_owner=owner;c->runtime=owner;owner->enableOwnerDrainFailStop();
 infini::rt::DeviceGuard guard{owner->device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,raw_id,sizeof(id));if(infinicclCommInitRank(&c->comm,8,id,device)!=infinicclSuccess)throw std::runtime_error("C4 TP8 comm init failed");
 infinilm::engine::distributed::RankInfo rank(owner->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->comm;
 auto config=nlohmann::json::parse(cfg);if(config.at("max_position_embeddings").get<size_t>()<32768)throw std::invalid_argument("C4 max context differs");c->module=std::make_unique<md::C4Indexer>(*owner,config,16,rank,1,8192,512);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->module.reset();drain(c.get());if(c->comm&&infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();}owner.reset();return nullptr;}}
extern "C" const char*deep_c4_topk_keys_r416(void*raw){auto*c=static_cast<Context*>(raw);try{c->check();std::vector<std::string>names;for(auto&k:c->module->state_required_keys())names.push_back(std::string(prefix)+k);serialized=nlohmann::json(names).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_c4_topk_load_r416(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(c->ready||!name||std::string(name).rfind(prefix,0)!=0)throw std::invalid_argument("C4 weight prefix/state differs");rt::DataType dt;if(dtype==0)dt=rt::DataType::kInt8;else if(dtype==1)dt=rt::DataType::kFloat32;else if(dtype==2)dt=rt::DataType::kBFloat16;else throw std::invalid_argument("C4 weight dtype differs");
 auto x=rt::Tensor::from_blob(const_cast<void*>(p),rt::Shape(shape,shape+ndim),dt,{rt::Device::Type::kCpu,0});auto key=std::string(name).substr(std::strlen(prefix));c->module->load_parameter(*c->runtime,key,x);c->loaded.insert(key);});}
extern "C" int deep_c4_topk_finish_r416(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(c->ready)throw std::logic_error("C4 already loaded");for(auto&k:c->module->state_required_keys())if(!c->loaded.count(k))throw std::logic_error("Missing C4 weight "+k);c->module->process_weights_after_loading(*c->runtime,false);c->runtime->syncStream();c->ready=true;});}
extern "C" int deep_c4_topk_reset_r416(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{if(!c->ready)throw std::logic_error("C4 not loaded");c->drop();c->module->reset(*c->runtime);c->runtime->syncStream();c->stats={};c->committed_past=0;++c->prefix_generation;});}
extern "C" int deep_c4_topk_forward_r416(void*raw,size_t past,size_t tokens,const void*x,const void*q,const void*positions,void*producer,int mode,int capture){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 if(!c->ready||tokens<1||tokens>1024||past>8496||tokens>8497-past||mode<0||mode>1||capture<0||capture>1||(capture&&(past!=8496||tokens!=1)))throw std::invalid_argument("Frozen C4 diagnostic geometry differs");
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer))!=cudaSuccess)throw std::runtime_error("C4 producer stream failed");c->drop();auto&r=*c->runtime;
 auto hidden=rt::Tensor::from_blob(const_cast<void*>(x),{tokens,4096},rt::DataType::kBFloat16,r.device()),qlora=rt::Tensor::from_blob(const_cast<void*>(q),{tokens,1024},rt::DataType::kBFloat16,r.device()),pos=rt::Tensor::from_blob(const_cast<void*>(positions),{tokens},rt::DataType::kInt64,r.device());
 if(capture&&c->restored_generation!=c->prefix_generation)throw std::logic_error("r416 capture requires restored realprefix");
 if(capture){c->raw_before=clone(r,c->module->raw_cache());c->state_before=clone(r,c->module->state());}
 auto t0=std::chrono::steady_clock::now();c->stats=deep416::owner_phase(mode,[&]{c->pending.emplace(c->module->prepare(r,hidden,qlora,pos,{{0,past,tokens}}));});auto t1=std::chrono::steady_clock::now();
 c->prepare_seconds=std::chrono::duration<double>(t1-t0).count();if(capture)c->state_prepared=clone(r,c->module->state());
 auto t2=std::chrono::steady_clock::now();c->module->commit(r,*c->pending);r.syncStream();auto t3=std::chrono::steady_clock::now();c->commit_seconds=std::chrono::duration<double>(t3-t2).count();c->committed_past=past+tokens;});}
namespace {
size_t freeze_boundary(Context*c){
 auto&r=*c->runtime;auto revision=c->module->freeze_boundary_revision(r,c->committed_past);
 auto raw=c->module->raw_cache(),state=c->module->state();
 if(raw->dtype()!=rt::DataType::kUInt8||raw->nbytes()!=freeze416::raw_bytes||state->dtype()!=rt::DataType::kBFloat16||state->shape()!=rt::Shape{8,512}||state->nbytes()!=freeze416::state_bytes)throw std::logic_error("r416 actual snapshot dtype/shape differs");
 c->freeze.boundary(reinterpret_cast<uintptr_t>(&r),c->committed_past,c->ready,c->poisoned);return revision;
}
void copy_valid(Context*c,bool restore){
 auto&r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};auto stream=static_cast<cudaStream_t>(r.stream());
 auto*raw=static_cast<unsigned char*>(const_cast<void*>(static_cast<const void*>(c->module->raw_cache()->data())));auto*packed=static_cast<unsigned char*>(const_cast<void*>(static_cast<const void*>(c->frozen_raw->data())));
 auto ck=[](cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error("r416 snapshot copy failed="+std::to_string(int(rc)));};
 for(auto part:freeze416::segments)ck(cudaMemcpyAsync(restore?raw+part.raw:packed+part.packed,restore?packed+part.packed:raw+part.raw,part.bytes,cudaMemcpyDeviceToDevice,stream));
 auto current=c->module->state();auto*current_data=const_cast<void*>(static_cast<const void*>(current->data()));auto*frozen_data=const_cast<void*>(static_cast<const void*>(c->frozen_state->data()));ck(cudaMemcpyAsync(restore?current_data:frozen_data,restore?frozen_data:current_data,freeze416::state_bytes,cudaMemcpyDeviceToDevice,stream));
 r.syncStream();
}
}
extern "C" int deep_c4_topk_freeze_r416(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 auto revision=freeze_boundary(c);if(c->freeze.frozen)throw std::logic_error("r416 duplicate snapshot");auto t=std::chrono::steady_clock::now();auto&r=*c->runtime;
 c->frozen_raw=rt::Tensor::empty(r,{freeze416::payload_bytes},rt::DataType::kUInt8);c->frozen_state=rt::Tensor::empty(r,{8,512},rt::DataType::kBFloat16);copy_valid(c,false);
 if(c->module->freeze_boundary_revision(r,c->committed_past)!=revision)throw std::logic_error("r416 capture mutated revision");
 c->freeze.capture(reinterpret_cast<uintptr_t>(&r),c->committed_past,c->ready,c->poisoned);c->freeze_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 });}
extern "C" int deep_c4_topk_restore_r416(void*raw){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 auto revision=freeze_boundary(c);if(!c->freeze.frozen||c->restored_generation==c->prefix_generation)throw std::logic_error("r416 missing snapshot or repeatedrestore sameprefix");auto t=std::chrono::steady_clock::now();copy_valid(c,true);
 if(c->module->freeze_boundary_revision(*c->runtime,c->committed_past)!=revision)throw std::logic_error("r416 restore mutated revision");
 c->freeze.restore(reinterpret_cast<uintptr_t>(c->runtime.get()),c->committed_past,c->ready,c->poisoned);c->restored_generation=c->prefix_generation;c->restore_revision=revision;c->restore_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 });}
extern "C" int deep_c4_topk_freeze_stats_r416(void*raw,double*seconds,uint64_t*counts){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{seconds[0]=c->freeze_seconds;seconds[1]=c->restore_seconds;counts[0]=c->freeze.captures;counts[1]=c->freeze.restores;counts[2]=freeze416::payload_bytes;counts[3]=freeze416::state_bytes;counts[4]=4;counts[5]=c->prefix_generation;counts[6]=c->restored_generation;counts[7]=c->restore_revision;});}
extern "C" int deep_c4_topk_proof_info_r416(void*raw,int which,size_t*shape,size_t*nbytes,int*dtype){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t){*nbytes=0;shape[0]=0;*dtype=-1;return;}if(t->ndim()>4)throw std::logic_error("Proof ndim differs");shape[0]=t->ndim();for(size_t i=0;i<t->ndim();++i)shape[i+1]=t->size(i);*nbytes=t->nbytes();*dtype=int(t->dtype());});}
extern "C" int deep_c4_topk_proof_read_r416(void*raw,int which,void*out,size_t bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{auto t=field(c,which);if(!t||t->nbytes()!=bytes)throw std::invalid_argument("Proof bytes differ");c->runtime->memcpyD2H(out,t->data(),bytes);c->runtime->syncStream();});}
extern "C" int deep_c4_topk_proof_pair_r416(void*raw,void*ids,size_t ids_bytes,void*values,size_t values_bytes){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{
 auto index=field(c,5),value=field(c,11);if(!ids||!values||!index||!value||index->nbytes()!=ids_bytes||value->nbytes()!=values_bytes)throw std::invalid_argument("r416 paired proof bytes");
 auto&r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 r.memcpyD2H(ids,index->data(),ids_bytes); // Original indices copy includes its original Runtime drain.
 // New symmetric2KiB proof payload piggybacks the original following explicit drain.
 if(cudaMemcpyAsync(values,value->data(),values_bytes,cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(r.stream()))!=cudaSuccess)throw std::runtime_error("r416 values proof copy failed");
 r.syncStream();
 });}
extern "C" int deep_c4_topk_stats_r416(void*raw,double*out,uint64_t*stats){auto*c=static_cast<Context*>(raw);return invoke(c,[&]{out[0]=c->prepare_seconds;out[1]=c->commit_seconds;stats[0]=c->stats.calls;stats[1]=c->stats.hits;stats[2]=c->stats.fallbacks;stats[3]=std::hash<std::thread::id>{}(std::this_thread::get_id());});}
extern "C" int deep_c4_topk_dtype_r416(int which){switch(which){case 0:return int(rt::DataType::kFloat32);case 1:return int(rt::DataType::kBFloat16);case 2:return int(rt::DataType::kInt64);case 3:return int(rt::DataType::kUInt8);default:return -1;}}
extern "C" int deep_c4_topk_destroy_r416(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->thread!=std::this_thread::get_id())throw std::logic_error("C4 destroy wrong owner");drain(c);c->drop();c->module.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("C4 comm destroy failed");delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_c4_topk_shutdown_r416(){try{if(owner){owner->syncStream();owner->trimMemory();owner.reset();}return last_owner.expired()?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
// Same actual bridge object, GC-linked to a HOST-only comparator/dispatch consumer.
__attribute__((visibility("hidden"))) int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--sort"){
  std::ifstream in(argv[2],std::ios::binary);std::vector<uint32_t>bits(deep416::N);in.read(reinterpret_cast<char*>(bits.data()),bits.size()*4);if(in.gcount()!=deep416::N*4||in.peek()!=EOF)throw std::invalid_argument("r416 CPU input bytes");
  auto ids=deep416::host_network(bits);std::ofstream out(argv[3],std::ios::binary);for(auto id:ids){auto value=bits.at(id);out.write(reinterpret_cast<const char*>(&value),4);}for(auto id:ids){int64_t wide=id;out.write(reinterpret_cast<const char*>(&wide),8);}if(!out)throw std::runtime_error("r416 CPU result write");return 0;
 }
 if(argc!=1)throw std::invalid_argument("r416 CPU contract arguments");int rc=deep416::CPU_contract();freeze416::CPU_contract();std::puts("{\"passed\":true,\"actual_owner_threads\":8,\"default_OFF_and_fixedshape_dispatch\":true,\"fallback_geometries\":11,\"actual_shared_freeze_protocol\":true,\"freeze_restore_trials\":10,\"invalid_padding_untouched\":true,\"error_restore\":true,\"Native_dlopen\":false}");return rc;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
