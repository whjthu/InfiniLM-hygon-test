#include "geometry.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "engine/distributed/communication_group.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <cstring>
#include <memory>
#include <vector>
#include <cstdio>
#include <chrono>
namespace rt=infinilm::runtime;namespace dist=infinilm::engine::distributed;namespace g=glm_global_tp16_r470;
namespace {
thread_local std::string error,serialized;
struct Context {std::shared_ptr<rt::Runtime>r;std::unique_ptr<dist::RankInfo>rank;infinicclComm_t world=nullptr,tp=nullptr;g::J report;bool poisoned=false,probed=false;std::FILE*trace_file=nullptr;~Context(){if(trace_file)std::fclose(trace_file);}};
void owner(Context*c){if(!c||!c->r)throw std::invalid_argument("r470 context");c->r->assertOwnerThread();int d=-1;if(cudaGetDevice(&d)!=cudaSuccess||d!=c->r->device().index()||c->r->isGraphRecording())throw std::invalid_argument("r470 original owner/device/eager");}
// Private opt-in host-only observation. Files are created exclusively under
// the existing /local tmpfs bind; no shared-filesystem/peer gate or GPU sync.
void trace(Context*c,const char*phase){if(!c->trace_file)return;auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();if(std::fprintf(c->trace_file,"{\"rank\":%d,\"phase\":\"%s\",\"steady_ns\":%lld}\n",c->rank->tp_rank,phase,static_cast<long long>(ns))<0||std::fflush(c->trace_file)!=0)throw std::runtime_error("r470 local passive trace write");}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void comms(Context*c){if(c->tp){if(infinicclCommDestroy(c->tp)!=infinicclSuccess)throw std::runtime_error("r470 TP destroy");c->tp=nullptr;}if(c->world){if(infinicclCommDestroy(c->world)!=infinicclSuccess)throw std::runtime_error("r470 world destroy");c->world=nullptr;}}
}
extern "C" const char* glm_global_tp16_error_r470(){return error.c_str();}
extern "C" size_t glm_global_tp16_id_size_r470(){return sizeof(infinicclUniqueId);}
extern "C" int glm_global_tp16_id_r470(void*out,size_t bytes){try{if(!out||bytes!=sizeof(infinicclUniqueId))throw std::invalid_argument("r470 exact UID size");infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)throw std::runtime_error("r470 UID");std::memcpy(out,&id,sizeof(id));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" void* glm_global_tp16_create_r470(int node,int device,size_t rows,size_t capacity,const char*config,const void*world_id,const void*tp_id,size_t id_bytes){
 std::unique_ptr<Context>c;try{if(!config||!world_id||!tp_id||id_bytes!=sizeof(infinicclUniqueId)||std::memcmp(world_id,tp_id,id_bytes)==0)throw std::invalid_argument("r470 two distinct exact UID buffers");auto cfg=g::J::parse(config);auto geo=g::geometry(cfg,node,device,rows,capacity);g::Scope optin(cfg);auto p=g::placement(node,device);c=std::make_unique<Context>();c->report=geo;c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();c->r->assertOwnerThread();infini::rt::DeviceGuard device_guard{c->r->device().type(),device,true};infinicclUniqueId w{},t{};std::memcpy(&w,world_id,id_bytes);std::memcpy(&t,tp_id,id_bytes);
 if(infinicclCommInitRank(&c->world,16,w,p.global)!=infinicclSuccess)throw std::runtime_error("r470 world16 init");
 if(infinicclCommInitRank(&c->tp,16,t,p.global)!=infinicclSuccess)throw std::runtime_error("r470 crossnode TP16 init");
 c->rank=std::make_unique<dist::RankInfo>(c->r->device());auto&r=*c->rank;r.tp_size=16;r.tp_rank=p.global;r.comm=c->tp;r.pp_size=1;r.pp_stage=0;r.world_size=16;r.world_rank=p.global;r.world_comm=c->world;if(!g::admitted(r))throw std::logic_error("r470 exact global RankInfo");c->r->syncStream();c->report["component_created"]=true;c->report["Model_READY"]=false;return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c)try{comms(c.get());}catch(...){std::terminate();}return nullptr;}}
extern "C" const char* glm_global_tp16_geometry_r470(void*raw){try{auto*c=static_cast<Context*>(raw);owner(c);serialized=c->report.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_global_tp16_trace_path_r470(void*raw,const char*path){try{auto*c=static_cast<Context*>(raw);owner(c);if(c->probed||c->trace_file||!path||std::strncmp(path,"/local/",7)!=0||std::strstr(path,"..")||std::strchr(path+7,'/'))throw std::invalid_argument("r470 exclusive local trace path");std::unique_ptr<std::FILE,int(*)(std::FILE*)>file(std::fopen(path,"wx"),&std::fclose);if(!file)throw std::runtime_error("r470 local trace exclusive open");c->report["passive_trace_enabled"]=true;c->trace_file=file.release();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
// Future device component proof only: real TP16 SUM and rank-ordered allgather.
// No Model tensor, cache, checkpoint payload, token or HTTP operation.
extern "C" int glm_global_tp16_probe_r470(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);if(c->poisoned||c->probed)throw std::logic_error("r470 one proof/no replay");c->probed=true;auto&r=*c->r;float value=float(c->rank->tp_rank+1);auto x=rt::Tensor::empty(r,{1},rt::DataType::kFloat32),sum=rt::Tensor::empty(r,{1},rt::DataType::kFloat32),gather=rt::Tensor::empty(r,{16},rt::DataType::kFloat32);r.memcpyH2D(x->data(),&value,4,false);trace(c,"allreduce-submit-begin");dist::allreduce_(r,sum,x,infinicclSum,c->tp);trace(c,"allreduce-submit-return");trace(c,"allgather-submit-begin");dist::allgather_(r,gather,x,c->tp);trace(c,"allgather-submit-return");trace(c,"original-sync-begin");r.syncStream();trace(c,"original-sync-return");float s=0;std::vector<float>v(16);r.memcpyD2H(&s,sum->data(),4);r.memcpyD2H(v.data(),gather->data(),64);if(s!=136)throw std::runtime_error("r470 global SUM differs");for(int i=0;i<16;++i)if(v[i]!=float(i+1))throw std::runtime_error("r470 TP rank order differs");c->report["TP_SUM"]=s;c->report["TP_order"]=v;c->report["collective_proof"]=true;return 0;}catch(const std::exception&e){error=e.what();if(c)c->poisoned=true;drain(c);return 1;}}
extern "C" int glm_global_tp16_close_r470(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);drain(c);comms(c);std::weak_ptr<rt::Runtime>weak=c->r;c->rank.reset();c->r.reset();if(!weak.expired())throw std::runtime_error("r470 Runtime retained");if(c->trace_file){auto*f=c->trace_file;c->trace_file=nullptr;if(std::fclose(f)!=0)throw std::runtime_error("r470 trace close");}delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
