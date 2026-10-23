#include <infini/rt/device_guard.h>
#include "glm_moe_dsa_projected_decoder_r22.hpp"
#include "glm_decode_leases_r1354.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "glm_layer_trace_r5670.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <ctime>
#include <set>
#include <array>
#include <optional>
#include <cstring>
namespace rt=infinilm::runtime;namespace model=infinilm::models::glm_moe_dsa;
namespace infinilm::models::glm_moe_dsa {void glm_indexer_policy_r998(rt::Runtime&,bool);void glm_indexer_policy_clear_r998(rt::Runtime&);}
namespace{
thread_local std::string error,serialized;thread_local bool tracing=false;
using Clock=std::chrono::steady_clock;
double cpu(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return double(t.tv_sec)+double(t.tv_nsec)*1e-9;}
struct Mark{std::string name;cudaEvent_t event;Clock::time_point wall;double threadCPU;};thread_local std::vector<Mark>marks;
void ck(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void clear(){for(auto&m:marks)ck(cudaEventDestroy(m.event));marks.clear();}
struct Context{std::shared_ptr<rt::Runtime>r;std::unique_ptr<model::DecoderProjectedR22>layer;infinicclComm_t tp=nullptr;std::set<std::string>loaded;std::array<size_t,64>lengths{};bool closing=false;bool ready=false,poison=false,bounded=true,owned=true;int device=-1;};
void owner(Context*c){if(!c||!c->r||c->closing)throw std::invalid_argument("context");c->r->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device||c->r->isGraphRecording())throw std::invalid_argument("device/eager");cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(static_cast<cudaStream_t>(c->r->stream()),&status));if(status!=cudaStreamCaptureStatusNone)throw std::invalid_argument("capture");}
void input(Context*c,const void*p){if(!p)throw std::invalid_argument("null GPU pointer");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("GPU pointer device/type");}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;std::optional<model::DecoderProjectedR22::Output>output;~Work(){try{r.syncStream();}catch(...){std::terminate();}}};
}
void glm_layer_mark_r5670(rt::Runtime&r,const char*name){if(!tracing)return;cudaEvent_t e{};ck(cudaEventCreate(&e));try{ck(cudaEventRecord(e,static_cast<cudaStream_t>(r.stream())));marks.push_back({name,e,Clock::now(),cpu()});}catch(...){cudaEventDestroy(e);throw;}}
extern "C" const char*glm_layer_error_r5670(){return error.c_str();}
extern "C" const char*glm_layer_report_r5670(){return serialized.c_str();}
extern "C" size_t glm_layer_id_size_r5670(){return sizeof(infinicclUniqueId);}
extern "C" int glm_layer_id_r5670(void*p){if(!p)return 1;infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(p,&id,sizeof(id));return 0;}
extern "C" void*glm_layer_create_r5670(int device,const char*config,const void*uid){std::unique_ptr<Context>c;try{
 if(device<0||device>7||!config||!uid)throw std::invalid_argument("create args");auto cfg=nlohmann::json::parse(config);if(cfg.at("mlp_layer_types").at(38)!="sparse"||cfg.at("indexer_types").at(38)!="full")throw std::invalid_argument("layer38 full/sparse boundary");
 c=std::make_unique<Context>();c->device=device;infini::rt::DeviceGuard g{rt::Device::Type::kHygon,device,true};c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();infinicclUniqueId id{};std::memcpy(&id,uid,sizeof(id));if(infinicclCommInitRank(&c->tp,8,id,device)!=infinicclSuccess)throw std::runtime_error("TP8 create");
 infinilm::engine::distributed::RankInfo rank(c->r->device());rank.tp_size=8;rank.tp_rank=device;rank.world_size=8;rank.world_rank=device;rank.comm=c->tp;
 c->layer=std::make_unique<model::DecoderProjectedR22>(*c->r,cfg,38,rank,64,640,2048,&c->bounded,&c->owned);return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c&&c->tp)infinicclCommDestroy(c->tp);return nullptr;}}
extern "C" const char*glm_layer_keys_r5670(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);serialized=nlohmann::json(c->layer->state_dict_keys()).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_layer_load_r5670(void*raw,const char*json){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;if(c->ready||c->poison||!json)throw std::invalid_argument("load state");struct Sources{rt::Runtime&r;std::unordered_map<std::string,rt::Tensor>map;~Sources(){try{r.syncStream();}catch(...){std::terminate();}}}held{*c->r};auto&sources=held.map;for(auto&v:nlohmann::json::parse(json)){auto name=v.at("name").get<std::string>();auto shape=v.at("shape").get<rt::Shape>();int d=v.at("dtype");auto p=reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>());if(!p||d<0||d>2||sources.count(name))throw std::invalid_argument("load descriptor");sources.emplace(name,rt::Tensor::from_blob(p,shape,d==0?rt::DataType::kFloat32:d==1?rt::DataType::kBFloat16:rt::DataType::kUInt8,{rt::Device::Type::kCpu,0}));}c->layer->load_parameters_no_sync(*c->r,sources,true);c->r->syncStream();for(auto&v:sources)c->loaded.insert(v.first);return 0;}catch(const std::exception&e){error=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_layer_finish_r5670(void*raw){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;if(c->ready||c->poison)throw std::invalid_argument("finish state");for(auto&k:c->layer->state_dict_keys())if(!c->loaded.count(k))throw std::invalid_argument("missing weight");c->layer->process_weights_after_loading(*c->r,false);c->r->syncStream();model::glm_indexer_policy_r998(*c->r,true);c->ready=true;return 0;}catch(const std::exception&e){error=e.what();if(own){drain(c);c->poison=true;}return 1;}}
extern "C" int glm_layer_reset_r5670(void*raw){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;c->layer->reset(*c->r);c->lengths.fill(0);c->poison=false;return 0;}catch(const std::exception&e){error=e.what();if(own)drain(c);return 1;}}
// Producer must be complete before entry. No borrowed caller stream/API or
// extra stream is used; future Python harness synchronizes producer outsidewall.
extern "C" int glm_layer_forward_r5670(void*raw,size_t N,size_t B,const size_t*desc,const void*x,const void*positions,void*out,int profile){auto*c=static_cast<Context*>(raw);bool own=false,mutating=false;try{
 owner(c);own=true;if(!c->ready||c->poison||!desc||!N||N>4096||!B||B>8||(profile!=0&&profile!=1))throw std::invalid_argument("forward state/count");input(c,x);input(c,positions);input(c,out);
 std::vector<model::NativeAttentionProjectedR22::Request>requests;std::set<size_t>seen;size_t total=0;for(size_t i=0;i<B;++i){size_t row=desc[i*3],past=desc[i*3+1],count=desc[i*3+2];if(row>=64||!seen.insert(row).second||!count||past!=c->lengths[row]||past>640||count>640-past||total>N||count>N-total)throw std::invalid_argument("wholelist row/past/count");requests.push_back({row,past,count});total+=count;}if(total!=N)throw std::invalid_argument("packed N");
 Work w{*c->r};w.held={rt::Tensor::from_blob(const_cast<void*>(x),{N,6144},rt::DataType::kBFloat16,c->r->device()),rt::Tensor::from_blob(const_cast<void*>(positions),{N},rt::DataType::kInt64,c->r->device()),rt::Tensor::from_blob(out,{N,6144},rt::DataType::kBFloat16,c->r->device())};std::vector<int64_t>pos(N);c->r->memcpyD2H(pos.data(),positions,N*8);c->r->syncStream();size_t off=0;for(auto q:requests)for(size_t j=0;j<q.tokens;++j)if(pos[off++]!=int64_t(q.past+j))throw std::invalid_argument("positions");
 clear();tracing=profile!=0;auto begin=Clock::now();double cp=cpu();mutating=true;bool lease_decode=N==B;for(auto q:requests)lease_decode=lease_decode&&q.tokens==1&&q.past>0;model::decode_leases_r1354::Guard lease(*c->r,lease_decode);glm_layer_mark_r5670(*c->r,"layer.begin");w.output=c->layer->forward(*c->r,w.held[0],w.held[1],requests);glm_layer_mark_r5670(*c->r,"layer.return");w.held[2]->copy_from(*c->r,w.output->hidden);auto ls=lease.finish();c->r->syncStream();auto wall=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();double ct=(cpu()-cp)*1000;tracing=false;
 nlohmann::json report={{"N",N},{"B",B},{"layer",38},{"profile",profile},{"wall_ms",wall},{"threadCPU_ms",ct},{"stages",nlohmann::json::array()},{"lease_peak_logical_bytes",ls.peak_logical_bytes},{"Runtime_identity",reinterpret_cast<uintptr_t>(c->r.get())}};
 for(size_t i=1;i<marks.size();++i){float ms=0;ck(cudaEventElapsedTime(&ms,marks[i-1].event,marks[i].event));report["stages"].push_back({{"from",marks[i-1].name},{"to",marks[i].name},{"stream_timeline_ms_includes_host_enqueue_idle",ms},{"host_wall_ms",std::chrono::duration<double,std::milli>(marks[i].wall-marks[i-1].wall).count()},{"threadCPU_ms",(marks[i].threadCPU-marks[i-1].threadCPU)*1000}});}
 // Diagnostic route readback is AFTER complete-layer wall, same for profileOFF.
 auto ids=w.output->ffn.route_indices;std::vector<int64_t>route(ids->numel());c->r->memcpyD2H(route.data(),ids->data(),route.size()*8);c->r->syncStream();std::vector<size_t>hist(256,0);for(auto v:route){if(v<0||v>=256)throw std::runtime_error("route range");++hist[size_t(v)];}size_t U=0,maximum=0;for(auto n:hist){U+=n>0;maximum=std::max(maximum,n);}report["route_histogram"]=hist;report["route_U"]=U;report["route_maxrows"]=maximum;report["route_readback_outside_wall"]=true;
 for(auto q:requests)c->lengths[q.row]=q.past+q.tokens;serialized=report.dump();clear();return 0;
 }catch(const std::exception&e){error=e.what();tracing=false;if(own){drain(c);if(mutating)c->poison=true;try{clear();}catch(...){std::terminate();}}return 1;}}
extern "C" int glm_layer_destroy_r5670(void*raw,int*expired){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;if(!expired)throw std::invalid_argument("destroy output");std::weak_ptr<rt::Runtime>w=c->r;drain(c);if(modern_glm_fp8_runtime_clear_r809(c->r.get()))throw std::runtime_error("FP8 clear");model::glm_indexer_policy_clear_r998(*c->r);c->closing=true;c->layer.reset();drain(c);if(infinicclCommDestroy(c->tp)!=infinicclSuccess)throw std::runtime_error("TP destroy");c->tp=nullptr;c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){error=e.what();if(own)drain(c);return 1;}}

// Out-of-timing TP8 unanimous quality/control vote. One replicated F32 scalar
// on the owning device, producer complete on entry, same Runtime stream.
extern "C" int glm_layer_control_r5670(void*raw,void*values){auto*c=static_cast<Context*>(raw);bool own=false;try{owner(c);own=true;input(c,values);Work w{*c->r};w.held.push_back(rt::Tensor::from_blob(values,{1},rt::DataType::kFloat32,c->r->device()));infinilm::engine::distributed::allreduce_(*c->r,w.held[0],w.held[0],infinicclSum,c->tp);c->r->syncStream();return 0;}catch(const std::exception&e){error=e.what();if(own)drain(c);return 1;}}
