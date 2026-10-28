#pragma once
#include "deep_owner_events_r8900.hpp"
#include "deep_reduce_wide_r3443.hpp"
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <exception>
#include <cmath>
namespace deep_attention_inner_r9193 {
using RT=infinilm::runtime::Runtime;
constexpr size_t devices=8,layers=43,events=7;
struct Sequence {unsigned next=0;void mark(unsigned id){if(id!=next||id>=events)throw std::logic_error("9193 marker sequence");++next;}void complete()const{if(next!=events)throw std::logic_error("9193 incomplete markers");}};
struct Row {bool claimed=false,done=false;int device=-1,layer=-1,kind=-1,error=0,create_during=-1,query_during=-1,destroy_during=-1;unsigned created=0,queried=0,destroyed=0;uintptr_t owner=0,stream=0,thread=0;size_t tokens=0,requests=0;std::array<double,events>gpu{},wall{},cpu{};double final_drain_host_ms=0;};
inline std::array<std::array<Row,layers>,devices>rows{};
inline std::atomic<int>mode{0};inline std::atomic<unsigned>running{0};
inline int arm(int value){namespace m=infinilm::models::deepseek_v4;std::lock_guard<std::mutex>lock(m::wide_r3443::policy_mutex);if(value<0||value>1||m::wide_r3443::in_flight||running.load()||deep_owner_r8900::window.load())return 1;if(value==0){mode.store(0);return 0;}if(mode.load())return 2;for(auto&rank:rows)for(auto&row:rank)row=Row{};mode.store(1);return 0;}
inline int locate_layer(RT*owner,void*stream,deep_owner_r8900::Layer*active,int d,int kind){
 if(!active||d<0||d>=int(devices)||kind<0||kind>2)throw std::logic_error("9193 owner bounds");int layer=-1;for(int l=0;l<int(layers);++l)if(active==&deep_owner_r8900::pool[d][l]){layer=l;break;}if(layer<0||active->owner!=owner||active->stream!=stream)throw std::logic_error("9193 owner/layer/stream binding");return layer;
}
struct Frame {
 RT&r;Row*row=nullptr;std::array<cudaEvent_t,events>ev{};Sequence seq;bool counted=false,finished=false;
 Frame(RT&runtime,int kind,size_t tokens,size_t count):r(runtime){
  if(!mode.load(std::memory_order_acquire))return;
  r.assertOwnerThread();auto*active=deep_owner_r8900::active;int d=r.device().index();if(!active||deep_owner_r8900::window.load()!=1||d<0||d>=int(devices)||r.isGraphRecording())throw std::logic_error("9193 requires originalOwnerwindow1/context");int layer=locate_layer(&r,r.stream(),active,d,kind);
  if(!tokens||!count)throw std::logic_error("9193 owner/layer/stream binding");row=&rows[d][layer];if(row->claimed)throw std::logic_error("9193 duplicate layer forward");row->claimed=true;row->device=d;row->layer=layer;row->kind=kind;row->owner=reinterpret_cast<uintptr_t>(&r);row->stream=reinterpret_cast<uintptr_t>(r.stream());row->thread=std::hash<std::thread::id>{}(std::this_thread::get_id());row->tokens=tokens;row->requests=count;running.fetch_add(1);counted=true;
  try {infini::rt::DeviceGuard guard(r.device().type(),d,true);deep_owner_r8900::ok(cudaGetDevice(&row->create_during));if(row->create_during!=d)throw std::logic_error("9193 create device");for(auto&e:ev){deep_owner_r8900::ok(cudaEventCreate(&e));++row->created;}mark(0);}catch(...){row->error=1;dispose();running.fetch_sub(1,std::memory_order_release);counted=false;throw;}
 }
 Frame(const Frame&)=delete;Frame&operator=(const Frame&)=delete;
 void mark(unsigned id){if(!row)return;r.assertOwnerThread();seq.mark(id);infini::rt::DeviceGuard guard(r.device().type(),row->device,true);int d=-1;deep_owner_r8900::ok(cudaGetDevice(&d));if(d!=row->device||deep_owner_r8900::active!=&deep_owner_r8900::pool[d][row->layer]||row->owner!=reinterpret_cast<uintptr_t>(&r)||row->stream!=reinterpret_cast<uintptr_t>(r.stream()))throw std::logic_error("9193 mark binding");row->wall[id]=deep_owner_r8900::wall();row->cpu[id]=deep_owner_r8900::cpu();deep_owner_r8900::ok(cudaEventRecord(ev[id],static_cast<cudaStream_t>(r.stream())));}
 void finish_after_original_drain(){if(!row)return;r.assertOwnerThread();seq.complete();row->final_drain_host_ms=1000*(deep_owner_r8900::wall()-row->wall[events-1]);infini::rt::DeviceGuard guard(r.device().type(),row->device,true);deep_owner_r8900::ok(cudaGetDevice(&row->query_during));if(row->query_during!=row->device)throw std::logic_error("9193 query device");for(size_t i=0;i<events;++i){deep_owner_r8900::ok(cudaEventQuery(ev[i]));++row->queried;float ms=0;deep_owner_r8900::ok(cudaEventElapsedTime(&ms,ev[0],ev[i]));row->gpu[i]=ms;if(!std::isfinite(ms)||ms<0||(i&&row->gpu[i]<row->gpu[i-1])||(i&&row->wall[i]<row->wall[i-1])||(i&&row->cpu[i]<row->cpu[i-1]))throw std::logic_error("9193 time order");}finished=true;}
 void dispose()noexcept{if(!row)return;try{infini::rt::DeviceGuard guard(r.device().type(),row->device,true);if(cudaGetDevice(&row->destroy_during)!=cudaSuccess||row->destroy_during!=row->device)row->error=2;for(auto&e:ev)if(e){if(cudaEventDestroy(e)==cudaSuccess)++row->destroyed;else row->error=3;e=nullptr;}}catch(...){row->error=4;}}
 ~Frame(){if(!row)return;dispose();row->done=finished&&row->error==0&&row->created==events&&row->queried==events&&row->destroyed==events;if(!row->done&&!row->error)row->error=5;if(counted)running.fetch_sub(1,std::memory_order_release);}
};
inline std::string read(){namespace m=infinilm::models::deepseek_v4;std::lock_guard<std::mutex>lock(m::wide_r3443::policy_mutex);if(mode.load()!=1||running.load(std::memory_order_acquire)||m::wide_r3443::in_flight||deep_owner_r8900::window.load())throw std::logic_error("9193 read requires completed CAPI + originalownerread");nlohmann::json result=nlohmann::json::array();for(int d=0;d<int(devices);++d)for(int l=0;l<int(layers);++l){const auto&row=rows[d][l];if(!row.done||row.error||row.created!=events||row.queried!=events||row.destroyed!=events||row.create_during!=d||row.query_during!=d||row.destroy_during!=d)throw std::logic_error("9193 incomplete/failed owner frame");nlohmann::json seg=nlohmann::json::array();for(size_t i=0;i+1<events;++i)seg.push_back({{"segment",i},{"start_ms",row.gpu[i]},{"end_ms",row.gpu[i+1]},{"stream_ms",row.gpu[i+1]-row.gpu[i]},{"host_wall_ms",1000*(row.wall[i+1]-row.wall[i])},{"thread_cpu_ms",1000*(row.cpu[i+1]-row.cpu[i])}});result.push_back({{"device",d},{"layer",l},{"kind",row.kind},{"owner",row.owner},{"stream",row.stream},{"thread",row.thread},{"tokens",row.tokens},{"requests",row.requests},{"events_created",row.created},{"events_queried",row.queried},{"events_destroyed",row.destroyed},{"create_during_device",row.create_during},{"query_during_device",row.query_during},{"destroy_during_device",row.destroy_during},{"whole_stream_ms",row.gpu[events-1]},{"original_final_drain_host_ms",row.final_drain_host_ms},{"segments",seg}});}mode.store(0);return nlohmann::json{{"schema",1},{"profile","attention-inner-r9193"},{"rows",result},{"GPU_handles_retained",0},{"scope","six sibling intervals within each owned Runtime; stream elapsed includes host launch gaps/queue/dependency/communication wait, not pure busy; do not sum across devices; outer input metadata validation and original Work destructor drain excluded"}}.dump();}
}
