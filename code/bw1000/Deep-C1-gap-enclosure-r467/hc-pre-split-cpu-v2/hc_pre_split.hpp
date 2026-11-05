#pragma once
#include "deep_owner_events_r8900.hpp"
#include <pthread.h>
#include <cmath>
namespace hc_pre_split_r467 {
using RT=infinilm::runtime::Runtime;
struct Row{bool claimed=false,done=false;uintptr_t owner=0,stream=0,thread=0;unsigned seen=0,created=0,queried=0,destroyed=0;std::array<double,6>gpu{},wall{},cpu{};double drain_ms=0;};
inline std::array<std::array<std::array<Row,2>,43>,8>rows{};
struct Frame{
 RT&r;Row*row=nullptr;std::array<cudaEvent_t,6>ev{};bool finished=false;
 Frame(RT&runtime,size_t t):r(runtime){
  if(!deep_owner_r8900::window.load())return;
  r.assertOwnerThread();auto*a=deep_owner_r8900::active;int d=r.device().index();if(t!=1||!a||d<0||d>=8||a->owner!=&r||a->stream!=r.stream()||r.isGraphRecording())throw std::logic_error("HCsplit requires originalT1ownerstream");
  int l=-1;for(int i=0;i<43;++i)if(a==&deep_owner_r8900::pool[d][i])l=i;if(l<0)throw std::logic_error("HCsplit layer missing");auto&pair=rows[d][l];row=pair[0].claimed?&pair[1]:&pair[0];if(row->claimed)throw std::logic_error("HCsplit thirdHCcall");row->claimed=true;row->owner=reinterpret_cast<uintptr_t>(&r);row->stream=reinterpret_cast<uintptr_t>(r.stream());row->thread=static_cast<uintptr_t>(pthread_self());
  try{infini::rt::DeviceGuard g(r.device().type(),d,true);for(auto&e:ev){deep_owner_r8900::ok(cudaEventCreate(&e));++row->created;}mark(0);}catch(...){dispose();throw;}
 }
 void mark(unsigned i){if(!row)return;r.assertOwnerThread();if(i>=6||row->seen!=i||row->owner!=reinterpret_cast<uintptr_t>(&r)||row->stream!=reinterpret_cast<uintptr_t>(r.stream()))throw std::logic_error("HCsplit mark owner/order");infini::rt::DeviceGuard g(r.device().type(),r.device().index(),true);row->wall[i]=deep_owner_r8900::wall();row->cpu[i]=deep_owner_r8900::cpu();deep_owner_r8900::ok(cudaEventRecord(ev[i],static_cast<cudaStream_t>(r.stream())));++row->seen;}
 void after_original_drain(){if(!row)return;if(row->seen!=6)throw std::logic_error("HCsplit incomplete marks");row->drain_ms=1000*(deep_owner_r8900::wall()-row->wall[5]);infini::rt::DeviceGuard g(r.device().type(),r.device().index(),true);for(unsigned i=0;i<6;++i){deep_owner_r8900::ok(cudaEventQuery(ev[i]));float ms=0;deep_owner_r8900::ok(cudaEventElapsedTime(&ms,ev[0],ev[i]));if(!std::isfinite(ms)||ms<0||(i&&ms<row->gpu[i-1]))throw std::logic_error("HCsplit event order");row->gpu[i]=ms;++row->queried;}finished=true;}
 void dispose()noexcept{if(!row)return;try{infini::rt::DeviceGuard g(r.device().type(),r.device().index(),true);for(auto&e:ev)if(e){if(cudaEventDestroy(e)==cudaSuccess)++row->destroyed;e=nullptr;}}catch(...){}row->done=finished&&row->created==6&&row->queried==6&&row->destroyed==6;}
 ~Frame(){dispose();}
};
inline std::string read(){
 if(deep_owner_r8900::window.load())throw std::logic_error("HCsplit read after originalownerread");nlohmann::json result=nlohmann::json::array();
 for(unsigned d=0;d<8;++d)for(unsigned l=0;l<43;++l)for(unsigned hc=0;hc<2;++hc){const auto&q=rows[d][l][hc];if(!q.claimed||!q.done||q.seen!=6)throw std::logic_error("HCsplit missing ownerHC");auto spans=nlohmann::json::array();for(unsigned i=0;i<5;++i)spans.push_back({{"stage",i},{"stream_ms",q.gpu[i+1]-q.gpu[i]},{"host_wall_ms",1000*(q.wall[i+1]-q.wall[i])},{"thread_cpu_ms",1000*(q.cpu[i+1]-q.cpu[i])}});result.push_back({{"device",d},{"layer",l},{"HC",hc},{"owner",q.owner},{"stream",q.stream},{"thread",q.thread},{"created",q.created},{"queried",q.queried},{"destroyed",q.destroyed},{"whole_stream_ms",q.gpu[5]},{"original_drain_host_ms",q.drain_ms},{"spans",spans}});}
 std::string out=nlohmann::json{{"schema",1},{"profile","HCpre-r467"},{"labels",{"cast_square_mean_normalize","mixes_GEMM","affine_sigmoid_softmax","sinkhorn","weighted_collapse_cast"}},{"rows",result},{"scope","Streamelapsed includes launch/dependency gaps; threadCPU includes driver/spin. No new sync; read after originalHC and decoder drains. Eventcreate/query/destroy overhead outside sibling intervals but included callerwall."}}.dump();for(auto&rank:rows)for(auto&layer:rank)for(auto&row:layer)row=Row{};return out;
}
}
