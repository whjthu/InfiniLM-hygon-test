// Private fixed-window observer. No Tensor/KV owners, new streams or synchronizations.
#include "glm_model_window_r5797.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <string>
#include <thread>
namespace glm_window_r5795 {
using Clock=std::chrono::steady_clock;
struct Stamp{const char*name=nullptr;Clock::time_point wall;double cpu=0;};
struct State{rt::Runtime*runtime=nullptr;std::thread::id owner;std::array<cudaEvent_t,512>events{};std::array<Stamp,512>stamps{};size_t used=0;bool armed=false,active=false,pending=false,aborted=false;std::array<size_t,8>rows{},pasts{},counts{};unsigned matched=0;int round=-1,cohort=-1,stage=-1,rank=-1;std::string result;};
thread_local State s;
bool armed()noexcept{return s.armed;}
bool request_matches(size_t i,size_t row,size_t past,size_t count)noexcept{bool ok=s.armed&&i<8&&row==size_t(s.cohort*8)+i&&past==(s.round==0?size_t(0):size_t(512+s.round-1))&&count==(s.round==0?size_t(512):size_t(1));if(ok){s.rows[i]=row;s.pasts[i]=past;s.counts[i]=count;s.matched|=1u<<i;}return ok;}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
double cpu(){timespec t{};if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))throw std::runtime_error("5795 threadCPU");return double(t.tv_sec)+double(t.tv_nsec)*1e-9;}
void owner(rt::Runtime&r){r.assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=r.device().index()||r.isGraphRecording())throw std::invalid_argument("5795 owner/device/eager");cudaStreamCaptureStatus cap;ck(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&cap));if(cap!=cudaStreamCaptureStatusNone)throw std::invalid_argument("5795 capture");if(s.runtime&&(s.runtime!=&r||s.owner!=std::this_thread::get_id()))throw std::invalid_argument("5795 Runtime identity");}
void prepare(rt::Runtime&r){owner(r);if(s.active||s.pending||s.armed)throw std::logic_error("5797 prepare pending");s.runtime=&r;s.owner=std::this_thread::get_id();try{for(auto&e:s.events)if(!e)ck(cudaEventCreate(&e));}catch(...){for(auto&e:s.events)if(e){cudaEventDestroy(e);e=nullptr;}throw;}}
void arm(rt::Runtime&r,int round,int cohort,bool enabled){owner(r);if(s.active||s.pending||s.armed)throw std::logic_error("5795 pending arm");if(!enabled)return;if((round!=0&&(round<63||round>65))||cohort<0||cohort>1)throw std::invalid_argument("5795 fixed window");if(s.runtime!=&r||!s.events.back())throw std::logic_error("5797 prepare pool before decode");s.round=round;s.cohort=cohort;s.used=0;s.matched=0;s.aborted=false;s.armed=true;}
bool mark(rt::Runtime&r,const char*name){if(!s.active)return false;owner(r);if(s.used==s.events.size())throw std::runtime_error("5795 bounded marker overflow");ck(cudaEventRecord(s.events[s.used],static_cast<cudaStream_t>(r.stream())));s.stamps[s.used]={name,Clock::now(),cpu()};++s.used;return true;}
bool enter(rt::Runtime&r,size_t layer,int stage,int rank,bool valid){if(!s.armed||layer!=size_t(s.round==0?s.cohort:38))return false;owner(r);if(stage!=(s.round==0?0:1)||rank<0||rank>=8||!valid||s.matched!=255)throw std::invalid_argument("5795 selected geometry");s.stage=stage;s.rank=rank;s.armed=false;s.active=true;s.pending=true;mark(r,"wholelayer.begin");return true;}
void finish(rt::Runtime&r){owner(r);if(!s.active)throw std::logic_error("5795 finish");mark(r,"wholelayer.end");s.active=false;}
void abort_noexcept()noexcept{s.active=false;s.armed=false;s.aborted=true;}
const char*read_after_CAPI(rt::Runtime&r){owner(r);if(s.active||s.armed||!s.pending||s.aborted||s.used<2)throw std::logic_error("5795 unavailable/aborted sample");
 // Query only: caller MUST have returned from the original Model CAPI drain.
 for(size_t i=0;i<s.used;++i)ck(cudaEventQuery(s.events[i]));
 nlohmann::json intervals=nlohmann::json::array();for(size_t i=1;i<s.used;++i){float ms=0;ck(cudaEventElapsedTime(&ms,s.events[i-1],s.events[i]));intervals.push_back({{"from",s.stamps[i-1].name},{"to",s.stamps[i].name},{"stream_timeline_ms",ms},{"host_wall_ms",std::chrono::duration<double,std::milli>(s.stamps[i].wall-s.stamps[i-1].wall).count()},{"threadCPU_ms",1000*(s.stamps[i].cpu-s.stamps[i-1].cpu)}});}
 float whole=0;ck(cudaEventElapsedTime(&whole,s.events[0],s.events[s.used-1]));s.result=nlohmann::json({{"mode",s.round==0?"prefill-round0":"decode-round63to65"},{"observed_rows",s.rows},{"observed_pasts",s.pasts},{"observed_counts",s.counts},{"matched_requests",s.matched},{"layer",s.round==0?s.cohort:38},{"stage",s.stage},{"rank",s.rank},{"round",s.round},{"cohort",s.cohort},{"past",s.round==0?0:512+s.round-1},{"Runtime_identity",reinterpret_cast<uintptr_t>(&r)},{"marker_count",s.used},{"whole_stream_timeline_ms",whole},{"whole_host_wall_ms",std::chrono::duration<double,std::milli>(s.stamps[s.used-1].wall-s.stamps[0].wall).count()},{"intervals",intervals},{"scope","Same Runtime stream; queue/dependency/enqueue gaps included, not GPUbusy. Adjacent intervals only; nested group totals must not be added."}}).dump();s.pending=false;s.used=0;return s.result.c_str();}
void close_after_drain(rt::Runtime&r){owner(r);if(s.active)throw std::logic_error("5795 active close");for(auto&e:s.events)if(e){ck(cudaEventDestroy(e));e=nullptr;}s=State{};}
}
