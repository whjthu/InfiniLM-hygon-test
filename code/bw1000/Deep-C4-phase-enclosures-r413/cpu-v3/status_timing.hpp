#pragma once
#include <array>
#include <cstdint>
#include <ctime>
#include <exception>
#include <thread>
#include <stdexcept>
#include <cassert>
namespace deep413 {
enum class Site:unsigned {Query=0,Gather=1,Scores=2,Store=3,Key=4};
struct Stats {std::array<uint64_t,5> calls{},errors{},wall_ns{},cpu_ns{};std::array<uint64_t,5> phase_calls{},phase_errors{},phase_wall_ns{},phase_cpu_ns{};uint64_t clock_reads=0;bool clocks_valid=true;};
struct State {bool enabled=false;Stats stats{};};
inline thread_local State state{};
inline uint64_t clock_ns(clockid_t kind) noexcept {timespec t{};++state.stats.clock_reads;if(clock_gettime(kind,&t)!=0){state.stats.clocks_valid=false;return 0;}return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);}
struct Sample {
 unsigned site;bool enabled;int exceptions;uint64_t wall=0,cpu=0;
 explicit Sample(Site s) noexcept:site(unsigned(s)),enabled(state.enabled),exceptions(std::uncaught_exceptions()){
  ++state.stats.calls[site];if(enabled){wall=clock_ns(CLOCK_MONOTONIC);cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);}
 }
 ~Sample() noexcept {if(std::uncaught_exceptions()>exceptions)++state.stats.errors[site];if(enabled){auto c=clock_ns(CLOCK_THREAD_CPUTIME_ID),w=clock_ns(CLOCK_MONOTONIC);if(w<wall||c<cpu)state.stats.clocks_valid=false;else{state.stats.wall_ns[site]+=w-wall;state.stats.cpu_ns[site]+=c-cpu;}}}
};
enum class Phase:unsigned {QueryProjectionRope=0,FeaturesQueryStatus=1,CacheGather=2,ScoresStatus=3,TPsumTopKComplete=4};
struct PhaseSample {
 unsigned phase=0;bool active=false,timed=false;int exceptions=0;uint64_t wall=0,cpu=0;
 PhaseSample() noexcept=default;
 explicit PhaseSample(Phase p) noexcept{begin(p);}
 void begin(Phase p) noexcept{if(active)return;phase=unsigned(p);active=true;timed=state.enabled;exceptions=std::uncaught_exceptions();++state.stats.phase_calls[phase];if(timed){wall=clock_ns(CLOCK_MONOTONIC);cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);}}
 void finish() noexcept{if(!active)return;active=false;if(std::uncaught_exceptions()>exceptions)++state.stats.phase_errors[phase];if(timed){auto c=clock_ns(CLOCK_THREAD_CPUTIME_ID),w=clock_ns(CLOCK_MONOTONIC);if(w<wall||c<cpu)state.stats.clocks_valid=false;else{state.stats.phase_wall_ns[phase]+=w-wall;state.stats.phase_cpu_ns[phase]+=c-cpu;}}}
 ~PhaseSample() noexcept{finish();}
 PhaseSample(const PhaseSample&)=delete;PhaseSample&operator=(const PhaseSample&)=delete;
};
template<class F>void status_boundary(Site s,F&& original_check){Sample sample(s);original_check();}
template<class F>Stats owner_phase(bool enabled,F&& forward){
 if(state.enabled)throw std::logic_error("Nested status timing phase");state.stats={};state.enabled=enabled;
 try{forward();state.enabled=false;return state.stats;}catch(...){state.enabled=false;throw;}
}
inline int CPU_contract(){
 std::array<bool,8> done{};std::array<std::thread,8> workers;
 for(unsigned d=0;d<8;++d)workers[d]=std::thread([&,d]{
  assert(!state.enabled&&state.stats.clock_reads==0);
  for(int mode:{0,1,0,1,0}){
   unsigned original_checks=0,original_syncs=0;
   auto q=owner_phase(mode,[&]{
    {PhaseSample p(Phase::QueryProjectionRope);++original_syncs;}
    {PhaseSample p(Phase::FeaturesQueryStatus);status_boundary(Site::Query,[&]{++original_checks;++original_syncs;});}
    {PhaseSample p(Phase::CacheGather);status_boundary(Site::Gather,[&]{++original_checks;++original_syncs;});}
    {PhaseSample p(Phase::ScoresStatus);status_boundary(Site::Scores,[&]{++original_checks;++original_syncs;});}
    {PhaseSample p(Phase::TPsumTopKComplete);++original_syncs;}
   });
   assert(original_checks==3&&original_syncs==5&&q.calls[0]==1&&q.calls[1]==1&&q.calls[2]==1&&q.calls[3]==0&&q.calls[4]==0);
   assert((q.phase_calls==std::array<uint64_t,5>({1,1,1,1,1})));
   assert(q.clocks_valid&&q.clock_reads==(mode?32:0)&&!state.enabled);
  }
  bool failed=false;
  try{owner_phase(true,[&]{PhaseSample p(Phase::FeaturesQueryStatus);status_boundary(Site::Query,[]{throw std::invalid_argument("DeepSeek C4 indexer GPU validation status=2");});});}
  catch(const std::invalid_argument&e){failed=std::string(e.what())=="DeepSeek C4 indexer GPU validation status=2";}
  assert(failed&&!state.enabled&&state.stats.errors[0]==1&&state.stats.phase_errors[1]==1);
  auto restored=owner_phase(false,[]{PhaseSample p(Phase::ScoresStatus);status_boundary(Site::Scores,[]{});});assert(restored.clock_reads==0&&restored.errors[2]==0&&restored.phase_errors[3]==0);
  done[d]=true;
 });
 for(auto&t:workers)t.join();for(bool x:done)assert(x);assert(!state.enabled&&state.stats.clock_reads==0);return 0;
}
}
