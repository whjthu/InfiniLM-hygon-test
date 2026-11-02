#pragma once
#include <array>
#include <cstdint>
#include <ctime>
#include <exception>
#include <thread>
#include <stdexcept>
#include <cassert>
namespace deep406 {
enum class Site:unsigned {Query=0,Gather=1,Scores=2,Store=3,Key=4};
struct Stats {std::array<uint64_t,5> calls{},errors{},wall_ns{},cpu_ns{};uint64_t clock_reads=0;bool clocks_valid=true;};
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
template<class F>void status_boundary(Site s,F&& original_check){Sample sample(s);original_check();}
template<class F>Stats owner_phase(bool enabled,F&& forward){
 if(state.enabled)throw std::logic_error("Nested status timing phase");state.stats={};state.enabled=enabled;
 try{forward();state.enabled=false;return state.stats;}catch(...){state.enabled=false;throw;}
}
inline int CPU_contract(){
 std::array<Stats,8> last{};std::array<bool,8> failure{};std::array<std::thread,8> workers;
 for(unsigned d=0;d<8;++d)workers[d]=std::thread([&,d]{
  assert(!state.enabled&&state.stats.clock_reads==0);
  for(int mode:{0,0,0,1,0}){
   unsigned memcpy_calls=0,sync_calls=0;
   auto q=owner_phase(mode,[&]{for(auto s:{Site::Query,Site::Gather,Site::Scores})status_boundary(s,[&]{++memcpy_calls;++sync_calls;});});
   assert(memcpy_calls==3&&sync_calls==3&&q.calls[0]==1&&q.calls[1]==1&&q.calls[2]==1&&q.calls[3]==0&&q.calls[4]==0);
   assert(q.clocks_valid&&q.clock_reads==(mode?12:0)&&!state.enabled);last[d]=q;
  }
  try{owner_phase(true,[&]{status_boundary(Site::Query,[&]{throw std::invalid_argument("DeepSeek C4 indexer GPU validation status=2");});});}
  catch(const std::invalid_argument&e){failure[d]=std::string(e.what())=="DeepSeek C4 indexer GPU validation status=2";}
  assert(failure[d]&&!state.enabled&&state.stats.errors[0]==1);
  auto restored=owner_phase(false,[]{status_boundary(Site::Query,[]{});});assert(restored.clock_reads==0&&restored.errors[0]==0);
 });
 for(auto&t:workers)t.join();for(unsigned d=0;d<8;++d)assert(failure[d]&&last[d].clock_reads==0);
 assert(!state.enabled&&state.stats.clock_reads==0);return 0;
}
}
