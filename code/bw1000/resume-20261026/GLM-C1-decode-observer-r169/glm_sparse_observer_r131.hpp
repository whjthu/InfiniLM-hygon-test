#pragma once
#include <cuda_runtime_api.h>
#include "glm_decode_selector_r169.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
// This namespace owns no model tensors and changes no Context layout.
namespace glm_sparse_r131 {
using Clock=std::chrono::steady_clock;
struct Selector { uint64_t sequence=0,tokens=0;int stage=0,rank=0;std::vector<std::array<size_t,3>> rows; };
struct Mark {std::string name;size_t event=0;double host_s=0;};
struct State {
 std::array<cudaEvent_t,512> events{};bool prepared=false,enabled=false,active=false,detail=false;
 uint64_t sequence=0,decode1_seen=0,matched[5]{};size_t used=0;int stage=-1,rank=-1;size_t current_layer=~size_t(0),full_layer=~size_t(0),shared_layer=~size_t(0);
 Selector selector;std::vector<Mark> marks;nlohmann::json samples=nlohmann::json::array();
};
inline thread_local std::unordered_map<void*,std::unique_ptr<State>> states;
inline thread_local State*current=nullptr;
inline bool detailing(){return current&&current->detail;}
inline State&get(void*owner){auto it=states.find(owner);if(it==states.end())throw std::logic_error("r131 observer not prepared");return *it->second;}
inline void check(cudaError_t e,const char*what){if(e!=cudaSuccess)throw std::runtime_error(what);}
inline void check_current_device(int expected){int d=-1;check(cudaGetDevice(&d),"r131 current device query");if(d!=expected)throw std::invalid_argument("r131 current device mismatch");}
inline void prepare(void*owner,int stage,int rank){
 if(states.count(owner))throw std::logic_error("r131 duplicate prepare");auto s=std::make_unique<State>();s->stage=stage;s->rank=rank;
 try{for(auto&e:s->events)check(cudaEventCreate(&e),"r131 event create");}catch(...){for(auto e:s->events)if(e)cudaEventDestroy(e);throw;}
 s->prepared=true;states.emplace(owner,std::move(s));
}
inline void policy(void*owner,bool enabled){auto&s=get(owner);if(s.active)throw std::logic_error("r131 active policy");s.enabled=enabled;s.sequence=0;s.decode1_seen=0;for(auto&n:s.matched)n=0;s.samples=nlohmann::json::array();}
inline void mark(cudaStream_t stream,const char*name){if(!current||!current->active)return;auto&s=*current;if(s.used==s.events.size())throw std::logic_error("r131 event bound exceeded");auto n=s.used++;check(cudaEventRecord(s.events[n],stream),"r131 event record");s.marks.push_back({name,n,std::chrono::duration<double>(Clock::now().time_since_epoch()).count()});}
inline void begin(void*owner,size_t tokens,size_t count,const size_t*rows,cudaStream_t stream){
 auto it=states.find(owner);if(it==states.end())return;auto&s=*it->second;if(s.active||current)throw std::logic_error("r131 nested forward");++s.sequence;if(!s.enabled)return;
 // R169 observes ONLY actual single-row decode, after sixteen eligible warm decode calls.
 int kind=glm_decode_r169::capture(tokens,count,rows,s.decode1_seen)?4:-1;
 if(kind<0||s.matched[kind]>=3u)return;++s.matched[kind];s.active=true;s.used=0;s.marks.clear();s.selector={s.sequence,tokens,s.stage,s.rank,{}};for(size_t i=0;i<count;++i)s.selector.rows.push_back({rows[3*i],rows[3*i+1],rows[3*i+2]});s.full_layer=s.shared_layer=~size_t(0);s.current_layer=~size_t(0);current=&s;mark(stream,"CAPI.owner.begin");
}
inline void layer_begin(cudaStream_t stream,size_t layer,bool full){if(!current)return;auto&s=*current;s.current_layer=layer;if(full&&s.full_layer==~size_t(0))s.full_layer=layer;if(!full&&s.shared_layer==~size_t(0))s.shared_layer=layer;s.detail=layer==s.full_layer||layer==s.shared_layer;std::string name="layer."+std::to_string(layer)+(full?".full.begin":".shared.begin");mark(stream,name.c_str());}
inline void layer_end(cudaStream_t stream,size_t layer){if(!current)return;std::string name="layer."+std::to_string(layer)+".end";mark(stream,name.c_str());current->detail=false;}
inline void detail(cudaStream_t stream,const char*name){if(!current||!current->detail)return;std::string n="detail.layer."+std::to_string(current->current_layer)+"."+name;mark(stream,n.c_str());}
// Called only AFTER the original forward's existing owner drain. No synchronization API here.
inline void finish_after_original_drain(void*owner,cudaStream_t stream){
 auto it=states.find(owner);if(it==states.end()||!it->second->active)return;auto&s=*it->second;
 nlohmann::json j={{"selector_revision","r169"},{"selector","decode_T1"},{"eligible_decode1_index",s.decode1_seen},{"sequence",s.selector.sequence},{"tokens",s.selector.tokens},{"stage",s.selector.stage},{"rank",s.selector.rank},{"row_past_count",s.selector.rows},{"clock_scope","one owner original stream; GPU event intervals overlap host spans"},{"event_count",s.used},{"full_detail_layer",s.full_layer},{"shared_detail_layer",s.shared_layer},{"marks",nlohmann::json::array()}};
 for(auto&m:s.marks){check(cudaEventQuery(s.events[m.event]),"r131 existing drain did not complete recorded events");float ms=0;check(cudaEventElapsedTime(&ms,s.events[0],s.events[m.event]),"r131 event elapsed");j["marks"].push_back({{"name",m.name},{"owner_host_s",m.host_s},{"gpu_from_first_s",double(ms)/1000.}});}
 s.samples.push_back(std::move(j));s.active=false;s.detail=false;current=nullptr;
}
inline void abort_after_original_drain(void*owner){auto it=states.find(owner);if(it!=states.end()){it->second->active=false;it->second->detail=false;}current=nullptr;}
inline nlohmann::json read(void*owner){auto&s=get(owner);if(s.active)throw std::logic_error("r131 active read");return {{"enabled",s.enabled},{"sequence",s.sequence},{"selector_revision","r169"},{"selector_names",{"decode_T16_DISABLED","decode_T15_DISABLED","prefix_B8_DISABLED","prefix_B1_DISABLED","decode_T1"}},{"decode1_seen",s.decode1_seen},{"coverage",{s.matched[0],s.matched[1],s.matched[2],s.matched[3],s.matched[4]}},{"required",{0,0,0,0,3}},{"missing_is_UNKNOWN",true},{"events_allocated",512},{"samples",s.samples}};}
inline void close_after_original_drain(void*owner){auto it=states.find(owner);if(it==states.end())return;if(it->second->active)throw std::logic_error("r131 active close");for(auto&e:it->second->events){check(cudaEventDestroy(e),"r131 event destroy");e=nullptr;}states.erase(it);}
}
