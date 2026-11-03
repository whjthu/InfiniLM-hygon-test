#pragma once
#include <sched.h>
#include <array>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>
namespace kimi_r437_owner {
inline std::vector<int> mask(){cpu_set_t s;CPU_ZERO(&s);if(sched_getaffinity(0,sizeof(s),&s))throw std::runtime_error("437 sched_getaffinity");std::vector<int>v;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&s))v.push_back(c);return v;}
inline std::vector<int> selected(const std::vector<int>&entry,int rank){if(entry.size()<32||rank<0||rank>=8||!std::is_sorted(entry.begin(),entry.end())||std::adjacent_find(entry.begin(),entry.end())!=entry.end())throw std::invalid_argument("437 eight width4 owners require sorted unique entry32+");return {entry.begin()+rank*4,entry.begin()+rank*4+4};}
inline void set(const std::vector<int>&v){cpu_set_t s;CPU_ZERO(&s);for(int c:v){if(c<0||c>=CPU_SETSIZE)throw std::invalid_argument("437 CPU id outside cpu_set_t");CPU_SET(c,&s);}if(sched_setaffinity(0,sizeof(s),&s))throw std::runtime_error("437 sched_setaffinity");}
struct Pin {
 std::vector<int> before,wanted;bool active=false;bool*restored;
 Pin(const std::vector<int>&entry,int rank,bool*status=nullptr):before(mask()),wanted(selected(entry,rank)),restored(status){if(before!=entry)throw std::runtime_error("437 owner must inherit complete entry mask before pin");set(wanted);active=true;if(mask()!=wanted){restore();throw std::runtime_error("437 owner pin verification");}}
 void restore(){if(active){set(before);if(mask()!=before)throw std::runtime_error("437 owner restore verification");active=false;if(restored)*restored=true;}}
 ~Pin()noexcept{if(active)try{restore();}catch(...){if(restored)*restored=false;}}
};
struct Record{std::vector<int>entry,owner,after;bool boundary=false,restored=false,injected=false;};
inline std::array<Record,8> check(bool inject){const auto entry=mask();if(entry.size()<32)throw std::invalid_argument("437 owner preflight needs actual32 CPUs");std::array<Record,8>rows;std::array<std::exception_ptr,8>errors;std::vector<std::thread>workers;
 for(int rank=0;rank<8;++rank)workers.emplace_back([&,rank]{auto&v=rows[rank];v.entry=mask();try{try{Pin pin(entry,rank,&v.restored);v.owner=mask();v.boundary=v.owner==selected(entry,rank);if(inject&&rank==3){v.injected=true;throw std::runtime_error("437 injected recording FFI leaf error");}pin.restore();}catch(const std::runtime_error&){if(!(inject&&rank==3&&v.injected))throw;}v.after=mask();if(!v.boundary||!v.restored||v.after!=entry)throw std::runtime_error("437 pin/FFI-boundary/restore failed");}catch(...){errors[rank]=std::current_exception();}});
 for(auto&t:workers)t.join();std::vector<int>used;for(const auto&v:rows)used.insert(used.end(),v.owner.begin(),v.owner.end());std::sort(used.begin(),used.end());if(used.size()!=32||std::adjacent_find(used.begin(),used.end())!=used.end())throw std::runtime_error("437 owners not32 unique CPUs");for(auto&e:errors)if(e)std::rethrow_exception(e);if(mask()!=entry)throw std::runtime_error("437 entry thread mask changed");return rows;}
}
