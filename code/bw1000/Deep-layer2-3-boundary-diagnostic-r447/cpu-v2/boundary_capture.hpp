#pragma once
// Private bounded diagnostic. No math replacement, GPU clone, new event or
// global device barrier. All reads run on the original Runtime owner, after
// the decoder's existing drain. Runtime memcpyD2H itself is completion-returning
// and adds per-copy staging/synchronization cost: this is not performance data.
#include "runtime/runtime.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace deep447 {
namespace rt = infinilm::runtime;
constexpr size_t owner_count=8, point_count=7, header_width=15;
struct Record { std::array<uint64_t,header_width> info{}; std::vector<unsigned char> bytes; };
struct Store { std::vector<Record> records; std::atomic<bool> active{false}; bool completed=false,failed=false; };
inline std::array<Store,owner_count> stores;
inline std::atomic<bool> armed{false},sealed{false};
inline uint64_t pass=0,output=0,forward=0;
inline std::mutex control;
struct Entry { rt::Runtime&r;size_t owner;unsigned mask=0; };
inline thread_local Entry* current=nullptr;

inline bool active() { for(auto&s:stores)if(s.active.load())return true;return false; }
inline int arm(int p,int o,uint64_t f) {
 if(p<0||p>1||(o!=82&&o!=83)||f!=uint64_t(9+o))return 1;
 std::lock_guard<std::mutex>guard(control);
 // Caller must separately prove original whole-engine quiescence; active=false
 // alone is not a GPU completion or a valid permission to change a live run.
 if(active()||armed.load()||sealed.load())return 2;
 for(auto&s:stores)if(!s.records.empty()||s.failed||s.completed)return 3;
 pass=uint64_t(p);output=uint64_t(o);forward=f;armed.store(true);return 0;
}

class Window {
 Entry state_;bool enabled_=false,done_=false;
public:
 Window(rt::Runtime&r,size_t requests,size_t past,size_t tokens):state_{r,size_t(r.device().index())} {
  if(!armed.load())return;
  r.assertOwnerThread();
  if(current||state_.owner>=owner_count||requests!=1||tokens!=1||past!=8488+output||forward!=9+output)
   throw std::logic_error("r447 actual T1/past/owner differs from frozen window");
  auto&s=stores[state_.owner];
  if(s.active.exchange(true))throw std::logic_error("r447 duplicate Runtime owner");
  if(s.completed||s.failed||!s.records.empty()){s.active.store(false);throw std::logic_error("r447 stale window");}
  current=&state_;enabled_=true;
 }
 Window(const Window&)=delete;Window&operator=(const Window&)=delete;
 void complete(){
  if(!enabled_)return;
  if(state_.mask!=127||stores[state_.owner].records.size()!=point_count)throw std::logic_error("r447 incomplete boundary coverage");
  stores[state_.owner].completed=true;done_=true;
 }
 ~Window()noexcept{
  if(!enabled_)return;
  current=nullptr;auto&s=stores[state_.owner];if(!done_)s.failed=true;s.active.store(false);
  // CPU byte records only. No Tensor destruction or Runtime call here. Original
  // TextModel Work/catch retains and drains all GPU handles on this owner.
 }
};

inline void save_point(rt::Runtime&r,size_t layer,int point,const rt::Tensor&t) {
 if(!current)return;
 if(point<0||point>=int(point_count)||layer!=(point==0?2:3))return;
 r.assertOwnerThread();auto&e=*current;
 if(&r!=&e.r||(e.mask&(1u<<point))||!t||t->device()!=r.device()||!t->is_contiguous()||t->dtype()!=rt::DataType::kBFloat16)
  throw std::logic_error("r447 duplicate/foreign/dtype boundary");
 const bool residual=point==0||point==1||point==4;
 if(t->shape()!=(residual?rt::Shape{1,4,4096}:rt::Shape{1,4096}))throw std::logic_error("r447 actual boundary shape differs");
 Record value;
 // pass,output,forward,owner,layer,point,dtype(1=BF16),ndim,d0,d1,d2,
 // bytes,Runtime pointer,stream pointer,opaque owner-thread hash.
 value.info={pass,output,forward,e.owner,layer,uint64_t(point),1,uint64_t(residual?3:2),1,uint64_t(residual?4:4096),uint64_t(residual?4096:1),t->nbytes(),
  reinterpret_cast<uint64_t>(&r),reinterpret_cast<uint64_t>(r.stream()),uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()))};
 if(t->nbytes()!=(residual?32768:8192))throw std::logic_error("r447 payload byte count differs");
 value.bytes.resize(t->nbytes());
 // These exact source tensors are still strongly held by original hidden/detail.
 // Layer2 exit and layer3 entry follow the same original layer2 drain; the five
 // detail fields follow original layer3 drain. No additional explicit sync call.
 r.memcpyD2H(value.bytes.data(),t->data(),value.bytes.size());
 stores[e.owner].records.push_back(std::move(value));e.mask|=1u<<point;
}

inline int seal() {
 std::lock_guard<std::mutex>guard(control);if(active()||!armed.load()||sealed.load())return 1;
 for(auto&s:stores)if(s.failed||!s.completed||s.records.size()!=point_count)return 2;
 armed.store(false);sealed.store(true);return 0;
}
inline int abandon() {
 std::lock_guard<std::mutex>guard(control);if(active())return 1;
 // Only after original whole-engine completion/reset. Does not manufacture a
 // complete window: partial/failed records remain marked and available to save.
 armed.store(false);sealed.store(true);return 0;
}
inline int clear() {
 std::lock_guard<std::mutex>guard(control);if(active()||armed.load())return 1;
 for(auto&s:stores){s.records.clear();s.completed=false;s.failed=false;}sealed.store(false);return 0;
}
inline uint64_t count(int owner) {
 if(owner<0||owner>=int(owner_count)||!sealed.load()||active())return UINT64_MAX;
 return stores[size_t(owner)].records.size();
}
inline int info(int owner,size_t point,uint64_t*out,size_t n) {
 if(!out||n!=header_width||count(owner)==UINT64_MAX||point>=stores[size_t(owner)].records.size())return 1;
 std::memcpy(out,stores[size_t(owner)].records[point].info.data(),n*sizeof(uint64_t));return 0;
}
inline int copy(int owner,size_t point,void*out,size_t n) {
 if(count(owner)==UINT64_MAX||point>=stores[size_t(owner)].records.size())return 1;
 const auto&v=stores[size_t(owner)].records[point].bytes;
 if(!out||n!=v.size())return 2;std::memcpy(out,v.data(),n);return 0;
}
}
// Same HOST TU as the real TextModel definitions: no second class/registry shim.
extern "C" int deep_boundary_arm_r447(int p,int o,uint64_t f){try{return deep447::arm(p,o,f);}catch(...){return 9;}}
extern "C" int deep_boundary_seal_r447(){try{return deep447::seal();}catch(...){return 9;}}
extern "C" int deep_boundary_abandon_r447(){try{return deep447::abandon();}catch(...){return 9;}}
extern "C" int deep_boundary_clear_r447(){try{return deep447::clear();}catch(...){return 9;}}
extern "C" uint64_t deep_boundary_count_r447(int o){try{return deep447::count(o);}catch(...){return UINT64_MAX;}}
extern "C" int deep_boundary_info_r447(int o,size_t p,uint64_t*out,size_t n){try{return deep447::info(o,p,out,n);}catch(...){return 9;}}
extern "C" int deep_boundary_copy_r447(int o,size_t p,void*out,size_t n){try{return deep447::copy(o,p,out,n);}catch(...){return 9;}}
