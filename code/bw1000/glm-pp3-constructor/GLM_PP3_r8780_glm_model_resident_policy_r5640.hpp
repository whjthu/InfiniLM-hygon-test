#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
namespace glm_model_resident_r5640 {
constexpr size_t logical_rows=64,active_rows=64,capacity=640,heads=8,dim=256;
constexpr uint64_t bank_bytes=2ull*active_rows*capacity*heads*dim*2;
constexpr uint64_t reserve_bytes=6ull*1024*1024*1024;
// Additional explicit allowance, NOT proof of the existing Model's phase peak.
// Direct contiguous slot-prefix views require no complete-prefix packing.
constexpr uint64_t new_workspace_allowance=64ull*1024*1024;
using Mask=std::array<bool,78>;
inline Mask mask(const uint8_t*p,size_t count,size_t begin,size_t end){
 if(!p||count!=78||!((begin==0&&end==26)||(begin==26&&end==54)||(begin==54&&end==78)))throw std::invalid_argument("5640 mask TP8PP2 partition");
 Mask result{};size_t n=0;
 for(size_t i=0;i<78;++i){if(p[i]>1||(p[i]&&(i<begin||i>=end)))throw std::invalid_argument("5640 global layer mask/foreign stage");result[i]=p[i]!=0;n+=result[i];}
 if(n>end-begin)throw std::invalid_argument("5640 stage resident layer maximum");
 return result;
}
inline size_t selected(const Mask&m){size_t n=0;for(bool x:m)n+=x;return n;}
inline void admit(uint64_t free,uint64_t additional){
 if(additional>free||free-additional<reserve_bytes+new_workspace_allowance)throw std::invalid_argument("5640 resident reserve before allocation");
}
inline size_t slot_offset_elements(size_t slot){if(slot>=active_rows)throw std::invalid_argument("5640 slot range");return slot*capacity*heads*dim;}
inline size_t prefix_elements(size_t prefix){if(!prefix||prefix>capacity)throw std::invalid_argument("5640 prefix range");return prefix*heads*dim;}
struct Request{size_t row,past,tokens;};
class Slots {
 std::thread::id owner_=std::this_thread::get_id();
 std::array<int,logical_rows>slot_{};std::array<size_t,logical_rows>length_{};
public:
 Slots(){slot_.fill(-1);}
 void owner()const{if(owner_!=std::this_thread::get_id())throw std::invalid_argument("5640 metadata foreign owner");}
 void reset()noexcept{slot_.fill(-1);length_.fill(0);}
 bool empty()const{owner();for(int s:slot_)if(s>=0)return false;return true;}
 size_t active()const{owner();size_t n=0;for(int s:slot_)n+=s>=0;return n;}
 int slot(size_t row)const{owner();if(row>=logical_rows||slot_[row]<0)throw std::invalid_argument("5640 missing slot");return slot_[row];}
 size_t length(size_t row)const{owner();if(row>=logical_rows)throw std::invalid_argument("5640 row range");return length_[row];}
 Slots propose(const std::vector<Request>&requests)const{
  owner();if(requests.empty()||requests.size()>8)throw std::invalid_argument("5640 cohort must be1..8");
  std::array<bool,logical_rows>seen{};size_t fresh=0;
  // Validate WHOLE list before even the unpublished proposal changes.
  for(auto q:requests){if(q.row>=logical_rows||seen[q.row]||!q.tokens||q.past>capacity||q.tokens>capacity-q.past)throw std::invalid_argument("5640 row/past/tokens");seen[q.row]=true;
   if(slot_[q.row]<0){if(q.past)throw std::invalid_argument("5640 missing history");++fresh;}
   else if(q.past!=length_[q.row])throw std::invalid_argument("5640 stale position");}
  if(active()+fresh>active_rows)throw std::invalid_argument("5640 active64 overflow;finished slots retained until reset");
  Slots next=*this;std::array<bool,active_rows>used{};for(int s:slot_)if(s>=0)used[size_t(s)]=true;
  for(auto q:requests){if(next.slot_[q.row]<0){size_t s=0;while(used[s])++s;next.slot_[q.row]=int(s);used[s]=true;}next.length_[q.row]=q.past+q.tokens;}
  return next;
 }
};
} // namespace glm_model_resident_r5640
