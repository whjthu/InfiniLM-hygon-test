#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace freeze416 {
constexpr size_t slots=2124,page_slots=64,page_bytes=8448,raw_bytes=128*page_bytes;
constexpr size_t payload_bytes=slots*(128+4),state_bytes=8*512*2;
struct Segment{size_t raw,packed,bytes;};
// Only valid prefix key/scale bytes, never invalid page padding.
inline constexpr std::array<Segment,3>segments{{{0,0,33*page_bytes},{33*page_bytes,33*page_bytes,12*128},{33*page_bytes+8192,33*page_bytes+12*128,12*4}}};
struct Protocol {
 bool frozen=false;uintptr_t owner=0;uint64_t captures=0,restores=0;
 void boundary(uintptr_t actual_owner,size_t past,bool ready,bool poisoned)const{
  if(!actual_owner||past!=8496||!ready||poisoned||(frozen&&owner!=actual_owner))throw std::invalid_argument("r416 freeze owner/boundary differs");
 }
 void capture(uintptr_t actual_owner,size_t past,bool ready,bool poisoned){boundary(actual_owner,past,ready,poisoned);if(frozen)throw std::logic_error("r416 snapshot already frozen");owner=actual_owner;frozen=true;++captures;}
 void restore(uintptr_t actual_owner,size_t past,bool ready,bool poisoned){boundary(actual_owner,past,ready,poisoned);if(!frozen)throw std::logic_error("r416 restore without snapshot");++restores;}
};
inline int CPU_contract(){
 std::vector<uint8_t>raw(raw_bytes,0xa5),snapshot(payload_bytes),state(state_bytes),saved(state_bytes),restored(raw_bytes,0x39);
 for(size_t i=0;i<raw.size();++i)raw[i]=uint8_t(i*17+3);for(size_t i=0;i<state.size();++i)state[i]=uint8_t(i*11+7);
 Protocol p;bool rejected=false;try{p.restore(7,8496,true,false);}catch(...){rejected=true;}if(!rejected)throw std::logic_error("r416 unfrozen restore accepted");
 p.capture(7,8496,true,false);for(auto s:segments)std::memcpy(snapshot.data()+s.packed,raw.data()+s.raw,s.bytes);saved=state;
 for(size_t trial=0;trial<10;++trial){
  // Real prefix metadata is regenerated, revision is never restored by this protocol.
  const size_t revision=100+trial*10,revision_before=revision;p.restore(7,8496,true,false);
  std::vector<uint8_t>before=restored;std::fill(state.begin(),state.end(),0);
  for(auto s:segments)std::memcpy(restored.data()+s.raw,snapshot.data()+s.packed,s.bytes);state=saved;
  for(size_t i=0;i<restored.size();++i){bool valid=false;for(auto s:segments)valid|=i>=s.raw&&i<s.raw+s.bytes;if(restored[i]!=(valid?raw[i]:before[i]))throw std::logic_error("r416 padding or validpayload corrupted");}
  if(state!=saved||revision!=revision_before)throw std::logic_error("r416 state/revision changed");
 }
 for(int n=0;n<5;++n){rejected=false;try{if(n==0)p.capture(7,8496,true,false);else p.restore(n==1?8:7,n==2?8497:8496,n!=3,n==4);}catch(...){rejected=true;}if(!rejected)throw std::logic_error("r416 invalid freeze boundary accepted");}
 if(p.captures!=1||p.restores!=10)throw std::logic_error("r416 freeze protocol count differs");return 0;
}
}
