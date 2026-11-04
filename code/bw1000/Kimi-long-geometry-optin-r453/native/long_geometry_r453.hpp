#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
namespace kimi_long_geometry_r453 {
constexpr size_t context=32768,logical=1,physical=2,pool=3,pages=128,blocks=260,page_size=256;
inline bool topology(bool optin,size_t tp,size_t pp){return optin?((tp==8&&pp==4)||(tp==32&&pp==1)):(tp==16&&pp==2);}
inline void require_topology(bool optin,size_t tp,size_t pp){if(!topology(optin,tp,pp))throw std::invalid_argument("r453 explicit long geometry");}
struct Geometry{size_t tp,pp,tp_nodes,heads,expert_i,shared_i,head_rows,head_bytes;};
inline Geometry geometry(bool optin,size_t tp,size_t pp){require_topology(optin,tp,pp);return{tp,pp,tp/8,96/tp,3072/tp,6144/tp,163840/tp,(163840/tp)*7168*2};}
// emitter_node denotes the one node-level engine returning sampled output;
// it does not designate eight independently emitting owner threads.
struct Rank{size_t node,device,tp_rank,stage,world_rank,begin,end,head_begin,head_end;bool embedding,head,emitter_node;};
// Same 93-layer partition used by the actual model and allocator. CPU checks
// compare all stages to the exact retained production partition header.
inline std::pair<size_t,size_t> layers(size_t pp,size_t stage){
 if(stage>=pp||(pp!=1&&pp!=2&&pp!=4))throw std::invalid_argument("r453 stage");
 if(pp==4){constexpr std::array<size_t,5>b{0,24,47,71,93};return{b[stage],b[stage+1]};}
 if(pp==1)return{0,93};return{stage==0?0u:46u,stage==0?46u:93u};
}
inline Rank rank(bool optin,size_t tp,size_t pp,size_t node,size_t device){
 const auto g=geometry(optin,tp,pp);if(node>=g.tp_nodes*pp||device>=8)throw std::invalid_argument("r453 node/local rank");
 const size_t stage=node/g.tp_nodes,tr=(node%g.tp_nodes)*8+device;const auto l=layers(pp,stage);
 return{node,device,tr,stage,stage*tp+tr,l.first,l.second,tr*g.head_rows,(tr+1)*g.head_rows,stage==0,stage+1==pp,node==(pp-1)*g.tp_nodes};
}
inline void call_scope(int kind,size_t count,int32_t row,int32_t past,size_t tokens,int flags){
 if(count!=1||row!=1||past<0||tokens==0||tokens>context||size_t(past)>context-tokens||flags!=3||
    (kind!=0&&kind!=1)||(kind==0&&tokens>512)||(kind==1&&(tokens!=1||past==0)))
  throw std::invalid_argument("r453 AR-only C1 prompt512/decode1 fixed32K");
}
struct HeadGeometry{size_t TP,rows,bytes;std::array<size_t,3>samples;};
inline HeadGeometry head_geometry(size_t tp){if(tp!=8&&tp!=16&&tp!=32)throw std::invalid_argument("r453 head TP");const size_t n=163840/tp;return{tp,n,n*7168*2,{0,n/2,n-1}};}
}
