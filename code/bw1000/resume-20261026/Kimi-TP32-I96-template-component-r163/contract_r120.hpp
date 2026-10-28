#pragma once
#include <cstddef>
#include <stdexcept>
namespace r120 {
constexpr size_t local_workers=8,tp_nodes=4,tp_size=32,hidden=7168,routed=3584,global_i=3072,local_i=96,experts=896,topk=16;
static_assert(local_workers*tp_nodes==tp_size && global_i/tp_size==local_i && local_i%32==0);
static_assert(local_i*tp_size==3072 && (local_i/2)*tp_size==1536 && (local_i/32)*tp_size==96);
inline void topology(int node,int workers,int tp,int rank,int pp,int stage,int world,int world_rank){
 if(node<0||node>=4||workers!=8||tp!=32||rank<node*8||rank>=(node+1)*8||pp!=1||stage!=0||world!=32||world_rank!=rank)throw std::invalid_argument("r120 actual32owner topology");
}
inline bool shape(size_t t){return t==8||t==16||t==64;}
}
