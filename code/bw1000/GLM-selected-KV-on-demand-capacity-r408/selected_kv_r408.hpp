#pragma once
#include "selected_values_r400.hpp"
#include "policy_r408.hpp"
namespace infinilm::models::glm_moe_dsa::kv_on_demand_r408 {
struct Counts {uint64_t completed_expand_calls=0,K_copy_calls=0,K_copy_bytes=0;};
inline thread_local Counts counts{};
// Shared byte-for-byte state-copy implementation for Model caller and component.
inline void append_latent_rotary(runtime::Runtime&r,const runtime::Tensor&latent,const runtime::Tensor&rotary,const runtime::Tensor&fresh_l,const runtime::Tensor&fresh_r,size_t past,size_t count){latent->narrow({{0,past,count}})->copy_from(r,fresh_l);rotary->narrow({{0,past,count}})->copy_from(r,fresh_r->view({count,64}));}
inline void store_projected(runtime::Runtime&r,const runtime::Tensor&key,const runtime::Tensor&value,const runtime::Tensor&fresh_k,const runtime::Tensor&fresh_v,size_t past,size_t count){if(key)key->narrow({{0,past,count}})->copy_from(r,fresh_k);if(value)value->narrow({{0,past,count}})->copy_from(r,fresh_v);}
inline std::pair<runtime::Tensor,runtime::Tensor> selected(AttentionPrepare&prepare,runtime::Runtime&r,
 const runtime::Tensor&latent,const runtime::Tensor&rotary,const runtime::Tensor&ids,std::vector<runtime::Tensor>&held,bool keepK=true){
 namespace rt=runtime;namespace ex=execution;r.assertOwnerThread();
 if(!keepK){auto before=v_on_demand_r400::completed_expand_calls;auto v=v_on_demand_r400::selected_values(prepare,r,latent,rotary,ids,held);counts.completed_expand_calls+=v_on_demand_r400::completed_expand_calls-before;return {rt::Tensor{},v};}
 if(!latent||!rotary||!ids||ids->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0)||ids->size(0)>32768)throw std::invalid_argument("r408 selected state");
 size_t U=ids->size(0);auto l=rt::Tensor::empty(r,{U,512},rt::DataType::kBFloat16),k=rt::Tensor::empty(r,{U,64},rt::DataType::kBFloat16);held.insert(held.end(),{l,k});
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),latent,ids,int64_t{0},l);
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),rotary,ids,int64_t{0},k);
 if(U<=v_on_demand_r400::max_block_rows){auto result=prepare.expand_cached(r,l,k->view({U,1,64}));++counts.completed_expand_calls;held.insert(held.end(),{result.first,result.second});return result;}
 auto allK=rt::Tensor::empty(r,{U,prepare.local_heads(),256},rt::DataType::kBFloat16),allV=rt::Tensor::empty(r,{U,prepare.local_heads(),256},rt::DataType::kBFloat16);held.insert(held.end(),{allK,allV});
 v_on_demand_r400::for_blocks(U,[&](size_t begin,size_t count){
  auto kv=prepare.expand_cached(r,l->narrow({{0,begin,count}}),k->narrow({{0,begin,count}})->view({count,1,64}));++counts.completed_expand_calls;
  // Both chunk sources and both contiguous destinations are retained before copy.
  held.insert(held.end(),{kv.first,kv.second});allK->narrow({{0,begin,count}})->copy_from(r,kv.first);++counts.K_copy_calls;counts.K_copy_bytes+=count*prepare.local_heads()*256*2;
  allV->narrow({{0,begin,count}})->copy_from(r,kv.second);
 });return {allK,allV};
}
}
