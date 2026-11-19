#pragma once
#include "glm_moe_dsa_attention_prepare.hpp"
#include "execution/submit.hpp"
#include "chunk_geometry_r400.hpp"
namespace infinilm::models::glm_moe_dsa::v_on_demand_r400 {
// HOST completed original expand_cached calls, not GPU events/kernel counts.
inline thread_local uint64_t completed_expand_calls=0;
inline runtime::Tensor selected_values(AttentionPrepare&prepare,runtime::Runtime&r,
 const runtime::Tensor&latent,const runtime::Tensor&rotary,const runtime::Tensor&ids,
 std::vector<runtime::Tensor>&held){
 namespace rt=runtime;namespace ex=execution;r.assertOwnerThread();
 if(!latent||!rotary||!ids||ids->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0)||ids->size(0)>32768)throw std::invalid_argument("r400 selected state");
 size_t U=ids->size(0);auto l=rt::Tensor::empty(r,{U,512},rt::DataType::kBFloat16),k=rt::Tensor::empty(r,{U,64},rt::DataType::kBFloat16);held.insert(held.end(),{l,k});
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),latent,ids,int64_t{0},l);
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),rotary,ids,int64_t{0},k);
 if(U<=max_block_rows){auto result=prepare.expand_cached(r,l,k->view({U,1,64}));++completed_expand_calls;held.insert(held.end(),{result.first,result.second});return result.second;}
 auto allV=rt::Tensor::empty(r,{U,prepare.local_heads(),256},rt::DataType::kBFloat16);held.push_back(allV);
 for_blocks(U,[&](size_t begin,size_t count){
  auto kv=prepare.expand_cached(r,l->narrow({{0,begin,count}}),k->narrow({{0,begin,count}})->view({count,1,64}));++completed_expand_calls;
  // Keep all chunk K/V until enclosing Work drain, including error unwind.
  // Source chunk survives asynchronous copy; unused K cost is not omitted.
  held.insert(held.end(),{kv.first,kv.second});allV->narrow({{0,begin,count}})->copy_from(r,kv.second);
 });
 return allV;
}
}
