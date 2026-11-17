#pragma once
#include "selected_values_r400.hpp"
namespace infinilm::models::glm_moe_dsa::history_value_only_v1 {
// Dedicated complete-prefix <=8192 caller only. Original helper stays unchanged.
inline runtime::Tensor selected_values(AttentionPrepare&prepare,runtime::Runtime&r,
 const runtime::Tensor&latent,const runtime::Tensor&rotary,const runtime::Tensor&ids,
 std::vector<runtime::Tensor>&held){
 namespace rt=runtime;namespace ex=execution;r.assertOwnerThread();
 if(!latent||!rotary||!ids||ids->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0)||ids->size(0)>32768)throw std::invalid_argument("r400 selected state");
 if(ids->size(0)>v_on_demand_r400::max_block_rows)throw std::invalid_argument("history value-only requires one original block");
 size_t U=ids->size(0);auto l=rt::Tensor::empty(r,{U,512},rt::DataType::kBFloat16),k=rt::Tensor::empty(r,{U,64},rt::DataType::kBFloat16);held.insert(held.end(),{l,k});
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),latent,ids,int64_t{0},l);
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),rotary,ids,int64_t{0},k);
 auto value=prepare.expand_cached_values(r,l,k->view({U,1,64}));
 ++v_on_demand_r400::completed_expand_calls;held.push_back(value);return value;
}
}
