#pragma once
#include "glm_moe_dsa_attention_prepare.hpp"
#include "execution/submit.hpp"
#include <cstdlib>
#include <string>

namespace infinilm::models::glm_moe_dsa::v_on_demand_r398 {
inline bool requested() {
 const char* p=std::getenv("GLM_V_ON_DEMAND_R398");
 if(p && std::string(p)!="0" && std::string(p)!="1")
  throw std::invalid_argument("r398 requires explicit0/1");
 return p && std::string(p)=="1";
}
// Caller validates row/past, selection_map, owner and all mutually exclusive
// storage policies before reaching this producer. No persistent KV state here.
inline runtime::Tensor selected_values(AttentionPrepare& prepare,
 runtime::Runtime& r,const runtime::Tensor& latent,const runtime::Tensor& rotary,
 const runtime::Tensor& ids,std::vector<runtime::Tensor>& held) {
 namespace rt=runtime;namespace ex=execution;
 r.assertOwnerThread();
 if(!latent||!rotary||!ids||ids->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0))
  throw std::invalid_argument("r398 selected state");
 const size_t U=ids->size(0);
 auto l=rt::Tensor::empty(r,{U,512},rt::DataType::kBFloat16);
 auto k=rt::Tensor::empty(r,{U,64},rt::DataType::kBFloat16);
 held.insert(held.end(),{l,k});
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),latent,ids,int64_t{0},l);
 ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),rotary,ids,int64_t{0},k);
 auto result=prepare.expand_cached(r,l,k->view({U,1,64}));
 // Existing API computes both K and V; hold the unused K until owner drain.
 held.insert(held.end(),{result.first,result.second});
 return result.second;
}
}
