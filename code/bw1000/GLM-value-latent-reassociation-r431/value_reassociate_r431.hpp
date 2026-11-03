#pragma once
#include "geometry_r431.hpp"
#include "glm_moe_dsa_attention_prepare.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <optional>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
namespace infinilm::models::glm_moe_dsa::value_r431 {
namespace rt=runtime;namespace ex=execution;
struct Counts{uint64_t chosen{},aggregation_Gemm_submits{},M8_expand_completed{},diagonal_copies{};};
inline thread_local Counts counts;
// Only C1/U2048. QK/scaling/softmax/probability casts preserve r601 statements.
// Approximation: raw BF16 latent is aggregated before BF16 rounding and a NEW
// M8 activation FP8 quantization; this is not finite-precision linear equality.
inline rt::Tensor attention(rt::Runtime&r,AttentionPrepare&prepare,const rt::Tensor&q,
 const rt::Tensor&k,const rt::Tensor&selected_latent,const rt::Tensor&rotary8,float scale){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!selected_latent||!rotary8||
  q->shape()!=rt::Shape{1,8,256}||k->shape()!=rt::Shape{2048,8,256}||selected_latent->shape()!=rt::Shape{2048,512}||
  rotary8->shape()!=rt::Shape{8,64}||prepare.local_heads()!=8||!std::isfinite(scale)||scale<=0)
  throw std::invalid_argument("r431 exact C1/U2048/TP8 value geometry");
 for(const auto&x:{q,k,selected_latent,rotary8})if(x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||!x->is_contiguous())throw std::invalid_argument("r431 owned contiguous BF16");
 struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}
  rt::Tensor alloc(rt::Shape shape){auto x=rt::Tensor::empty(r,shape,rt::DataType::kBFloat16);held.push_back(x);return x;}}w{r,{q,k,selected_latent,rotary8}};
 auto dots=w.alloc({8,1,2048});auto logits=rt::Tensor::empty(r,{8,1,2048},rt::DataType::kFloat32),prob=rt::Tensor::empty(r,{8,1,2048},rt::DataType::kFloat32);
 w.held.insert(w.held.end(),{logits,prob});auto pb=w.alloc({8,1,2048}),aggregate=w.alloc({8,512}),out=w.alloc({1,8,256});
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),q->view({8,1,256}),k->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
 if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,8*2048,scale,r.stream()))throw std::runtime_error("r431 original scale/round failed");
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
 // Explicit existing 2D GEMMs avoid assuming zero-stride batch broadcasts.
 for(size_t h=0;h<8;++h){auto p=pb->narrow({{0,h,1}})->view({1,2048}),a=aggregate->narrow({{0,h,1}});w.held.insert(w.held.end(),{p,a});
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),p,selected_latent,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},a);++counts.aggregation_Gemm_submits;}
 auto expanded=prepare.expand_cached(r,aggregate,rotary8->view({8,1,64}));w.held.insert(w.held.end(),{expanded.first,expanded.second});++counts.M8_expand_completed;
 if(expanded.second->shape()!=rt::Shape{8,8,256}||!expanded.second->is_contiguous())throw std::logic_error("r431 original expanded V layout");
 // Row h contains all eight heads from aggregate[h]. Only V[h,h,:] is used.
 // The K192+V256 interleave and global row128 scales are kept in expand_cached.
 auto rows=expanded.second->view({64,256});w.held.push_back(rows);
 for(size_t h=0;h<8;++h){auto row=glm_value_r431::diagonal_value_index(h,h,0)/256;auto src=rows->narrow({{0,row,1}})->view({1,1,256}),dst=out->narrow({{1,h,1}});w.held.insert(w.held.end(),{src,dst});dst->copy_from(r,src);++counts.diagonal_copies;}
 r.syncStream();++counts.chosen;return out;
}
}
