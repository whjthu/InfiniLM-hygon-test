#pragma once
#include "router_math.hpp"
#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <optional>
extern "C" int deep_router_suffix_launch_r419(const void*,const void*,const void*,const void*,void*,void*,void*,void*,void*);
namespace infinilm::models::deepseek_v4::router_suffix_r419 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
// Keep the original full-forward intermediates alive through the expert/TP tail.
inline void gate_project(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&gate,const rt::Tensor&logits){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),x,gate->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);}
struct RouterBuffers{rt::Tensor soft,scores,indices,selected_weights,sum,denominator,normalized,weights;};
inline RouterBuffers router_buffers(rt::Runtime&r,size_t tokens,size_t experts,bool fused){
 RouterBuffers b;if(!fused){b.soft=rt::Tensor::empty(r,{tokens,experts},rt::DataType::kFloat32);b.scores=rt::Tensor::empty(r,{tokens,experts},rt::DataType::kFloat32);}return b;
}
inline void router_suffix(rt::Runtime&r,const rt::Tensor&logits,const rt::Tensor&bias,const rt::Tensor&table,const rt::Tensor&token_ids,const rt::Tensor&epsilon,const rt::Tensor&factor,size_t vocab,bool hash,size_t topk,bool fused,RouterBuffers&b,rt::Tensor debug_scores={},rt::Tensor debug_selection={}){
 r.assertOwnerThread();for(auto&t:{debug_scores,debug_selection})if(t&&(t->device()!=r.device()||t->dtype()!=rt::DataType::kFloat32||t->shape()!=logits->shape()||!t->is_contiguous()))throw std::invalid_argument("r419 debugshape/owner differs");const size_t tokens=logits->size(0),experts=logits->size(1);const auto device_=r.device();
 auto alloc=[&](rt::Shape shape,rt::DataType type){return rt::Tensor::empty(r,shape,type);};
 if(fused){
  if(!deep419::eligible({tokens,experts,topk,hash,logits->dtype()==rt::DataType::kFloat32,bias&&bias->dtype()==rt::DataType::kFloat32,epsilon->dtype()==rt::DataType::kFloat32&&factor->dtype()==rt::DataType::kFloat32,true,logits->is_contiguous()&&bias&&bias->is_contiguous()}))throw std::invalid_argument("r419 fusedgeometry differs");
  for(auto&t:{logits,bias,epsilon,factor})if(!t||t->device()!=device_)throw std::invalid_argument("r419 fusedowner differs");
  if(bias->shape()!=rt::Shape{256}||epsilon->shape()!=rt::Shape{1}||factor->shape()!=rt::Shape{1})throw std::invalid_argument("r419 scalar/bias shape differs");
  b.indices=alloc({1,6},rt::DataType::kInt64);b.weights=alloc({1,6},rt::DataType::kFloat32);
  infini::rt::DeviceGuard guard{device_.type(),device_.index(),true};
  int code=deep_router_suffix_launch_r419(logits->data(),bias->data(),epsilon->data(),factor->data(),const_cast<std::byte*>(b.weights->data()),const_cast<std::byte*>(b.indices->data()),debug_scores?const_cast<std::byte*>(debug_scores->data()):nullptr,debug_selection?const_cast<std::byte*>(debug_selection->data()):nullptr,r.stream());
  if(code)throw std::runtime_error("r419 router launch status="+std::to_string(code));return;
 }
 ex::submit<infini::ops::Softplus>(r,infini::ops::Softplus::DefaultConfig(device_.type()),logits,double{1},double{20},b.soft);
 ex::submit<infini::ops::Sqrt>(r,infini::ops::Sqrt::DefaultConfig(device_.type()),b.soft,b.scores);
 b.indices=alloc({tokens,topk},rt::DataType::kInt64);b.selected_weights=alloc({tokens,topk},rt::DataType::kFloat32);
 if(hash){
  if(!token_ids||token_ids->device()!=device_||token_ids->dtype()!=rt::DataType::kInt64||token_ids->shape()!=rt::Shape{tokens})throw std::invalid_argument("DeepSeek hash routing requires packed token IDs");
  std::vector<int64_t>ids(tokens);r.syncStream();r.memcpyD2H(ids.data(),token_ids->data(),tokens*sizeof(int64_t));for(auto id:ids)if(id<0||size_t(id)>=vocab)throw std::invalid_argument("DeepSeek token ID outside hash table");
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),table,token_ids,int64_t{0},b.indices);
 }else{
  auto selection=alloc({tokens,experts},rt::DataType::kFloat32),values=alloc({tokens,topk},rt::DataType::kFloat32);
  ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),b.scores,bias,selection);
  ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(device_.type()),selection,int64_t(topk),int64_t{1},true,true,values,b.indices);
  if(debug_selection)debug_selection->copy_from(r,selection);
 }
 ex::submit<infini::ops::Gather>(r,infini::ops::Gather::DefaultConfig(device_.type()),b.scores,b.indices,int64_t{1},false,b.selected_weights);
 b.sum=alloc({tokens,1},rt::DataType::kFloat32);b.denominator=alloc({tokens,1},rt::DataType::kFloat32);b.normalized=alloc({tokens,topk},rt::DataType::kFloat32);b.weights=alloc({tokens,topk},rt::DataType::kFloat32);
 ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),b.selected_weights,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},b.sum);
 ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),b.sum,epsilon,b.denominator);
 ex::submit<infini::ops::Div>(r,infini::ops::Div::DefaultConfig(device_.type()),b.selected_weights,b.denominator,b.normalized);
 ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),b.normalized,factor,b.weights);
 if(debug_scores)debug_scores->copy_from(r,b.scores);
}
}
