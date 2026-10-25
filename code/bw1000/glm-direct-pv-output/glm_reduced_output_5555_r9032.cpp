// LOCAL PREPARED, NOT NATIVE COMPILED OR GPU RUN.
// Same arithmetic order as actual5390 r5096 per-row body; exact P_i only.
// No C ABI, Model dispatch, cache ownership or native runtime creation here.
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <optional>
#include <vector>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
#include <array>
#include <cstdint>
namespace {struct ReducedOutput9032{int enabled=0;uint64_t count[4]{};};thread_local std::array<ReducedOutput9032,8> reduced_output9032;}
extern "C" int glm_reduced_output_policy_r9032(int d,int enabled){if(d<0||d>=8||enabled<0||enabled>1)return-1;reduced_output9032[d].enabled=enabled;return 0;}
extern "C" int glm_reduced_output_counts_r9032(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return-1;for(size_t i=0;i<4;++i)out[i]=reduced_output9032[d].count[i];return 0;}
extern "C" int glm_reduced_output_choose_r9032(int d,int entry){auto&s=reduced_output9032.at(d);++s.count[2*entry+s.enabled];return s.enabled;}

namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
rt::Tensor sparse_attention_all_selected_decode_r601(rt::Runtime&,const rt::Tensor&,const rt::Tensor&,const rt::Tensor&,float);
namespace {
struct ExactRowsWork5555 {
 rt::Runtime&r;std::vector<rt::Tensor>held;
 ~ExactRowsWork5555(){try{r.syncStream();}catch(...){std::terminate();}}
};
}
rt::Tensor sparse_attention_exact_rows_r5555(rt::Runtime&r,
 const rt::Tensor&q,const std::vector<rt::Tensor>&keys,
 const std::vector<rt::Tensor>&values,int backend){
 if(backend<0||backend>2)throw std::invalid_argument("5555 backend:0actualr601,1shared,2rowSync");
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||q->ndim()!=3||
    (q->size(0)!=8&&q->size(0)!=16)||q->size(1)!=8||q->size(2)!=256||
    q->dtype()!=rt::DataType::kBFloat16||q->device()!=r.device()||!q->is_contiguous()||
    keys.size()!=q->size(0)||values.size()!=q->size(0))
  throw std::invalid_argument("5555 bounded exact-row query/list geometry");
 // Entire list validated before the first allocation/submit. No padded keys or
 // common maximum P is used in any Softmax dimension.
 for(size_t i=0;i<keys.size();++i){
  const auto&k=keys[i];const auto&v=values[i];
  if(!k||!v||k->ndim()!=3||!k->size(0)||k->size(0)>640||k->size(1)!=8||k->size(2)!=256||
     v->shape()!=k->shape()||k->dtype()!=rt::DataType::kBFloat16||v->dtype()!=k->dtype()||
     k->device()!=r.device()||v->device()!=r.device()||!k->is_contiguous()||!v->is_contiguous())
   throw std::invalid_argument("5555 exact contiguous perrow KV geometry");
 }
 ExactRowsWork5555 work{r};work.held.push_back(q);
 work.held.insert(work.held.end(),keys.begin(),keys.end());
 work.held.insert(work.held.end(),values.begin(),values.end());
 auto alloc=[&](rt::Shape s,rt::DataType dtype){auto t=rt::Tensor::empty(r,s,dtype);work.held.push_back(t);return t;};
 const size_t B=q->size(0);auto out=alloc({B,8,256},rt::DataType::kBFloat16);
 const bool direct_output9032=backend==1&&B==8&&glm_reduced_output_choose_r9032(r.device().index(),0);
 for(size_t i=0;i<B;++i){
  const size_t P=keys[i]->size(0);auto qr=q->narrow({{0,i,1}});work.held.push_back(qr);
  const auto&kr=keys[i];const auto&vr=values[i];
  if(backend==0){
   auto result=sparse_attention_all_selected_decode_r601(r,qr,kr,vr,1.f/16);
   work.held.push_back(result);out->narrow({{0,i,1}})->copy_from(r,result);continue;
  }
  auto dots=alloc({8,1,P},rt::DataType::kBFloat16),logits=alloc({8,1,P},rt::DataType::kFloat32),
       prob=alloc({8,1,P},rt::DataType::kFloat32),pb=alloc({8,1,P},rt::DataType::kBFloat16),
       result=direct_output9032?out->narrow({{0,i,1}}):alloc({1,8,256},rt::DataType::kBFloat16);
  if(direct_output9032)work.held.push_back(result);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
   qr->view({8,1,256}),kr->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},
   std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
  if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,8*P,1.f/16,r.stream()))
   throw std::runtime_error("5555 scale round");
  ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),
   logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
   pb,vr->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},
   std::optional<float>{0},std::optional<int>{},std::optional<int>{},result->view({8,1,256}));
  if(!direct_output9032)out->narrow({{0,i,1}})->copy_from(r,result);
  // backend0 calls the pinned actual5390 r601 symbol. backend2 is a
  // separate row-sync-factor diagnostic; backend1 shares only the end drain.
  if(backend==2)r.syncStream();
 }
 r.syncStream();return out;
}
}
