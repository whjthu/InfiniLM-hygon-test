// LOCAL PREPARED, NOT NATIVE COMPILED OR GPU RUN.
// Same arithmetic order as actual5390 r5096 per-row body; exact P_i only.
// No C ABI, Model dispatch, cache ownership or native runtime creation here.
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <optional>
#include <vector>
#include <array>
#include <cstdint>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
extern "C" int glm_fused_decode_r9026(void*,const void*,const void*const*,const void*const*,const size_t*,size_t,size_t,size_t,void*);
namespace {struct FusedState9026{int enabled=0;uint64_t ordinary=0,fused=0,errors=0;};thread_local std::array<FusedState9026,8> fused_states9026;}
extern "C" int glm_fused_decode_policy_r9026(int d,int enabled){if(d<0||d>=8||enabled<0||enabled>1)return-1;fused_states9026[d].enabled=enabled;return 0;}
extern "C" int glm_fused_decode_counts_r9026(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return-1;auto&s=fused_states9026[d];out[0]=s.ordinary;out[1]=s.fused;out[2]=s.errors;return 0;}

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
 auto&policy=fused_states9026.at(r.device().index());
 const bool eligible9026=backend==1&&B==8;
 if(eligible9026&&policy.enabled){
  std::array<const void*,8>kp{},vp{};std::array<size_t,8>pp{};
  for(size_t i=0;i<8;++i){kp[i]=keys[i]->data();vp[i]=values[i]->data();pp[i]=keys[i]->size(0);}
  int error=glm_fused_decode_r9026(out->data(),q->data(),kp.data(),vp.data(),pp.data(),B,8,256,r.stream());
  if(error){++policy.errors;throw std::runtime_error("9026 fused decode launch "+std::to_string(error));}
  ++policy.fused;r.syncStream();return out;
 }
 if(eligible9026)++policy.ordinary;

 for(size_t i=0;i<B;++i){
  const size_t P=keys[i]->size(0);auto qr=q->narrow({{0,i,1}});work.held.push_back(qr);
  const auto&kr=keys[i];const auto&vr=values[i];
  if(backend==0){
   auto result=sparse_attention_all_selected_decode_r601(r,qr,kr,vr,1.f/16);
   work.held.push_back(result);out->narrow({{0,i,1}})->copy_from(r,result);continue;
  }
  auto dots=alloc({8,1,P},rt::DataType::kBFloat16),logits=alloc({8,1,P},rt::DataType::kFloat32),
       prob=alloc({8,1,P},rt::DataType::kFloat32),pb=alloc({8,1,P},rt::DataType::kBFloat16),
       result=alloc({1,8,256},rt::DataType::kBFloat16);
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
  out->narrow({{0,i,1}})->copy_from(r,result);
  // backend0 calls the pinned actual5390 r601 symbol. backend2 is a
  // separate row-sync-factor diagnostic; backend1 shares only the end drain.
  if(backend==2)r.syncStream();
 }
 r.syncStream();return out;
}
}
