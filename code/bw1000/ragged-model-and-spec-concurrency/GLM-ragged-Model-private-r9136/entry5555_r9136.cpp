#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <memory>
#include <thread>
#include <string>
#include <cstring>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <vector>
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


namespace { struct AttentionCohort9054 {int enabled=0;uint64_t counts[4]{};};thread_local std::array<AttentionCohort9054,8> attention_cohort9054; }
extern "C" int glm_attention_cohort_policy_r9054(int d,int mode){if(d<0||d>=8||mode<0||mode>1)return -1;attention_cohort9054[d].enabled=mode;return 0;}
extern "C" int glm_attention_cohort_counts_r9054(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return -1;for(size_t j=0;j<4;++j)out[j]=attention_cohort9054[d].counts[j];return 0;}
extern "C" int glm_attention_cohort_choose_r9054(int d,int entry,int eligible){if(d<0||d>=8||entry<0||entry>1)return 0;auto&s=attention_cohort9054[d];int selected=eligible&&s.enabled;++s.counts[entry*2+selected];return selected;}
namespace {struct RaggedState9136{int mode=0;uint64_t calls[4]{};uint64_t byB[17][4]{};};thread_local RaggedState9136 ragged_state9136[8];
void record9136(int d,size_t B,int branch){++ragged_state9136[d].calls[branch];++ragged_state9136[d].byB[B][branch];}}
extern "C" int glm_ragged_policy_r9136(int d,int on){if(d<0||d>=8||on<0||on>1)return -1;ragged_state9136[d].mode=on;return 0;}
extern "C" int glm_ragged_enabled_r9136(int d){return d>=0&&d<8?ragged_state9136[d].mode:0;}
extern "C" int glm_ragged_counts_r9136(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return -1;for(size_t i=0;i<4;++i)out[i]=ragged_state9136[d].calls[i];return 0;}
extern "C" int glm_ragged_Bcounts_r9136(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=68)return -1;for(size_t B=0;B<17;++B)for(size_t j=0;j<4;++j)out[B*4+j]=ragged_state9136[d].byB[B][j];return 0;}
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
 const int mode=glm_ragged_enabled_r9136(r.device().index());
 if(backend<0||backend>2)throw std::invalid_argument("5555 backend:0actualr601,1shared,2rowSync");
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||q->ndim()!=3||
    (q->size(0)<1||q->size(0)>16)||q->size(1)!=8||q->size(2)!=256||
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
 if(B==1){auto one=sparse_attention_all_selected_decode_r601(r,q,keys[0],values[0],1.f/16);work.held.push_back(one);record9136(r.device().index(),B,3);return one;}

 // Default OFF. Only equal-P cohorts; original per-row QK/PV shapes/providers.
 const size_t cohortP=keys[0]->size(0);bool equalP=true;for(const auto&key:keys)equalP=equalP&&key->size(0)==cohortP;const bool cohortEligible=backend==1&&equalP&&(B==8||B==16);
 if(glm_attention_cohort_choose_r9054(r.device().index(),0,cohortEligible)) {
  auto cdots=alloc({B*8,1,cohortP},rt::DataType::kBFloat16),clogits=alloc({B*8,1,cohortP},rt::DataType::kFloat32),cprob=alloc({B*8,1,cohortP},rt::DataType::kFloat32),cpb=alloc({B*8,1,cohortP},rt::DataType::kBFloat16),cresults=alloc({B,8,256},rt::DataType::kBFloat16);
  std::vector<rt::Tensor> qrows,krows,vrows,drows,prows,rrows;
  for(size_t i=0;i<B;++i) {
   auto qr=q->narrow({{0,i,1}});auto kr=keys[i],vr=values[i];
   qrows.push_back(qr->view({8,1,256}));krows.push_back(kr->permute({1,2,0}));vrows.push_back(vr->permute({1,0,2}));drows.push_back(cdots->narrow({{0,i*8,8}}));prows.push_back(cpb->narrow({{0,i*8,8}}));rrows.push_back(cresults->narrow({{0,i,1}})->view({8,1,256}));
   work.held.insert(work.held.end(),{qrows.back(),krows.back(),vrows.back(),drows.back(),prows.back(),rrows.back()});
   if(drows.back()->shape()!=rt::Shape({8,1,cohortP})||prows.back()->shape()!=rt::Shape({8,1,cohortP})||!drows.back()->is_contiguous()||!prows.back()->is_contiguous()||reinterpret_cast<uintptr_t>(drows.back()->data())%16||reinterpret_cast<uintptr_t>(prows.back()->data())%16)throw std::logic_error("9054 cohort view layout");
  }
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qrows[i],krows[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},drows[i]);
  if(glm_scale_round_r392(cdots->data(),clogits->data(),nullptr,B*8*cohortP,1.f/16,r.stream()))throw std::runtime_error("9054 original scale round");
  ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),clogits,int64_t{2},std::optional<infini::ops::DataType>{},cprob);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),cprob,cpb);
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),prows[i],vrows[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},rrows[i]);
  out->copy_from(r,cresults);
  r.syncStream();record9136(r.device().index(),B,2);return out;
 }

 // Private default-OFF ragged path. Exact P_i Softmax bounds; no padding.
 if(mode==1&&backend==1&&!equalP&&B>=2&&B<=16){
  size_t L=0;std::vector<size_t>offsets;for(const auto&key:keys){offsets.push_back(L);L+=8*key->size(0);}
  auto dots=alloc({L},rt::DataType::kBFloat16),logits=alloc({L},rt::DataType::kFloat32),prob=alloc({L},rt::DataType::kFloat32),pb=alloc({L},rt::DataType::kBFloat16),results=alloc({B,8,256},rt::DataType::kBFloat16);
  std::vector<rt::Tensor>qr,kr,vr,dr,lr,pr,br,rr;
  for(size_t i=0;i<B;++i){size_t P=keys[i]->size(0),off=offsets[i],n=8*P;
   qr.push_back(q->narrow({{0,i,1}})->view({8,1,256}));kr.push_back(keys[i]->permute({1,2,0}));vr.push_back(values[i]->permute({1,0,2}));
   dr.push_back(dots->narrow({{0,off,n}})->view({8,1,P}));lr.push_back(logits->narrow({{0,off,n}})->view({8,1,P}));pr.push_back(prob->narrow({{0,off,n}})->view({8,1,P}));br.push_back(pb->narrow({{0,off,n}})->view({8,1,P}));rr.push_back(results->narrow({{0,i,1}})->view({8,1,256}));
   for(auto&t:{qr.back(),kr.back(),vr.back(),dr.back(),lr.back(),pr.back(),br.back(),rr.back()})work.held.push_back(t);
   if(!dr.back()->is_contiguous()||!br.back()->is_contiguous()||reinterpret_cast<uintptr_t>(dr.back()->data())%16||reinterpret_cast<uintptr_t>(br.back()->data())%16)throw std::logic_error("9135 ragged segment alignment/layout");
  }
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qr[i],kr[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dr[i]);
  if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,L,1.f/16,r.stream()))throw std::runtime_error("9135 original flat scale round");
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),lr[i],int64_t{2},std::optional<infini::ops::DataType>{},pr[i]);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),br[i],vr[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},rr[i]);
  out->copy_from(r,results);r.syncStream();record9136(r.device().index(),B,1);return out;
 }
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
 r.syncStream();record9136(r.device().index(),B,0);return out;
}
}

