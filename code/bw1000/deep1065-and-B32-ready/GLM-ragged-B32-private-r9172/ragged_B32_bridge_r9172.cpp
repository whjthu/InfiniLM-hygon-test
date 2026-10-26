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
namespace {struct RaggedState9136{int mode=0;uint64_t calls[4]{};uint64_t byB[33][4]{};};thread_local RaggedState9136 ragged_state9136[8];
void record9136(int d,size_t B,int branch){if(d<0||d>=8||B<1||B>32||branch<0||branch>3)throw std::logic_error("9172 branch boundary");++ragged_state9136[d].calls[branch];++ragged_state9136[d].byB[B][branch];}}
extern "C" int glm_ragged_policy_r9136(int d,int on){if(d<0||d>=8||on<0||on>1)return -1;ragged_state9136[d].mode=on;return 0;}
extern "C" int glm_ragged_enabled_r9136(int d){return d>=0&&d<8?ragged_state9136[d].mode:0;}
extern "C" int glm_ragged_counts_r9136(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return -1;for(size_t i=0;i<4;++i)out[i]=ragged_state9136[d].calls[i];return 0;}
extern "C" int glm_ragged_Bcounts_r9136(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=132)return -1;for(size_t B=0;B<33;++B)for(size_t j=0;j<4;++j)out[B*4+j]=ragged_state9136[d].byB[B][j];return 0;}
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
    (q->size(0)<1||q->size(0)>32)||q->size(1)!=8||q->size(2)!=256||
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
 if(mode==1&&backend==1&&!equalP&&B>=2&&B<=32){
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


namespace rt9135=infinilm::runtime;
namespace {thread_local std::string err9135,report9135;
struct Context9135{std::shared_ptr<rt9135::Runtime>r;int device;bool poison=false;uint64_t counts[4]{};};
void ck9135(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void owner9135(Context9135*c){if(!c||!c->r||c->poison)throw std::invalid_argument("9135 context state");c->r->assertOwnerThread();int d=-1;ck9135(cudaGetDevice(&d));if(d!=c->device||c->r->isGraphRecording())throw std::invalid_argument("9135 device/recording");cudaStreamCaptureStatus st;ck9135(cudaStreamIsCapturing(static_cast<cudaStream_t>(c->r->stream()),&st));if(st!=cudaStreamCaptureStatusNone)throw std::invalid_argument("9135 captured stream");}
void ptr9135(Context9135*c,const void*p){if(!p)throw std::invalid_argument("9135 null");cudaPointerAttributes a{};ck9135(cudaPointerGetAttributes(&a,p));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("9135 pointer type/device");}
void drain9135(Context9135*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
bool overlap9135(const void*a,size_t an,const void*b,size_t bn){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+bn&&y<x+an;}
}
extern "C" const char*glm_ragged_error_r9135(){return err9135.c_str();}
extern "C" const char*glm_ragged_report_r9135(){return report9135.c_str();}
extern "C" void*glm_ragged_create_r9135(int d){try{if(d<0||d>7)throw std::invalid_argument("9135 device");infini::rt::DeviceGuard g{rt9135::Device::Type::kHygon,d,true};auto c=std::make_unique<Context9135>();c->device=d;c->r=rt9135::Runtime::create({rt9135::Device::Type::kHygon,d});c->r->enableOwnerDrainFailStop();return c.release();}catch(const std::exception&e){err9135=e.what();return nullptr;}}
extern "C" int glm_ragged_run_r9135(void*raw,size_t B,const size_t*desc,const void*q,const void*k,const void*v,void*out,int mode){auto*c=static_cast<Context9135*>(raw);bool submitted=false;try{
 owner9135(c);if(!desc||B<1||B>32||(mode!=0&&mode!=1))throw std::invalid_argument("9172 B/mode/metadata");std::array<bool,32>seen{};std::vector<size_t>P,rows;bool equal=true;size_t sum=0;for(size_t i=0;i<B;++i){size_t row=desc[2*i],n=desc[2*i+1];if(row>=32||seen[row]||!n||n>640)throw std::invalid_argument("9172 distinct resident row/P");seen[row]=true;rows.push_back(row);P.push_back(n);sum+=n;equal=equal&&n==P[0];}for(auto p:{q,k,v,static_cast<const void*>(out)})ptr9135(c,p);for(auto z:{std::make_pair(q,B*4096),std::make_pair(k,size_t(32*640*4096)),std::make_pair(v,size_t(32*640*4096))})if(overlap9135(out,B*4096,z.first,z.second))throw std::invalid_argument("9172 output alias");
 const auto dev=c->r->device();auto qt=rt9135::Tensor::from_blob(const_cast<void*>(q),{B,8,256},rt9135::DataType::kBFloat16,dev),kt=rt9135::Tensor::from_blob(const_cast<void*>(k),{32,640,8,256},rt9135::DataType::kBFloat16,dev),vt=rt9135::Tensor::from_blob(const_cast<void*>(v),{32,640,8,256},rt9135::DataType::kBFloat16,dev),ot=rt9135::Tensor::from_blob(out,{B,8,256},rt9135::DataType::kBFloat16,dev);std::vector<rt9135::Tensor>keys,values;for(size_t i=0;i<B;++i){keys.push_back(kt->narrow({{0,rows[i],1}})->view({640,8,256})->narrow({{0,0,P[i]}}));values.push_back(vt->narrow({{0,rows[i],1}})->view({640,8,256})->narrow({{0,0,P[i]}}));}
 uint64_t before[4],after[4],hist[132];glm_ragged_counts_r9136(c->device,before,4);glm_ragged_policy_r9136(c->device,mode);glm_reduced_output_policy_r9032(c->device,0);glm_attention_cohort_policy_r9054(c->device,1);auto start=std::chrono::steady_clock::now();submitted=true;auto result=infinilm::models::glm_moe_dsa::sparse_attention_exact_rows_r5555(*c->r,qt,keys,values,1);ot->copy_from(*c->r,result);c->r->syncStream();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();glm_ragged_counts_r9136(c->device,after,4);glm_ragged_Bcounts_r9136(c->device,hist,132);size_t route=4;unsigned hits=0;std::vector<uint64_t>delta(4);for(size_t i=0;i<4;++i){delta[i]=after[i]-before[i];hits+=delta[i];if(delta[i])route=i;}if(hits!=1)throw std::logic_error("9172 actualbranch");report9135=nlohmann::json{{"B",B},{"P",P},{"logical_rows",rows},{"mode",mode},{"route",route},{"actual_branch_delta",delta},{"counts",std::vector<uint64_t>(after,after+4)},{"Bhist",std::vector<uint64_t>(hist,hist+132)},{"whole_attention_native_ms",ms},{"model_weights_loaded",false},{"source_scope","syntheticuniqueQKV_actualservicePmetadata_residentbank32x640"}}.dump();return 0;
 }catch(const std::exception&e){err9135=e.what();drain9135(c);if(c&&submitted)c->poison=true;return 1;}}
extern "C" int glm_ragged_destroy_r9135(void*raw,int*expired){auto*c=static_cast<Context9135*>(raw);try{if(!c||!c->r||!expired)throw std::invalid_argument("9135 destroy");c->r->assertOwnerThread();std::weak_ptr<rt9135::Runtime>w=c->r;drain9135(c);c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){err9135=e.what();return 1;}}


#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <stdexcept>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8> direct_calls_r601{};}
// Internal contract: selection_map_r302(count=1) has already built sorted,
// unique causal K/V. Its mapped IDs cover the whole dense range exactly once
// after deduplication, so repeating IndexSelect is an identity copy.
rt::Tensor sparse_attention_all_selected_decode_r601(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,float scale){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!v||q->ndim()!=3||k->ndim()!=3||v->shape()!=k->shape()||q->size(0)!=1||!q->size(1)||q->size(1)>128||q->size(2)!=256||!k->size(0)||k->size(0)>2048||k->size(1)!=q->size(1)||k->size(2)!=256||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||!q->is_contiguous()||!k->is_contiguous()||!v->is_contiguous()||!std::isfinite(scale)||scale<=0)throw std::invalid_argument("GLM all-selected decode requires validated contiguous causal KV");
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dtype){auto x=rt::Tensor::empty(r,shape,dtype);held.push_back(x);return x;};
 try{
  size_t heads=q->size(1),count=k->size(0);auto dots=alloc({heads,1,count},q->dtype()),logits=alloc({heads,1,count},rt::DataType::kFloat32),prob=alloc({heads,1,count},rt::DataType::kFloat32),pb=alloc({heads,1,count},q->dtype()),out=alloc({1,heads,256},q->dtype());
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),q->view({heads,1,256}),k->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
  if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,heads*count,scale,r.stream()))throw std::runtime_error("GLM direct scale/round failed");
  ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,v->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},out->view({heads,1,256}));
  r.syncStream();direct_calls_r601.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);return out;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
extern "C" uint64_t modern_glm_direct_calls_r601(int device){return device>=0&&device<8?infinilm::models::glm_moe_dsa::direct_calls_r601[size_t(device)].load():0;}
