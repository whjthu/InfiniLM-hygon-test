#pragma once
#include "geometry_r433.hpp"
#include "glm_moe_dsa_attention_prepare.hpp"
#include "execution/submit.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <thread>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
extern "C" const char*glm_fp8_error_r776();
extern "C" void*glm_fp8_create_r776(int,int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*);
extern "C" int glm_fp8_run_r776(void*,int,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_destroy_r776(void*);
namespace infinilm::models::glm_moe_dsa::value_r433 {
namespace rt=runtime;namespace ex=execution;
struct Counts{uint64_t chosen{},rowquant_submits{},row_roundtrip_F32_identity_calls{},F32_aggregation_Gemm_submits{},F32_V_Gemm_submits{},final_BF16_cast_submits{};};
inline thread_local Counts counts;
struct Work{
 rt::Runtime&r;std::vector<rt::Tensor>held;
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor alloc(rt::Shape s,rt::DataType d){auto t=rt::Tensor::empty(r,s,d);held.push_back(t);return t;}
};
// The existing provider owns only temporary scratch. Its destroy drains before
// caller-owned Runtime tensor handles can leave this scope, also on exceptions.
struct OriginalF32Plan{
 void*plan{};
 OriginalF32Plan(rt::Runtime&r,size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro){
  r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon)throw std::invalid_argument("r433 eager Hygon Runtime");
  plan=glm_fp8_create_r776(r.device().index(),2,m,n,k,sr,sc,ro,0,r.stream());
  if(!plan)throw std::runtime_error(glm_fp8_error_r776());
 }
 OriginalF32Plan(const OriginalF32Plan&)=delete;
 ~OriginalF32Plan(){if(plan&&glm_fp8_destroy_r776(plan))std::terminate();}
 void run(const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws){
  if(glm_fp8_run_r776(plan,1,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_error_r776());
 }
};
inline rt::Tensor original_F32_linear(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro){
 r.assertOwnerThread();if(!x||!xs||!w||!ws||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||x->size(1)!=512||w->size(1)!=512||xs->shape()!=rt::Shape{x->size(0),4}||ws->size(1)!=4||!ws->size(0)||ws->size(0)>512||!x->size(0)||x->size(0)>8192||!w->size(0)||w->size(0)>32768||ro>ws->size(0)*128||w->size(0)>ws->size(0)*128-ro)throw std::invalid_argument("r433 original F32linear shape/scale");
 for(const auto&t:{x,w})if(t->device()!=r.device()||t->dtype()!=rt::DataType::kUInt8||!t->is_contiguous())throw std::invalid_argument("r433 original packed operand");
 for(const auto&t:{xs,ws})if(t->device()!=r.device()||t->dtype()!=rt::DataType::kFloat32||!t->is_contiguous())throw std::invalid_argument("r433 original F32scale");
 auto out=rt::Tensor::empty(r,{x->size(0),w->size(0)},rt::DataType::kFloat32);
 {OriginalF32Plan p(r,x->size(0),w->size(0),512,ws->size(0),4,ro);p.run(out,x,xs,w,ws);}
 return out;
}
struct Prepared final{
 rt::Runtime&r;void*stream;std::thread::id owner;int rank;bool ready=false;
 rt::Tensor identity,identity_xscale,identity_wscale,original_weight,original_scale,setup_transposed;
 std::array<rt::Tensor,8>V;
 // Ordinary host staging is a member: held through successful and failed H2D.
 std::vector<uint8_t>identity_host;std::vector<float>xscale_host,wscale_host;
 double setup_wall_ms{};uint64_t setup_original_F32_calls{};
 Prepared(rt::Runtime&runtime,AttentionPrepare&prepare,int device_rank):r(runtime),stream(r.stream()),owner(std::this_thread::get_id()),rank(device_rank){
  try{
   check();if(rank<0||rank>=8||prepare.local_heads()!=8)throw std::invalid_argument("r433 TP8 rank");
   auto weights=prepare.state_dict();original_weight=weights.at("kv_b_proj.weight").tensor();original_scale=weights.at("kv_b_proj.weight_scale_inv").tensor();
   if(original_weight->shape()!=rt::Shape{3584,512}||original_weight->dtype()!=rt::DataType::kUInt8||original_scale->shape()!=rt::Shape{224,4}||original_scale->dtype()!=rt::DataType::kFloat32||original_weight->device()!=r.device()||original_scale->device()!=r.device()||!original_weight->is_contiguous()||!original_scale->is_contiguous())throw std::invalid_argument("r433 public KVb weight/scale");
   auto start=std::chrono::steady_clock::now();identity_host.resize(512*512);for(size_t i=0;i<512;++i)for(size_t j=0;j<512;++j)identity_host[i*512+j]=glm_value_r433::identity_code(i,j);
   xscale_host.assign(512*4,1.f);wscale_host.assign(4*4,1.f);
   identity=rt::Tensor::empty(r,{512,512},rt::DataType::kUInt8);identity_xscale=rt::Tensor::empty(r,{512,4},rt::DataType::kFloat32);identity_wscale=rt::Tensor::empty(r,{4,4},rt::DataType::kFloat32);
   r.memcpyH2D(const_cast<std::byte*>(identity->data()),identity_host.data(),identity_host.size(),false);r.memcpyH2D(const_cast<std::byte*>(identity_xscale->data()),xscale_host.data(),xscale_host.size()*4,false);r.memcpyH2D(const_cast<std::byte*>(identity_wscale->data()),wscale_host.data(),wscale_host.size()*4,false);
   // I512 * original W^T yields scaled F32 W transpose; no buffer borrowed
   // from r807/r809, and no per-head raw W/scale slicing across128 boundaries.
   setup_transposed=original_F32_linear(r,identity,identity_xscale,original_weight,original_scale,glm_value_r433::row_offset(rank));++setup_original_F32_calls;
   for(size_t h=0;h<8;++h)V[h]=setup_transposed->narrow({{1,glm_value_r433::value_column(h),256}})->contiguous(r);
   // Member owns the source also if any copy throws; body catch drains before
   // member destruction. Release our temporary handle only after all copies.
   r.syncStream();setup_transposed.reset();
   r.syncStream();ready=true;setup_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
 }
 Prepared(const Prepared&)=delete;
 ~Prepared(){try{check();r.syncStream();}catch(...){std::terminate();}}
 void check()const{r.assertOwnerThread();if(owner!=std::this_thread::get_id()||r.stream()!=stream||r.device().index()!=rank||r.isGraphRecording())throw std::invalid_argument("r433 prepared owner/stream/device/eager");}
};
inline rt::Tensor attention(rt::Runtime&r,Prepared&prepared,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&selected_latent,float scale){
 prepared.check();if(&r!=&prepared.r||!prepared.ready||!q||!k||!selected_latent||q->shape()!=rt::Shape{1,8,256}||k->shape()!=rt::Shape{2048,8,256}||selected_latent->shape()!=rt::Shape{2048,512}||!std::isfinite(scale)||scale<=0)throw std::invalid_argument("r433 C1/U2048 exact shape");
 for(const auto&t:{q,k,selected_latent})if(t->device()!=r.device()||t->dtype()!=rt::DataType::kBFloat16||!t->is_contiguous())throw std::invalid_argument("r433 BF16 input");
 Work w{r,{q,k,selected_latent}};auto dots=w.alloc({8,1,2048},rt::DataType::kBFloat16),logits=w.alloc({8,1,2048},rt::DataType::kFloat32),prob=w.alloc({8,1,2048},rt::DataType::kFloat32),pb=w.alloc({8,1,2048},rt::DataType::kBFloat16);
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),q->view({8,1,256}),k->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
 if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,8*2048,scale,r.stream()))throw std::runtime_error("r433 original scale/round");
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
 auto packed=w.alloc({2048,512},rt::DataType::kUInt8),xs=w.alloc({2048,4},rt::DataType::kFloat32);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),selected_latent,packed,xs);++counts.rowquant_submits;
 auto roundtrip=original_F32_linear(r,packed,xs,prepared.identity,prepared.identity_wscale,0);w.held.push_back(roundtrip);++counts.row_roundtrip_F32_identity_calls;
 auto p32=w.alloc({8,1,2048},rt::DataType::kFloat32),aggregate=w.alloc({8,512},rt::DataType::kFloat32),out32=w.alloc({8,256},rt::DataType::kFloat32),out=w.alloc({1,8,256},rt::DataType::kBFloat16);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),pb,p32);
 // Explicit FullFp32 policy prevents silently introducing a new TF32 arm.
 const infini::ops::GemmConfig full{infini::ops::GemmConfig::Fp32Precision::FullFp32};
 for(size_t h=0;h<8;++h){auto p=p32->narrow({{0,h,1}})->view({1,2048}),a=aggregate->narrow({{0,h,1}}),o=out32->narrow({{0,h,1}});w.held.insert(w.held.end(),{p,a,o,prepared.V[h]});
  ex::submit<infini::ops::Gemm>(r,full,p,roundtrip,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},a);++counts.F32_aggregation_Gemm_submits;
  ex::submit<infini::ops::Gemm>(r,full,a,prepared.V[h],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},o);++counts.F32_V_Gemm_submits;
 }
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),out32->view({1,8,256}),out);++counts.final_BF16_cast_submits;r.syncStream();++counts.chosen;return out;
}
}
