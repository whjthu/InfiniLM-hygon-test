#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <optional>
#include <array>
#include <atomic>
#include <string>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
extern "C" int glm_fused_decode_r9026(void*,const void*,const void*const*,const void*const*,const size_t*,size_t,size_t,size_t,void*);
extern "C" int glm_fused_decode_policy_r9026(int,int);
extern "C" int glm_fused_decode_counts_r9026(int,uint64_t*,size_t);
namespace {struct Entry5096R9028{int enabled=0;uint64_t ordinary=0,fused=0,errors=0;};thread_local std::array<Entry5096R9028,8> entries9028;}
extern "C" int glm_fused_entries_policy_r9028(int d,int enabled){if(d<0||d>=8||enabled<0||enabled>1)return-1;int rc=glm_fused_decode_policy_r9026(d,enabled);if(rc)return rc;entries9028[d].enabled=enabled;return 0;}
extern "C" int glm_fused_entries_split_r9028(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=6)return-1;int rc=glm_fused_decode_counts_r9026(d,out,3);if(rc)return rc;auto&s=entries9028[d];out[3]=s.ordinary;out[4]=s.fused;out[5]=s.errors;return 0;}
extern "C" int glm_fused_entries_counts_r9028(int d,uint64_t*out,size_t n){if(!out||n!=3)return-1;uint64_t split[6]{};int rc=glm_fused_entries_split_r9028(d,split,6);if(rc)return rc;for(size_t j=0;j<3;++j)out[j]=split[j]+split[j+3];return 0;}

namespace{std::array<std::array<std::atomic<uint64_t>,2>,8>batch_stats5096{};struct Work5096{rt::Runtime&r;std::vector<rt::Tensor>held;~Work5096(){try{r.syncStream();}catch(...){std::terminate();}}};}
rt::Tensor sparse_attention_all_selected_decode_batch_r5096(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!v||q->ndim()!=3||k->ndim()!=4||v->shape()!=k->shape()||q->size(0)<8||q->size(0)>128||q->size(1)!=8||q->size(2)!=256||k->size(0)!=q->size(0)||!k->size(1)||k->size(1)>640||k->size(2)!=8||k->size(3)!=256||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||!q->is_contiguous()||!k->is_contiguous()||!v->is_contiguous())throw std::invalid_argument("5055 bounded batched allselected decode");
 const size_t B=q->size(0),P=k->size(1);Work5096 work{r};auto alloc=[&](rt::Shape s,rt::DataType dtype){auto x=rt::Tensor::empty(r,s,dtype);work.held.push_back(x);return x;};auto out=alloc({B,8,256},rt::DataType::kBFloat16);
 auto&policy=entries9028.at(r.device().index());
 if(B==8&&policy.enabled){
  std::array<const void*,8>kp{},vp{};std::array<size_t,8>pp{};
  for(size_t row=0;row<8;++row){kp[row]=k->data()+row*P*8*256*2;vp[row]=v->data()+row*P*8*256*2;pp[row]=P;}
  int rc=glm_fused_decode_r9026(out->data(),q->data(),kp.data(),vp.data(),pp.data(),B,8,256,r.stream());
  if(rc){++policy.errors;throw std::runtime_error("9028 fused5096 launch "+std::to_string(rc));}
  ++policy.fused;r.syncStream();++batch_stats5096.at(r.device().index())[0];batch_stats5096.at(r.device().index())[1].fetch_add(B);return out;
 }
 if(B==8)++policy.ordinary;

for(size_t i=0;i<B;++i){
   auto qr=q->narrow({{0,i,1}}),kr=k->narrow({{0,i,1}})->view({P,8,256}),vr=v->narrow({{0,i,1}})->view({P,8,256});
   auto dots=alloc({8,1,P},rt::DataType::kBFloat16),logits=alloc({8,1,P},rt::DataType::kFloat32),prob=alloc({8,1,P},rt::DataType::kFloat32),pb=alloc({8,1,P},rt::DataType::kBFloat16),result=alloc({1,8,256},rt::DataType::kBFloat16);
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qr->view({8,1,256}),kr->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
   if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,8*P,1.f/16,r.stream()))throw std::runtime_error("5092 scale round");
   ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,vr->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result->view({8,1,256}));
   out->narrow({{0,i,1}})->copy_from(r,result);
  }
 r.syncStream();++batch_stats5096.at(r.device().index())[0];batch_stats5096.at(r.device().index())[1].fetch_add(B);return out;
}
}
extern "C" int modern_glm_batch_attention_counts_r5096(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<n;++i)out[i]=infinilm::models::glm_moe_dsa::batch_stats5096[device][i].load();return 0;}
