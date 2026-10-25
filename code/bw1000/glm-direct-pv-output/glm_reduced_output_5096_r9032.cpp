#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <optional>
#include <array>
#include <atomic>
#include <string>
extern "C" int glm_scale_round_r392(const void*,void*,void*,size_t,float,void*);
extern "C" int glm_reduced_output_choose_r9032(int,int);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{std::array<std::array<std::atomic<uint64_t>,2>,8>batch_stats5096{};struct Work5096{rt::Runtime&r;std::vector<rt::Tensor>held;~Work5096(){try{r.syncStream();}catch(...){std::terminate();}}};}
rt::Tensor sparse_attention_all_selected_decode_batch_r5096(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!k||!v||q->ndim()!=3||k->ndim()!=4||v->shape()!=k->shape()||q->size(0)<8||q->size(0)>128||q->size(1)!=8||q->size(2)!=256||k->size(0)!=q->size(0)||!k->size(1)||k->size(1)>640||k->size(2)!=8||k->size(3)!=256||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||!q->is_contiguous()||!k->is_contiguous()||!v->is_contiguous())throw std::invalid_argument("5055 bounded batched allselected decode");
 const size_t B=q->size(0),P=k->size(1);Work5096 work{r};auto alloc=[&](rt::Shape s,rt::DataType dtype){auto x=rt::Tensor::empty(r,s,dtype);work.held.push_back(x);return x;};auto out=alloc({B,8,256},rt::DataType::kBFloat16);
const bool direct_output9032=B==8&&glm_reduced_output_choose_r9032(r.device().index(),1);
for(size_t i=0;i<B;++i){
   auto qr=q->narrow({{0,i,1}}),kr=k->narrow({{0,i,1}})->view({P,8,256}),vr=v->narrow({{0,i,1}})->view({P,8,256});
   auto dots=alloc({8,1,P},rt::DataType::kBFloat16),logits=alloc({8,1,P},rt::DataType::kFloat32),prob=alloc({8,1,P},rt::DataType::kFloat32),pb=alloc({8,1,P},rt::DataType::kBFloat16),result=direct_output9032?out->narrow({{0,i,1}}):alloc({1,8,256},rt::DataType::kBFloat16);
   if(direct_output9032)work.held.push_back(result);
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qr->view({8,1,256}),kr->permute({1,2,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
   if(glm_scale_round_r392(dots->data(),logits->data(),nullptr,8*P,1.f/16,r.stream()))throw std::runtime_error("5092 scale round");
   ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),prob,pb);
   ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),pb,vr->permute({1,0,2}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result->view({8,1,256}));
   if(!direct_output9032)out->narrow({{0,i,1}})->copy_from(r,result);
  }
 r.syncStream();++batch_stats5096.at(r.device().index())[0];batch_stats5096.at(r.device().index())[1].fetch_add(B);return out;
}
}
extern "C" int modern_glm_batch_attention_counts_r5096(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<n;++i)out[i]=infinilm::models::glm_moe_dsa::batch_stats5096[device][i].load();return 0;}
