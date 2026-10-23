// Private default-OFF Model helper. Same component math as qualified715.
#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <array>
#include <atomic>
#include <set>
#include <optional>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
extern "C" int modern_glm_dense_causal_mask_r25(void*,size_t,size_t,size_t,size_t,void*);
namespace {constexpr size_t B=8,L=512,H=8,D=256;
std::array<std::atomic<uint64_t>,8> calls{};
struct Work {rt::Runtime&r;std::vector<rt::Tensor> held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto t=rt::Tensor::empty(r,s,d);held.push_back(t);return t;}};
void gemm(rt::Runtime&r,const rt::Tensor&a,const rt::Tensor&b,const rt::Tensor&c){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},c);}
void cast(rt::Runtime&r,const rt::Tensor&a,const rt::Tensor&b){ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,b);}
rt::Tensor row(const rt::Tensor&t,size_t b){return t->narrow({{0,b,1}})->view({L,H,D});}
rt::Tensor cross_inner(Work&w,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v){
 auto&r=w.r;
 // Pack the same eight independent requests; no token/feature padding.
 auto qp=q->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 auto kp=k->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 auto vp=v->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 w.held.insert(w.held.end(),{qp,kp,vp});
 auto dots=w.alloc({B*H,L,L},rt::DataType::kBFloat16);r.setDeviceMemory(dots->data(),0,dots->nbytes());
 auto temp=w.alloc({B*H,L},rt::DataType::kBFloat16);
 for(size_t t=0;t<L;++t){size_t count=t+1;auto out=rt::Tensor::from_blob(temp->data(),{B*H,1,count},rt::DataType::kBFloat16,r.device());
  gemm(r,qp->narrow({{1,t,1}}),kp->narrow({{1,0,count}})->permute({0,2,1}),out);
  dots->narrow({{1,t,1},{2,0,count}})->copy_from(r,out);
 }
 // Preserve mode2's F32 scaling then BF16 rounding, F32 logits, Softmax, BF16 P.
 auto df=w.alloc(dots->shape(),rt::DataType::kFloat32),scaled=w.alloc(dots->shape(),rt::DataType::kFloat32),rounded=w.alloc(dots->shape(),rt::DataType::kBFloat16),logits=w.alloc(dots->shape(),rt::DataType::kFloat32),prob=w.alloc(dots->shape(),rt::DataType::kFloat32),pb=w.alloc(dots->shape(),rt::DataType::kBFloat16);
 float scale=1.f/16;auto sf=w.alloc({1},rt::DataType::kFloat32);r.memcpyH2D(sf->data(),&scale,4,false);cast(r,dots,df);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),df,sf,scaled);cast(r,scaled,rounded);cast(r,rounded,logits);
 if(modern_glm_dense_causal_mask_r25(logits->data(),B*H,L,L,0,r.stream()))throw std::runtime_error("mask launch");
 r.setDeviceMemory(prob->data(),0,prob->nbytes());r.setDeviceMemory(pb->data(),0,pb->nbytes());
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);cast(r,prob,pb);
 auto output=w.alloc({B*H,L,D},rt::DataType::kBFloat16),ptmp=w.alloc({B*H,L},rt::DataType::kBFloat16);
 for(size_t t=0;t<L;++t){size_t count=t+1;auto p=rt::Tensor::from_blob(ptmp->data(),{B*H,1,count},rt::DataType::kBFloat16,r.device());p->copy_from(r,pb->narrow({{1,t,1},{2,0,count}}));auto out=output->narrow({{1,t,1}});gemm(r,p,vp->narrow({{1,0,count}}),out);}
 auto packed=output->view({B,H,L,D})->permute({0,2,1,3})->contiguous(r);w.held.push_back(packed);return packed;
}
}
bool glm_crossrequest_fullprefix_r7328(rt::Runtime&r,const std::vector<rt::Tensor>&selection,const rt::Tensor&positions){
 r.assertOwnerThread();if(r.isGraphRecording()||selection.size()!=B||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||!positions->is_contiguous()||positions->shape()!=rt::Shape{B*L})return false;
 std::vector<int64_t>p(B*L);r.memcpyD2H(p.data(),positions->data(),p.size()*8);for(size_t i=0;i<p.size();++i)if(p[i]!=int64_t(i%L))return false;
 for(auto ids:selection){if(!ids||ids->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||!ids->is_contiguous()||ids->shape()!=rt::Shape{L,L})return false;
  std::vector<int64_t>host(L*L);r.memcpyD2H(host.data(),ids->data(),host.size()*8);
  for(size_t t=0;t<L;++t){std::set<int64_t>selected;for(size_t j=0;j<L;++j){auto id=host[t*L+j];if(id<0||size_t(id)>=L)return false;if(size_t(id)<=t)selected.insert(id);}if(selected.size()!=t+1)return false;size_t j=0;for(auto id:selected)if(id!=int64_t(j++))return false;}
 }
 return true;
}
rt::Tensor glm_crossrequest_prefill_r7328(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v){
 r.assertOwnerThread();for(auto t:{q,k,v})if(!t||r.isGraphRecording()||t->device()!=r.device()||t->dtype()!=rt::DataType::kBFloat16||t->shape()!=rt::Shape{B*L,H,D}||!t->is_contiguous())throw std::invalid_argument("7328 full packed BF16 QKV geometry");
 Work w{r};w.held={q,k,v};auto out=cross_inner(w,q->view({B,L,H,D}),k->view({B,L,H,D}),v->view({B,L,H,D}));r.syncStream();calls.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);return out->view({B*L,H,D});
}
extern "C" uint64_t modern_glm_crossrequest_prefill_calls_r7328(int device){return device<0||device>=8?0:calls[size_t(device)].load(std::memory_order_relaxed);}
}
