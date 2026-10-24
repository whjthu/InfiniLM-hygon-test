#include <atomic>
#include <array>
#include "deepseek_v4_hyper_connection.hpp"
#include "deep_collective_layout_r7040.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "execution/submit.hpp"
#include <cmath>
#include <cuda_runtime_api.h>
extern "C" int modern_deepseek_post_r135(const void*,const void*,const float*,const float*,void*,size_t,int,int,void*);
extern "C" int modern_deepseek_sinkhorn_r89(const float*,float*,size_t,float,int,void*,int);
#include <exception>
#include <optional>
#include <vector>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;

namespace rms_r7382 {
std::array<std::array<std::atomic<uint64_t>,3>,8> calls{};
bool enabled(){const char*p=std::getenv("INFINILM_DEEP_HC_RMS_R7382");if(!p||std::strcmp(p,"0")==0)return false;if(std::strcmp(p,"1")==0)return true;throw std::invalid_argument("7382 flag must exactly0or1");}
bool eligible(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&constants){
 auto&c=calls.at(r.device().index());if(!enabled()){++c[0];return false;}
 auto*f=deep_layout_core_r7032::current();
 if(!f||f->policy!=0||f->requests!=32||f->layout.total!=32||f->layout.lengths.size()!=32||!std::all_of(f->layout.lengths.begin(),f->layout.lengths.end(),[](size_t n){return n==1;})||x->dtype()!=rt::DataType::kBFloat16||x->shape()!=rt::Shape{32,4,4096}||wide_r3443::selected(r)!=5||constants->shape()!=rt::Shape{16387}){++c[2];return false;}
 deep_layout_core_r7032::verify(f->context,layout_r7040::snapshot(r));++c[1];return true;
}
}
extern "C" uint64_t deep_hc_rms_calls_r7382(int d,int k){if(d<0||d>=8||k<0||k>=3)return UINT64_MAX;return rms_r7382::calls[d][k].load();}

namespace canonical_r7130 { std::atomic<int> policy{0};std::array<std::array<std::atomic<uint64_t>,3>,8> calls{}; }
extern "C" int deep_hc_canonical_policy_r7130(int flag){if(flag!=0&&flag!=1)return 1;canonical_r7130::policy.store(flag);return 0;}
extern "C" int deep_hc_canonical_get_r7130(){return canonical_r7130::policy.load();}
extern "C" uint64_t deep_hc_canonical_calls_r7130(int dev,int kind){if(dev<0||dev>=8||kind<0||kind>=3)return UINT64_MAX;return canonical_r7130::calls[dev][kind].load();}

namespace {
struct Work{
 rt::Runtime&r;std::vector<rt::Tensor> held;
 explicit Work(rt::Runtime&owner):r(owner){}
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor alloc(rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto t=rt::Tensor::empty(r,shape,dt);held.push_back(t);return t;}
 template<class Op>rt::Tensor binary(const rt::Tensor&a,const rt::Tensor&b,rt::Shape shape){auto out=alloc(shape);ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,b,out);return out;}
 template<class Op>rt::Tensor unary(const rt::Tensor&a){auto out=alloc(a->shape());ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor fp32(const rt::Tensor&a){if(a->dtype()==rt::DataType::kFloat32)return a;auto out=alloc(a->shape());ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor cast(const rt::Tensor&a,rt::DataType dt){if(a->dtype()==dt)return a;auto out=alloc(a->shape(),dt);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor sum(const rt::Tensor&a,int64_t dim,bool keep,rt::Shape shape){auto out=alloc(shape);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),a,std::optional<std::vector<int64_t>>{{dim}},keep,std::optional<infini::ops::DataType>{},out);return out;}
};
void check(rt::Runtime&r,rt::Device device,const rt::Tensor&residual,bool ready){r.assertOwnerThread();if(!ready||r.device()!=device||r.isGraphRecording()||!residual||residual->device()!=device||residual->ndim()!=3||!residual->size(0)||residual->size(1)!=4||residual->size(2)!=4096||(residual->dtype()!=rt::DataType::kBFloat16&&residual->dtype()!=rt::DataType::kFloat32))throw std::invalid_argument("DeepSeek mHC requires loaded same-owner [T,4,4096] BF16/F32 input");}
}
HyperConnection::HyperConnection(rt::Runtime&r,double rms_eps,double hc_eps,int iterations):device_(r.device()),rms_eps_(rms_eps),hc_eps_(hc_eps),iterations_(iterations){
 r.assertOwnerThread();if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||!std::isfinite(rms_eps)||rms_eps<=0||!std::isfinite(hc_eps)||hc_eps<=0||iterations<1||iterations>128)throw std::invalid_argument("DeepSeek mHC construction contract differs");
 fn_=rt::Tensor::empty_parameter(r,{24,16384},rt::DataType::kFloat32);base_=rt::Tensor::empty_parameter(r,{24},rt::DataType::kFloat32);scale_=rt::Tensor::empty_parameter(r,{3},rt::DataType::kFloat32);
 bind_weight("fn",weights::Weight(fn_));bind_weight("base",weights::Weight(base_));bind_weight("scale",weights::Weight(scale_));if(rms_r7382::enabled()){
  // Immutable internal bank: original first3 values plus16384 ones. No publicheader/ABI change.
  constants_=rt::Tensor::empty(r,{16387},rt::DataType::kFloat32);std::vector<float>values(16387,1.f);values[0]=float(rms_eps);values[1]=float(hc_eps);values[2]=2.f;
  r.memcpyH2D(constants_->data(),values.data(),values.size()*sizeof(float),false);
 }else{
  constants_=rt::Tensor::empty(r,{3},rt::DataType::kFloat32);
  float values[]={float(rms_eps),float(hc_eps),2.f};constants_->copy_from(r,rt::Tensor::from_blob(values,{3},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));r.syncStream();
 }
}
void HyperConnection::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek mHC postload state differs");r.syncStream();ready_=true;}
HyperConnection::PreResult HyperConnection::pre(rt::Runtime&r,const rt::Tensor&residual)const{
 check(r,device_,residual,ready_);Work work(r);size_t t=residual->size(0);auto xf=work.fp32(residual),flat=xf->view({t,16384});
 rt::Tensor normalized;
 if(rms_r7382::eligible(r,residual,constants_)){
  normalized=work.alloc({t,16384});ex::submit<infini::ops::RmsNorm>(r,infini::ops::RmsNorm::DefaultConfig(device_.type()),flat,constants_->narrow({{0,3,16384}}),float(rms_eps_),normalized);
 }else{
 auto square=work.binary<infini::ops::Mul>(flat,flat,{t,16384}),mean=work.alloc({t,1});
 ex::submit<infini::ops::Mean>(r,infini::ops::Mean::DefaultConfig(device_.type()),square,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},mean);
 auto plus_eps=work.binary<infini::ops::Add>(mean,constants_->narrow({{0,0,1}}),{t,1});auto inv=work.unary<infini::ops::Rsqrt>(plus_eps);normalized=work.binary<infini::ops::Mul>(flat,inv,{t,16384});
 }
 auto mixes=work.alloc({t,24});

 if(device_.index()<0||device_.index()>=8)throw std::invalid_argument("HC canonical owner device outside TP8 lease");
 if(t==512 && canonical_r7130::policy.load()==1){
  // Work holds both scratch tensors until its existing owner sync; allocator pool
  // may reuse storage on the following eager call. No per-layer persistent bank.
  auto padded=work.alloc({2048,16384}),full_mixes=work.alloc({2048,24});
  r.setDeviceMemory(padded->data(),0,2048*16384*4);
  r.memcpyD2D(padded->data(),normalized->data(),512*16384*4);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),padded,fn_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},full_mixes);
  r.memcpyD2D(mixes->data(),full_mixes->data(),512*24*4);
  canonical_r7130::calls[device_.index()][1].fetch_add(1);
 }else{
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),normalized,fn_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},mixes);
  canonical_r7130::calls[device_.index()][t==512?0:2].fetch_add(1);
 }

 auto affine=[&](size_t offset,size_t width,size_t scale){auto product=work.binary<infini::ops::Mul>(mixes->narrow({{1,offset,width}}),scale_->narrow({{0,scale,1}}),{t,width});return work.binary<infini::ops::Add>(product,base_->narrow({{0,offset,width}}),{t,width});};
 auto pre_sig=work.unary<infini::ops::Sigmoid>(affine(0,4,0));auto pre=work.binary<infini::ops::Add>(pre_sig,constants_->narrow({{0,1,1}}),{t,4});auto post_sig=work.unary<infini::ops::Sigmoid>(affine(4,4,1));auto post=work.binary<infini::ops::Mul>(post_sig,constants_->narrow({{0,2,1}}),{t,4});
 auto comb_logits=affine(8,16,2)->view({t,4,4}),prob=work.alloc({t,4,4});ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(device_.type()),comb_logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);
 auto comb=work.binary<infini::ops::Add>(prob,constants_->narrow({{0,1,1}}),{t,4,4});
 auto normalized_comb=work.alloc({t,4,4});
 int sinkhorn_status=modern_deepseek_sinkhorn_r89(reinterpret_cast<const float*>(comb->data()),reinterpret_cast<float*>(normalized_comb->data()),t,float(hc_eps_),iterations_,r.stream(),device_.index());
 if(sinkhorn_status)throw std::runtime_error(std::string("mHC fused Sinkhorn: ")+cudaGetErrorString(static_cast<cudaError_t>(sinkhorn_status)));
 comb=normalized_comb;
 auto weighted=work.binary<infini::ops::Mul>(pre->view({t,4,1}),xf,{t,4,4096});auto collapsed=work.cast(work.sum(weighted,1,false,{t,4096}),residual->dtype());r.syncStream();return {collapsed,post,comb};
}
rt::Tensor HyperConnection::post(rt::Runtime&r,const rt::Tensor&block,const rt::Tensor&residual,const PreResult&pre)const{
 check(r,device_,residual,ready_);size_t t=residual->size(0);
 if(!block||!pre.post||!pre.comb||block->shape()!=rt::Shape{t,4096}||block->dtype()!=residual->dtype()||block->device()!=device_||pre.post->shape()!=rt::Shape{t,4}||pre.comb->shape()!=rt::Shape{t,4,4}||pre.post->dtype()!=rt::DataType::kFloat32||pre.comb->dtype()!=rt::DataType::kFloat32||pre.post->device()!=device_||pre.comb->device()!=device_)throw std::invalid_argument("DeepSeek mHC post shapes/dtypes/owners differ");
 if(t>4096)throw std::invalid_argument("Qualified fused post token capacity4096 exceeded");
 auto output=rt::Tensor::empty(r,{t,4,4096},residual->dtype());
 int status=modern_deepseek_post_r135(block->data(),residual->data(),reinterpret_cast<const float*>(pre.post->data()),reinterpret_cast<const float*>(pre.comb->data()),output->data(),t,residual->dtype()==rt::DataType::kBFloat16?1:0,1,r.stream());
 if(status)throw std::runtime_error(std::string("mHC strict fused post: ")+cudaGetErrorString(static_cast<cudaError_t>(status)));
 r.syncStream();return output;
}
}
