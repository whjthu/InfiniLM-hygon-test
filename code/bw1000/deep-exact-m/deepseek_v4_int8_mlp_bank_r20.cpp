#include "deepseek_v4_int8_mlp_bank_r20.hpp"
#include "execution/submit.hpp"
#include "deep_original_hygon_handle_r8833.hpp"
#include <exception>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
DeepseekV4Int8MLPBankR20::DeepseekV4Int8MLPBankR20(runtime::Runtime &rt,size_t hidden,size_t intermediate,
    const runtime::DataType &dtype,const engine::distributed::RankInfo &rank,double swiglu_limit)
    :device_(rt.device()),dtype_(dtype),swiglu_limit_(swiglu_limit),hidden_(hidden),local_intermediate_(0){
  rt.assertOwnerThread();
  if(rt.isGraphRecording()||rt.device().type()!=runtime::Device::Type::kHygon||rank.device!=rt.device()||rank.tp_size<=0||rank.tp_rank<0||rank.tp_rank>=rank.tp_size
     ||!std::isfinite(swiglu_limit_)||swiglu_limit_<=0||!hidden||!intermediate||intermediate%size_t(rank.tp_size)||hidden%4
     ||(dtype!=runtime::DataType::kBFloat16&&dtype!=runtime::DataType::kFloat32))
    throw std::invalid_argument("DeepSeek INT8 MLP construction resources or geometry unsupported");
  local_intermediate_=intermediate/size_t(rank.tp_size);
  if(local_intermediate_%4||local_intermediate_>size_t(std::numeric_limits<int>::max()/2)||hidden>size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("DeepSeek INT8 MLP partition exceeds provider contract");
  try{
    w13_=runtime::Tensor::empty_parameter(rt,{2*local_intermediate_,hidden},runtime::DataType::kInt8);
    s13_=runtime::Tensor::empty_parameter(rt,{2*local_intermediate_,1},runtime::DataType::kFloat32);
    w2_=runtime::Tensor::empty_parameter(rt,{hidden,local_intermediate_},runtime::DataType::kInt8);
    s2_=runtime::Tensor::empty_parameter(rt,{hidden,1},runtime::DataType::kFloat32);
    for(size_t i=0;i<2;++i){
      const std::string name=i==0?"w1":"w3";
      bind_weight(name+".weight",weights::Weight(w13_->narrow({{0,i*local_intermediate_,local_intermediate_}}),0,rank.tp_rank,rank.tp_size));
      bind_weight(name+".scale",weights::Weight(s13_->narrow({{0,i*local_intermediate_,local_intermediate_}}),0,rank.tp_rank,rank.tp_size));
    }
    bind_weight("w2.weight",weights::Weight(w2_,1,rank.tp_rank,rank.tp_size));
    bind_weight("w2.scale",weights::Weight(s2_));
  }catch(...){try{rt.syncStream();}catch(...){std::terminate();}throw;}
}
DeepseekV4Int8MLPBankR20::DeepseekV4Int8MLPBankR20(runtime::Runtime&r,size_t hidden,size_t intermediate,
 const runtime::DataType&dtype,const engine::distributed::RankInfo&rank,
 runtime::Tensor w13,runtime::Tensor s13,runtime::Tensor w2,runtime::Tensor s2)
 :device_(r.device()),dtype_(dtype),swiglu_limit_(10.0),hidden_(hidden),local_intermediate_(256),
  w13_(std::move(w13)),s13_(std::move(s13)),w2_(std::move(w2)),s2_(std::move(s2)){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=runtime::Device::Type::kHygon||rank.device!=device_||rank.tp_size!=8||rank.tp_rank<0||rank.tp_rank>=8||hidden!=4096||intermediate!=2048||dtype!=runtime::DataType::kBFloat16)throw std::invalid_argument("DeepSeek bank MLP requires actual TP8 geometry");
 auto valid=[&](const runtime::Tensor&t,runtime::Shape shape,runtime::DataType type){return t&&t->device()==device_&&t->shape()==shape&&t->dtype()==type&&t->is_contiguous();};
 if(!valid(w13_,{512,4096},runtime::DataType::kInt8)||!valid(s13_,{512,1},runtime::DataType::kFloat32)||!valid(w2_,{4096,256},runtime::DataType::kInt8)||!valid(s2_,{4096,1},runtime::DataType::kFloat32))throw std::invalid_argument("DeepSeek bank MLP tensor geometry differs");
 for(size_t i=0;i<2;++i){const std::string name=i==0?"w1":"w3";
  bind_weight(name+".weight",weights::Weight(w13_->narrow({{0,i*256,256}}),0,rank.tp_rank,rank.tp_size));
  bind_weight(name+".scale",weights::Weight(s13_->narrow({{0,i*256,256}}),0,rank.tp_rank,rank.tp_size));
 }
 bind_weight("w2.weight",weights::Weight(w2_,1,rank.tp_rank,rank.tp_size));
 bind_weight("w2.scale",weights::Weight(s2_));
}
void DeepseekV4Int8MLPBankR20::process_weights_after_loading(runtime::Runtime &rt,bool){
  rt.assertOwnerThread();if(rt.device()!=device_||rt.isGraphRecording()||postload_attempted_)throw std::logic_error("DeepSeek INT8 MLP postload owner/state invalid");
  postload_attempted_=true;
  // Weight::load performs strict checkpoint dtype/shape validation. Parameters
  // remain in their original I8 and F32 scale layouts on this backend.
  rt.syncStream();ready_=true;
}
runtime::Tensor DeepseekV4Int8MLPBankR20::forward_local(runtime::Runtime &rt,const runtime::Tensor &input) const{
  return forward_local_impl(rt,input,nullptr);
}
runtime::Tensor DeepseekV4Int8MLPBankR20::forward_local_retained(runtime::Runtime &rt,const runtime::Tensor &input,std::vector<runtime::Tensor> &retained) const{
  if(rt.isGraphRecording())throw std::invalid_argument("bank lease requires eager execution");
  return forward_local_impl(rt,input,&retained);
}
runtime::Tensor DeepseekV4Int8MLPBankR20::forward_local_impl(runtime::Runtime &rt,const runtime::Tensor &input,std::vector<runtime::Tensor> *retained) const{
  rt.assertOwnerThread();
  if(!ready_||rt.device()!=device_||!input||input->device()!=device_||input->dtype()!=dtype_||input->ndim()!=2||input->size(1)!=hidden_)
    throw std::invalid_argument("DeepSeek INT8 MLP requires loaded same-owner [tokens,hidden] input");
  const size_t tokens=input->size(0);if(!tokens)throw std::invalid_argument("DeepSeek INT8 MLP zero token invocation unsupported");
  std::vector<runtime::Tensor> owners{input,w13_,s13_,w2_,s2_};
  try {
  auto linear=[&](const runtime::Tensor &x,const runtime::Tensor &w,const runtime::Tensor &scale){
    auto packed=runtime::Tensor::empty(rt,x->shape(),runtime::DataType::kInt8);
    owners.push_back(packed);
    auto xs=runtime::Tensor::empty(rt,{tokens,1},runtime::DataType::kFloat32);
    owners.push_back(xs);
    auto out=runtime::Tensor::empty(rt,{tokens,w->size(0)},dtype_);
    owners.push_back(out);
    auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);
    auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);
    execution::submit<infini::ops::PerChannelQuantI8>(rt,execution::RecordingPolicy::HostOnly,quant,x,packed,xs);
    execution::submit<infini::ops::I8Gemm>(rt,execution::RecordingPolicy::HostOnly,gemm,out,packed,xs,w->permute({1,0}),scale,std::optional<runtime::Tensor>{});
    return out;
  };
  auto gate_up=linear(input,w13_,s13_);
  auto clamped=runtime::Tensor::empty(rt,gate_up->shape(),dtype_);
  owners.push_back(clamped);
  auto gate=gate_up->narrow({{1,0,local_intermediate_}}),up=gate_up->narrow({{1,local_intermediate_,local_intermediate_}});
  auto cg=clamped->narrow({{1,0,local_intermediate_}}),cu=clamped->narrow({{1,local_intermediate_,local_intermediate_}});
  execution::submit<infini::ops::Clamp>(rt,infini::ops::Clamp::DefaultConfig(device_.type()),gate,std::optional<double>{},std::optional<double>{swiglu_limit_},cg);
  execution::submit<infini::ops::Clamp>(rt,infini::ops::Clamp::DefaultConfig(device_.type()),up,std::optional<double>{-swiglu_limit_},std::optional<double>{swiglu_limit_},cu);
  auto activated=runtime::Tensor::empty(rt,{tokens,local_intermediate_},dtype_);
  owners.push_back(activated);
  execution::submit<infini::ops::SiluAndMul>(rt,infini::ops::SiluAndMul::DefaultConfig(device_.type()),clamped,activated);
  auto out=linear(activated,w2_,s2_);
  // One correctness fence retains every submitted operand through completion.
  // Replace with a worker-owned lease only after the asynchronous path is tested.
  if(retained)retained->insert(retained->end(),owners.begin(),owners.end());
  else rt.syncStream();
  return out;
  }catch(...){try{rt.syncStream();}catch(...){std::terminate();}throw;}
}
}

#include <array>
#include <map>
#include <cstdint>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <infini/rt/device_guard.h>
namespace infini::ops::hygon_i8_gemm_detail {
struct Layout {std::size_t m,n;std::int64_t cr,cc,ar,br,sar,sbr,biasr,biasc;};
static_assert(sizeof(Layout)==80 && offsetof(Layout,biasc)==72,"original epilogue Layout ABI");
void Epilogue(void*,const std::int32_t*,const float*,const float*,const void*,DataType,Layout,cudaStream_t);
}
namespace infinilm::models::deepseek_v4::bank_exact_m_r8833 {
namespace {
void check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("exactM cublas status "+std::to_string(int(e)));}
void* ptr(const runtime::Tensor&t){return const_cast<void*>(static_cast<const void*>(t->data()));}
thread_local std::string latest="{}";
}
struct Owner {runtime::Runtime*runtime;void*stream;std::shared_ptr<infini::ops::HygonGemmHandle>handle;runtime::Tensor workspace;};
struct Lease {std::shared_ptr<Owner>owner;std::vector<runtime::Tensor>buffers;std::vector<std::uint64_t>host_pointers;runtime::Tensor table;size_t bins=0,experts=0,rows=0,vendor_calls=0,epilogues=0,pointer_H2D_calls=0;};
std::shared_ptr<Owner> prepare(runtime::Runtime&r){r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=runtime::Device::Type::kHygon)throw std::invalid_argument("exactM eager Hygon");infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};auto o=std::make_shared<Owner>();o->runtime=&r;o->stream=r.stream();o->handle=infini::ops::HygonGemmHandle::Acquire(r.device().index());o->workspace=runtime::Tensor::empty(r,{4*1024*1024},runtime::DataType::kInt8);if(reinterpret_cast<std::uintptr_t>(ptr(o->workspace))%256)throw std::invalid_argument("exactM BLAS workspace alignment");return o;}
std::shared_ptr<Lease> make_lease(runtime::Runtime&r,const std::shared_ptr<Owner>&o){r.assertOwnerThread();if(!o||o->runtime!=&r||o->stream!=r.stream()||r.isGraphRecording())throw std::invalid_argument("exactM owner/stream");auto l=std::make_shared<Lease>();l->owner=o;l->buffers.reserve(256*20);l->host_pointers.reserve(256*6);return l;}
void finish_report(const std::shared_ptr<Lease>&l){latest=nlohmann::json({{"enabled",true},{"exact_M",true},{"padding_rows",0},{"bins",l->bins},{"experts",l->experts},{"routes",l->rows},{"vendor_GemmBatchedEx_calls",l->vendor_calls},{"original_epilogue_calls",l->epilogues},{"pointer_table_H2D_calls",l->pointer_H2D_calls},{"pointer_table_H2D_bytes",l->host_pointers.size()*sizeof(std::uint64_t)},{"caller_BLAS_workspace_bytes",4*1024*1024},{"all_intermediate_I32_and_host_tables_retained_to_Sum_drain",true}}).dump();}
void clear_report(){latest="{}";}
const std::string&report(){return latest;}
}
namespace infinilm::models::deepseek_v4 {
std::vector<runtime::Tensor> DeepseekV4Int8MLPBankR20::forward_exact_m(runtime::Runtime&r,const std::vector<std::shared_ptr<DeepseekV4Int8MLPBankR20>>&modules,const std::vector<runtime::Tensor>&inputs,const std::shared_ptr<bank_exact_m_r8833::Lease>&lease){
 using namespace bank_exact_m_r8833;r.assertOwnerThread();if(!lease||!lease->owner||lease->owner->runtime!=&r||lease->owner->stream!=r.stream()||r.isGraphRecording()||inputs.empty()||inputs.size()!=modules.size()||inputs.size()>256)throw std::invalid_argument("exactM bank batch contract");
 struct Item {std::shared_ptr<DeepseekV4Int8MLPBankR20>module;runtime::Tensor input,p1,s1,gate_up,clamped,activated,p2,s2,out,a1,a2;size_t m;};
 struct Bin {size_t m;std::vector<size_t>items;std::array<size_t,2>offset;};
 std::vector<Item>items;items.reserve(inputs.size());std::map<size_t,std::vector<size_t>>grouped;std::vector<Bin>bins;std::vector<runtime::Tensor>outputs;outputs.reserve(inputs.size());
 auto keep=[&](const runtime::Tensor&t){lease->buffers.push_back(t);return t;};auto alloc=[&](runtime::Shape shape,runtime::DataType dtype){return keep(runtime::Tensor::empty(r,shape,dtype));};
 try {
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 for(size_t i=0;i<inputs.size();++i){const auto&mod=modules[i];const auto&x=inputs[i];if(!mod||!mod->ready_||mod->device_!=r.device()||mod->dtype_!=runtime::DataType::kBFloat16||mod->hidden_!=4096||mod->local_intermediate_!=256||!x||x->device()!=r.device()||x->dtype()!=runtime::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=4096||!x->is_contiguous())throw std::invalid_argument("exactM actual TP8 bank operands");Item t;t.module=mod;t.input=keep(x);t.m=x->size(0);for(auto w:{mod->w13_,mod->s13_,mod->w2_,mod->s2_})keep(w);t.p1=alloc({t.m,4096},runtime::DataType::kInt8);t.s1=alloc({t.m,1},runtime::DataType::kFloat32);t.gate_up=alloc({t.m,512},runtime::DataType::kBFloat16);t.clamped=alloc({t.m,512},runtime::DataType::kBFloat16);t.activated=alloc({t.m,256},runtime::DataType::kBFloat16);t.p2=alloc({t.m,256},runtime::DataType::kInt8);t.s2=alloc({t.m,1},runtime::DataType::kFloat32);t.out=alloc({t.m,4096},runtime::DataType::kBFloat16);t.a1=alloc({t.m,512},runtime::DataType::kInt32);t.a2=alloc({t.m,4096},runtime::DataType::kInt32);grouped[t.m].push_back(i);lease->rows+=t.m;outputs.push_back(t.out);items.push_back(std::move(t));}
 lease->experts=items.size();lease->bins=grouped.size();
 static_assert(sizeof(std::uint64_t)==sizeof(void*));
 for(const auto&[m,indices]:grouped){Bin b{m,indices,{}};for(size_t stage=0;stage<2;++stage){b.offset[stage]=lease->host_pointers.size();for(int which=0;which<3;++which)for(size_t i:indices){const auto&t=items.at(i);auto p=which==0?ptr(stage?t.module->w2_:t.module->w13_):which==1?ptr(stage?t.p2:t.p1):ptr(stage?t.a2:t.a1);lease->host_pointers.push_back(reinterpret_cast<std::uintptr_t>(p));}}bins.push_back(std::move(b));}
 lease->table=alloc({lease->host_pointers.size()},runtime::DataType::kInt64);r.memcpyH2D(ptr(lease->table),lease->host_pointers.data(),lease->host_pointers.size()*sizeof(std::uint64_t),true);++lease->pointer_H2D_calls;
 auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(r.device().type());quant.set_implementation_index(0);
 for(auto&t:items)execution::submit<infini::ops::PerChannelQuantI8>(r,execution::RecordingPolicy::HostOnly,quant,t.input,t.p1,t.s1);
 auto vendor=[&](const Bin&b,size_t stage){const size_t n=stage?4096:512,k=stage?256:4096,count=b.items.size();if(b.m>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("exactM INTdimensions");const auto handle=lease->owner->handle->get();auto stream=static_cast<cudaStream_t>(r.stream());check(cublasSetStream(handle,stream));check(cublasSetWorkspace(handle,ptr(lease->owner->workspace),4*1024*1024));check(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));check(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));const std::int32_t alpha=1,beta=0;auto base=static_cast<unsigned char*>(ptr(lease->table))+b.offset[stage]*sizeof(std::uint64_t);auto aa=reinterpret_cast<const void*const*>(base);auto bb=reinterpret_cast<const void*const*>(base+count*sizeof(std::uint64_t));auto cc=reinterpret_cast<void*const*>(base+2*count*sizeof(std::uint64_t));check(cublasGemmBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,int(n),int(b.m),int(k),&alpha,aa,CUDA_R_8I,int(k),bb,CUDA_R_8I,int(k),&beta,cc,CUDA_R_32I,int(n),int(count),CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));++lease->vendor_calls;};
 auto epilogue=[&](Item&t,size_t stage){const auto&out=stage?t.out:t.gate_up;const auto&acc=stage?t.a2:t.a1;const auto&xs=stage?t.s2:t.s1;const auto&scale=stage?t.module->s2_:t.module->s13_;const size_t n=stage?4096:512,k=stage?256:4096;infini::ops::hygon_i8_gemm_detail::Layout layout{t.m,n,int64_t(out->strides().at(0)),int64_t(out->strides().at(1)),int64_t(k),int64_t(k),int64_t(xs->strides().at(0)),int64_t(scale->strides().at(0)),0,0};infini::ops::hygon_i8_gemm_detail::Epilogue(ptr(out),static_cast<const std::int32_t*>(ptr(acc)),static_cast<const float*>(ptr(xs)),static_cast<const float*>(ptr(scale)),nullptr,runtime::DataType::kBFloat16,layout,static_cast<cudaStream_t>(r.stream()));++lease->epilogues;};
 for(const auto&b:bins){vendor(b,0);for(size_t i:b.items){auto&t=items.at(i);epilogue(t,0);auto gate=t.gate_up->narrow({{1,0,256}}),up=t.gate_up->narrow({{1,256,256}}),cg=t.clamped->narrow({{1,0,256}}),cu=t.clamped->narrow({{1,256,256}});execution::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),gate,std::optional<double>{},std::optional<double>{t.module->swiglu_limit_},cg);execution::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),up,std::optional<double>{-t.module->swiglu_limit_},std::optional<double>{t.module->swiglu_limit_},cu);execution::submit<infini::ops::SiluAndMul>(r,infini::ops::SiluAndMul::DefaultConfig(r.device().type()),t.clamped,t.activated);execution::submit<infini::ops::PerChannelQuantI8>(r,execution::RecordingPolicy::HostOnly,quant,t.activated,t.p2,t.s2);}vendor(b,1);for(size_t i:b.items)epilogue(items.at(i),1);}
 return outputs;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
