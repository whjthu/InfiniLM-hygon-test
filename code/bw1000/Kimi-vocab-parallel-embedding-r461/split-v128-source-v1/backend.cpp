#include "contract.hpp"
#include "kernels.h"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include "global_state/forward_context.hpp"
#include <base/gemm.h>
#include <base/cast.h>
#include <base/softmax.h>
#include <base/add.h>
#include <algorithm>
#include <exception>
namespace kimi_split_v128 {
using namespace infinilm;using runtime::Tensor;using DT=runtime::DataType;
void check(int rc){if(rc)throw std::runtime_error("splitV private kernel submission failed");}
void gemm(runtime::Runtime&r,const Tensor&a,const Tensor&b,const Tensor&y,float alpha){
 execution::submit<infini::ops::Gemm>(r,execution::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b,std::optional<Tensor>{},std::optional<float>{alpha},std::optional<float>{0},std::optional<int>{},std::optional<int>{},y);
}
Tensor forward(runtime::Runtime&r,const Tensor&q,const Tensor&k,const Tensor&v,const Tensor&cache,const global_state::AttentionMetadata&md,float scale){
 r.assertOwnerThread();
 if(r.isGraphRecording()||!q||!k||!v||!cache||r.device().type()!=runtime::Device::Type::kHygon||q->dtype()!=DT::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||cache->dtype()!=q->dtype()
 ||q->shape()!=runtime::Shape{q->size(0),3,192}||k->shape()!=q->shape()||v->shape()!=runtime::Shape{q->size(0),3,128}||cache->shape()!=runtime::Shape{132,256,3,320}
 ||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||cache->device()!=r.device()
 ||!md.block_tables||!*md.block_tables||!md.slot_mapping||!*md.slot_mapping)throw std::invalid_argument("splitV exact eager BF16 C1 tensors/metadata");
 auto p=plan(q->size(0),md.max_sequence_length);auto table=*md.block_tables,slots=*md.slot_mapping;
 if(table->ndim()!=2||table->size(0)!=1||table->size(1)<128||!table->is_contiguous()||slots->ndim()!=1||slots->size(0)!=p.queries||!slots->is_contiguous()||table->device()!=r.device()||slots->device()!=r.device()
 ||(table->dtype()!=DT::kInt32&&table->dtype()!=DT::kInt64)||(slots->dtype()!=DT::kInt32&&slots->dtype()!=DT::kInt64)||q->stride(2)!=1||k->stride(2)!=1||v->stride(2)!=1)throw std::invalid_argument("splitV original C1 page table/slot metadata");
 if(md.first_total_sequence_length&&*md.first_total_sequence_length!=p.total)throw std::invalid_argument("splitV total identity");
 if(md.first_past_sequence_length&&*md.first_past_sequence_length!=p.past)throw std::invalid_argument("splitV past identity");
 Tensor out,scores,prob,keys,values,sum,part;
 try{
  const int64_t ks[]={int64_t(k->stride(0)),int64_t(k->stride(1))},vs[]={int64_t(v->stride(0)),int64_t(v->stride(1))};
  check(kimi_split_store(cache->data(),k->data(),v->data(),slots->data(),slots->dtype()==DT::kInt32?32:64,p.queries,ks,vs,r.stream()));
  out=Tensor::empty(r,{p.queries,3,128},DT::kBFloat16);
  for(size_t h=0;h<3;++h)for(size_t first=0;first<p.queries;first+=8){
   const size_t rows=std::min(size_t(8),p.queries-first);
   scores=Tensor::empty(r,{rows,p.total},DT::kFloat32);prob=Tensor::empty(r,{rows,p.total},DT::kBFloat16);
   keys=Tensor::empty(r,{256,192},DT::kBFloat16);values=Tensor::empty(r,{256,128},DT::kBFloat16);
   sum=Tensor::empty(r,{rows,128},DT::kFloat32);part=Tensor::empty(r,{rows,128},DT::kFloat32);
   auto query=q->narrow({{0,first,rows},{1,h,1}})->squeeze(1);
   for(size_t start=0;start<p.total;start+=256){auto count=std::min(size_t(256),p.total-start);
    check(kimi_split_gather(keys->data(),cache->data(),table->data(),table->dtype()==DT::kInt32?32:64,start,count,h,192,r.stream()));
    gemm(r,query,keys->narrow({{0,0,count}})->permute({1,0}),scores->narrow({{1,start,count}}),scale);
   }
   check(kimi_split_mask(scores->data(),rows,p.total,p.past,first,r.stream()));
   execution::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),scores,int64_t{1},std::optional<infini::ops::DataType>{},scores);
   execution::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),scores,prob);
   // Existing Gemm EffectiveBeta is always0. Never rely on beta=1 accumulation.
   for(size_t start=0;start<p.total;start+=256){auto count=std::min(size_t(256),p.total-start);
    check(kimi_split_gather(values->data(),cache->data(),table->data(),table->dtype()==DT::kInt32?32:64,start,count,h,128,r.stream()));
    gemm(r,prob->narrow({{1,start,count}}),values->narrow({{0,0,count}}),start?part:sum,1.f);
    if(start)execution::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),sum,part,sum);
   }
   auto target=out->narrow({{0,first,rows},{1,h,1}})->squeeze(1);
   execution::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),sum,target);
   // End tile: all uses are submitted on the owner stream; release before next allocations.
   scores={};prob={};keys={};values={};sum={};part={};
  }
  return out;
 }catch(...){auto e=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(e);}
}
}
