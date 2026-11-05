#include "backend.hpp"
#include "contract.hpp"
#include "kernels.h"
#include "fa_kernels.h"
#include "execution/submit.hpp"
#include <base/flash_attn_with_kvcache.h>
#include <algorithm>
#include <exception>
namespace kimi_split_v128 {
using namespace infinilm;using runtime::Tensor;using DT=runtime::DataType;
Tensor forward(runtime::Runtime&r,const Tensor&q,const Tensor&k,const Tensor&v,const Tensor&cache,const global_state::AttentionMetadata&md,float scale){
 r.assertOwnerThread();
 if(r.device().type()!=runtime::Device::Type::kHygon||r.isGraphRecording()||!q||!k||!v||!cache||q->ndim()!=3||q->shape()!=runtime::Shape{q->size(0),3,192}||k->shape()!=q->shape()||v->shape()!=runtime::Shape{q->size(0),3,128}||cache->shape()!=runtime::Shape{132,256,3,320}
 ||q->dtype()!=DT::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()||cache->dtype()!=q->dtype()||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()||cache->device()!=r.device()||!cache->is_contiguous()
 ||!md.block_tables||!*md.block_tables||!md.slot_mapping||!*md.slot_mapping)throw std::invalid_argument("splitFA exact eager C1 H3 K192 V128");
 auto p=plan(q->size(0),md.max_sequence_length);auto table=*md.block_tables,slots=*md.slot_mapping;
 if(table->ndim()!=2||table->size(0)!=1||table->size(1)<128||!table->is_contiguous()||slots->ndim()!=1||slots->size(0)!=p.queries||!slots->is_contiguous()||table->device()!=r.device()||slots->device()!=r.device()||(table->dtype()!=DT::kInt32&&table->dtype()!=DT::kInt64)||(slots->dtype()!=DT::kInt32&&slots->dtype()!=DT::kInt64)||q->stride(2)!=1||k->stride(2)!=1||v->stride(2)!=1)throw std::invalid_argument("splitFA original page table/slot metadata");
 auto check=[](int rc){if(rc)throw std::runtime_error("splitFA private kernel launch error");};
 Tensor out,kt,vt,partial,lse,state,factor,sum,lengths;
 try{
  const int64_t ks[]={int64_t(k->stride(0)),int64_t(k->stride(1))},vs[]={int64_t(v->stride(0)),int64_t(v->stride(1))};
  check(kimi_split_store(cache->data(),k->data(),v->data(),slots->data(),slots->dtype()==DT::kInt32?32:64,p.queries,ks,vs,r.stream()));
  out=Tensor::empty(r,{p.queries,3,128},DT::kBFloat16);
  kt=Tensor::empty(r,{1,3072,3,192},DT::kBFloat16);vt=Tensor::empty(r,{1,3072,3,192},DT::kBFloat16);lengths=Tensor::empty(r,{1},DT::kInt32);
  for(size_t first=0;first<p.queries;){
   // A Q tile never crosses an absolute key-chunk boundary. Previous chunks
   // are fully visible; only current chunk uses FA bottom-right causal mask.
   auto span=tile(p,first);size_t rows=span.rows,end=span.end;
   partial=Tensor::empty(r,{1,rows,3,192},DT::kBFloat16);lse=Tensor::empty(r,{1,3,rows},DT::kFloat32);
   state=Tensor::empty(r,{rows,3,2},DT::kFloat32);factor=Tensor::empty(r,{rows,3,2},DT::kFloat32);sum=Tensor::empty(r,{rows,3,128},DT::kFloat32);
   auto query=q->narrow({{0,first,rows}})->unsqueeze(0);
   for(size_t start=0;start<end;start+=3072){
    size_t count=std::min(size_t(3072),end-start);bool causal=start+count==end;
    check(kimi_split_fa_pack(kt->data(),vt->data(),lengths->data(),cache->data(),table->data(),table->dtype()==DT::kInt32?32:64,start,count,r.stream()));
    infini::ops::FlashAttnWithKvcacheConfig cfg(count);cfg.set_implementation_index(16); // pinned real Hygon delegate/LSE copy
    const std::optional<Tensor> none;
    execution::submit<infini::ops::FlashAttnWithKvcache>(r,execution::RecordingPolicy::DeviceOnly,cfg,query,kt,vt,none,none,none,none,std::optional<Tensor>{lengths},none,none,none,none,std::optional<double>{scale},causal,std::vector<int64_t>{-1,-1},double{0},true,int64_t{1},true,partial,std::optional<Tensor>{lse});
    check(kimi_split_fa_factors(state->data(),factor->data(),lse->data(),rows,start==0,r.stream()));
    check(kimi_split_fa_merge(sum->data(),partial->data(),factor->data(),rows,start==0,r.stream()));
   }
   check(kimi_split_fa_finish(out->data(),sum->data(),state->data(),rows,first,r.stream()));
   partial={};lse={};state={};factor={};sum={};first+=rows;
  }
  return out;
 }catch(...){auto e=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(e);}
}
}
