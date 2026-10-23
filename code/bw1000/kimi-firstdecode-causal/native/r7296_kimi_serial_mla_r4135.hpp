#pragma once
#include "layers/attention/backends/attention_layer.hpp"
#include "global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include <array>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <algorithm>
namespace infinilm::models::kimi_native::serial_mla_r4135 {
namespace rt=infinilm::runtime;
inline std::array<std::array<std::atomic<uint64_t>,3>,8> counters{};
extern "C" int modern_kimi_serial_mla_counts_r4135(int device,uint64_t*out,size_t n){
 if(device<0||device>=8||!out||n!=3)return 1;
 for(size_t i=0;i<3;++i)out[i]=counters[device][i].load();return 0;
}
inline rt::Tensor ints(rt::Runtime&r,const std::vector<int32_t>&values,rt::Shape shape){
 auto t=rt::Tensor::empty(r,shape,rt::DataType::kInt32);
 r.memcpyH2D(t->data(),values.data(),values.size()*sizeof(int32_t),false);return t;
}
inline rt::Tensor forward(infinilm::layers::attention::AttentionLayer&layer,
 const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,
 infinilm::global_state::ForwardContext&source,bool per_query_bound){
 auto&r=source.require_runtime();r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon
  ||r.device().index()<0||r.device().index()>=8||!q||!k||!v
  ||q->device()!=r.device()||k->device()!=r.device()||v->device()!=r.device()
  ||q->ndim()!=3||q->shape()!=k->shape()||q->shape()!=v->shape()
  ||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=q->dtype()||v->dtype()!=q->dtype()
  ||q->size(0)<2||q->size(0)>128)
  throw std::invalid_argument("4135 bounded eager MLA verification contract");
 auto&a=source.attn_metadata;
 if(!a.total_sequence_lengths||!a.input_offsets||!a.block_tables||!a.slot_mapping)
  throw std::invalid_argument("4135 MLA metadata missing");
 auto lens=*a.total_sequence_lengths,offsets=*a.input_offsets;
 auto tables=*a.block_tables,slots=*a.slot_mapping;
 const size_t N=q->size(0),B=lens->size(0),H=q->size(1),D=q->size(2);
 if(B<1||B>8||lens->ndim()!=1||offsets->ndim()!=1||offsets->size(0)!=B+1
  ||tables->ndim()!=2||tables->size(0)!=B||tables->size(1)<1
  ||slots->ndim()!=1||slots->size(0)!=N||slots->dtype()!=rt::DataType::kInt64
  ||lens->dtype()!=rt::DataType::kInt32||offsets->dtype()!=rt::DataType::kInt32
  ||!lens->is_contiguous()||!offsets->is_contiguous())
  throw std::invalid_argument("4135 MLA metadata shape/dtype");
 std::vector<int32_t> host_lens(B),host_offsets(B+1);
 r.syncStream();r.memcpyD2H(host_lens.data(),lens->data(),B*4);
 r.memcpyD2H(host_offsets.data(),offsets->data(),(B+1)*4);r.syncStream();
 if(host_offsets.front()!=0||host_offsets.back()!=int32_t(N))
  throw std::invalid_argument("4135 packed query offsets");
 std::vector<int32_t> row_of(N),position(N),past_of(N),cu(N*2),one_offsets{0,1};
 for(size_t row=0;row<B;++row){
  int width=host_offsets[row+1]-host_offsets[row],past=host_lens[row]-width;
  if(width<1||width>17||past<1||host_lens[row]<=past
   ||(a.logical_kv_upper_bound&&size_t(host_lens[row])>a.logical_kv_upper_bound))
   throw std::invalid_argument("4135 prefix/width/bound");
  for(int j=0;j<width;++j){size_t token=size_t(host_offsets[row]+j);
   row_of[token]=row;position[token]=past+j+1;past_of[token]=past+j;
   cu[2*token]=0;cu[2*token+1]=past+j+1;
  }
 }
 auto lengths_tensor=ints(r,position,{N}),pasts_tensor=ints(r,past_of,{N});
 auto cu_tensor=ints(r,cu,{N,2}),offset_tensor=ints(r,one_offsets,{2});
 auto output=rt::Tensor::empty(r,{1,N,H*D},q->dtype());
 std::vector<infinilm::global_state::ForwardContext> held;held.reserve(N);
 std::vector<rt::Tensor> results;results.reserve(N);
 // Drain before any held metadata/result is destroyed, also during unwinding.
 struct Drain{rt::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{r};
 for(size_t token=0;token<N;++token){
  held.push_back(source);auto&f=held.back();
  f.attn_metadata=infinilm::global_state::AttentionMetadata{};
  auto&single=f.attn_metadata;
  single.past_sequence_lengths=pasts_tensor->narrow({{0,token,1}});
  single.total_sequence_lengths=lengths_tensor->narrow({{0,token,1}});
  single.input_offsets=offset_tensor;
  single.cu_seqlens=cu_tensor->narrow({{0,token,1}})->view({2});
  single.block_tables=tables->narrow({{0,size_t(row_of[token]),1}});
  single.slot_mapping=slots->narrow({{0,token,1}});
  single.max_query_length=1;single.max_sequence_length=position[token];
  single.logical_kv_upper_bound=per_query_bound?size_t(position[token]):a.logical_kv_upper_bound;
  auto q_row=q->narrow({{0,token,1}}), k_row=k->narrow({{0,token,1}}), v_row=v->narrow({{0,token,1}});
  results.push_back(layer.forward(q_row,k_row,v_row,f));
  if(results.back()->shape()!=rt::Shape({1,1,H*D}))throw std::logic_error("4135 MLA result shape");
  output->narrow({{1,token,1}})->copy_from(r,results.back());
 }
 auto&count=counters[r.device().index()];count[0].fetch_add(1);count[1].fetch_add(N);count[2].fetch_add(1);
 return output;
}
}
