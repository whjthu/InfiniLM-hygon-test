#pragma once
#include "glm_prefix_proof_r1645.hpp"
#include <cstdint>
#include <limits>
namespace infinilm::models::glm_moe_dsa {
class NativeAttentionProjectedR22;
// Same owner/row/Tensor-identity capability pattern as PrefixProofR1645.
// Only the actual CPU remap producer can construct it. These local H2D tensors
// remain read-only and retained by Work until sparse_impl's existing drain.
class PrefillPrefixProofR8703 final {
 std::shared_ptr<runtime::Runtime> owner_;
 runtime::Tensor indices_,positions_;
 size_t row_,past_,count_,capacity_,width_;
 bool complete_=false;
 friend class NativeAttentionProjectedR22;
 PrefillPrefixProofR8703(runtime::Runtime&r,const runtime::Tensor&indices,const runtime::Tensor&positions,size_t row,size_t past,size_t count,size_t capacity,const std::vector<int64_t>&logical,const std::vector<int64_t>&mapped)
 :owner_(r.shared_from_this()),indices_(indices),positions_(positions),row_(row),past_(past),count_(count),capacity_(capacity),width_(indices&&indices->ndim()==2?indices->size(1):0){
  r.assertOwnerThread();
  if(r.isGraphRecording()||count<2||past>capacity||count>capacity-past||capacity>2048||!indices||!positions||indices->ndim()!=2||indices->shape()!=runtime::Shape{count,width_}||!width_||width_>2048||indices->device()!=r.device()||positions->device()!=r.device()||indices->dtype()!=runtime::DataType::kInt64||positions->dtype()!=runtime::DataType::kInt64||!indices->is_contiguous()||!positions->is_contiguous()||positions->shape()!=runtime::Shape{logical.size()}||mapped.size()!=count*width_)
   throw std::invalid_argument("8703 invalid CPU prefix-plan producer geometry");
  const size_t prefix=past+count;
  if(logical.size()!=prefix)return;
  for(size_t j=0;j<prefix;++j)if(logical[j]!=int64_t(j))return;
  // No inference from past/count. Verify every query's *actual CPU plan*.
  // Count unique causal mapped IDs without a tree/set or a device readback.
  std::vector<size_t>seen(prefix,0);
  for(size_t t=0;t<count;++t){size_t unique=0;
   for(size_t j=0;j<width_;++j){auto id=mapped[t*width_+j];
    if(id<0||size_t(id)>=prefix)throw std::invalid_argument("8703 CPU mapped ID outside prefix");
    if(size_t(id)<=past+t&&seen[size_t(id)]!=t+1){seen[size_t(id)]=t+1;++unique;}
   }
   if(unique!=past+t+1)return;
  }
  complete_=true;
 }
public:
 PrefillPrefixProofR8703(const PrefillPrefixProofR8703&)=delete;
 PrefillPrefixProofR8703&operator=(const PrefillPrefixProofR8703&)=delete;
 bool complete()const noexcept{return complete_;}
 bool matches(runtime::Runtime&r,const runtime::Tensor&indices,const runtime::Tensor&positions,size_t row,size_t past,size_t count,size_t capacity,size_t keys)const{
  r.assertOwnerThread();if(owner_.get()!=&r||r.isGraphRecording())throw std::invalid_argument("8703 foreign Runtime or graph consumer");
  return complete_&&row==row_&&past==past_&&count==count_&&capacity==capacity_&&keys==past_+count_&&indices&&positions&&indices.operator->()==indices_.operator->()&&positions.operator->()==positions_.operator->()&&indices->data()==indices_->data()&&positions->data()==positions_->data()&&indices->device()==r.device()&&positions->device()==r.device()&&indices->dtype()==runtime::DataType::kInt64&&positions->dtype()==runtime::DataType::kInt64&&indices->shape()==runtime::Shape{count_,width_}&&positions->shape()==runtime::Shape{keys}&&indices->is_contiguous()&&positions->is_contiguous();
 }
};
runtime::Tensor sparse_attention_mapped_bounded_proven_r8703(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,size_t,float,const PrefillPrefixProofR8703&,size_t,size_t);
}
