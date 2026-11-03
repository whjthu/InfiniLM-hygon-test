#pragma once
#include "geometry.hpp"
#include "single_row_span.hpp"
#include "lease_scope.hpp"
#include "runtime/runtime.hpp"
#include "layers/linear/linear.hpp"
#include "layers/linear/dense.hpp"
#include "engine/distributed/communication_group.hpp"
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
namespace kimi_r440 {
namespace rt=infinilm::runtime;
// Private source candidate only. No setter alters an installed module. The
// eventual MoE owner passes enabled explicitly; omission keeps the old path.
inline rt::Tensor project(rt::Runtime&r,
 const std::shared_ptr<infinilm::nn::Linear>&layer,rt::Tensor&input,
 int which,std::size_t rank,std::size_t tp,infinicclComm_t communicator,
 bool enabled=false) {
 r.assertOwnerThread();
 if(!layer||!input)throw std::invalid_argument("440 requires owning layer and input");
 if(!enabled||input->ndim()!=2||r.device().type()!=rt::Device::Type::kHygon||
    r.isGraphRecording()||!eligible(input->size(0),input->size(1),tp,which))
  return layer->forward(r,input);
 rt::Tensor weight,shard,local,output,prepared_input,prepared_weight;
 // Router/expert work may already be queued. Drain before these temporary
 // Tensor leases are released on an exception. Success additionally retains all
 // new tensors in the full-forward caller scope until proven completion.
 struct FailureDrain {
  rt::Runtime&r;int initial=std::uncaught_exceptions();
  ~FailureDrain()noexcept {if(std::uncaught_exceptions()>initial)try{r.syncStream();}catch(...){std::terminate();}}
 }drain{r};
 auto*scope=active_scope;if(!scope)throw std::invalid_argument("440 enabled enqueue without caller lease scope");scope->require(r,tp,communicator);if(scope->rank()!=int(rank))throw std::invalid_argument("440 actual global rank mismatch");
 const auto g=geometry(which,rank,tp);
 if(!communicator)throw std::invalid_argument("440 actual TP16 communicator required");
 if(input->device()!=r.device()||input->dtype()!=rt::DataType::kBFloat16||
    input->strides().size()!=2||!physical_single_row(input->ndim(),input->size(0),input->size(1),g.k,input->stride(0),input->stride(1)))
  throw std::invalid_argument("440 exact 2D C1 BF16 physical single row required");
 // Parent MoE's two exact source roles are unquantized, no-bias, alpha=1
 // ReplicatedLinear. No weight copy/repack, precision change or custom GEMM.
 weight=layer->weight();
 if(!weight||weight->device()!=r.device()||weight->dtype()!=rt::DataType::kBFloat16||
    weight->shape()!=rt::Shape{g.n,g.k}||
    weight->strides()!=rt::Strides{static_cast<rt::Stride>(g.k),1})
  throw std::invalid_argument("440 exact resident replicated weight required");
 shard=weight->narrow({{0,g.row_begin,g.local_n}});
 // Keep the original NONE preparation: positive singleton rowpitch may be2K,
 // so input.contiguous may copy. All preparation/allocation costs are retained.
 prepared_input=input->contiguous(r);
 prepared_weight=shard->contiguous(r);
 local=infinilm::layers::linear::dense_projection(
  r,prepared_input,prepared_weight,std::optional<rt::Tensor>{},1.0F,
  std::optional<rt::DataType>{rt::DataType::kBFloat16},false);
 output=rt::Tensor::empty(r,{1,g.n},rt::DataType::kBFloat16);
 // Keep every new Tensor through the caller's actual full-forward completion,
 // not merely until the eager CCL enqueue returns. Reserve14 occurs in timed scope.
 for(const auto&t:{input,weight,shard,prepared_input,prepared_weight,local,output})scope->retain(t);
 infinilm::engine::distributed::allgather_(r,output,local,communicator);scope->submitted(which);
 // Full-forward completion, including collective and allocations, remains the
 // eventual consumer's timer, including scope release. No async speedup claim.
 return output;
}
inline rt::Tensor project_scoped(rt::Runtime&r,const std::shared_ptr<infinilm::nn::Linear>&layer,rt::Tensor&input,int which,size_t tp,infinicclComm_t comm){auto*scope=active_scope;return project(r,layer,input,which,scope?size_t(scope->rank()):0,tp,comm,scope&&scope->enabled());}
}
