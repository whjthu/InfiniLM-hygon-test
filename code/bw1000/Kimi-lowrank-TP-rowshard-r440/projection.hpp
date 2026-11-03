#pragma once
#include "geometry.hpp"
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
 // Tensor leases are released on an exception; success keeps owner-stream FIFO.
 struct FailureDrain {
  rt::Runtime&r;int initial=std::uncaught_exceptions();
  ~FailureDrain()noexcept {if(std::uncaught_exceptions()>initial)try{r.syncStream();}catch(...){std::terminate();}}
 }drain{r};
 const auto g=geometry(which,rank,tp);
 if(!communicator)throw std::invalid_argument("440 actual TP16 communicator required");
 if(input->device()!=r.device()||input->dtype()!=rt::DataType::kBFloat16||
    input->strides()!=rt::Strides{static_cast<rt::Stride>(g.k),1})
  throw std::invalid_argument("440 C1 BF16 contiguous input required");
 // Parent MoE's two exact source roles are unquantized, no-bias, alpha=1
 // ReplicatedLinear. No weight copy/repack, precision change or custom GEMM.
 weight=layer->weight();
 if(!weight||weight->device()!=r.device()||weight->dtype()!=rt::DataType::kBFloat16||
    weight->shape()!=rt::Shape{g.n,g.k}||
    weight->strides()!=rt::Strides{static_cast<rt::Stride>(g.k),1})
  throw std::invalid_argument("440 exact resident replicated weight required");
 shard=weight->narrow({{0,g.row_begin,g.local_n}});
 // Retain the NONE path's explicit contiguity preparation. Both are already
 // contiguous here; only views change, not storage or asynchronous ownership.
 prepared_input=input->contiguous(r);
 prepared_weight=shard->contiguous(r);
 local=infinilm::layers::linear::dense_projection(
  r,prepared_input,prepared_weight,std::optional<rt::Tensor>{},1.0F,
  std::optional<rt::DataType>{rt::DataType::kBFloat16},false);
 output=rt::Tensor::empty(r,{1,g.n},rt::DataType::kBFloat16);
 infinilm::engine::distributed::allgather_(r,output,local,communicator);
 // Full-forward completion, including collective and allocations, remains the
 // eventual consumer's timer. This function claims no asynchronous speedup.
 return output;
}
}
