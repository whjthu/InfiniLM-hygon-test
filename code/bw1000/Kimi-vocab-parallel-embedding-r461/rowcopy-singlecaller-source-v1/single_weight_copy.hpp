#pragma once
// SOURCE-ONLY proposal. No Model loader hook or registry mutation installed.
#include "weights/weights.hpp"
#include "runtime/runtime.hpp"
#include "row_copy.hpp"
#include <exception>
#include <memory>
#include <vector>
#include <stdexcept>
namespace kimi_row_singlecaller_r461 {
// Caller proves this is the real r201 tp_dim1/rank/TP32 registered W2 binding.
// Caller retains global CPU bytes through this synchronous call (FFI pending
// source owner or an actual owning Tensor); no asynchronous admission is added.
inline void load(infinilm::runtime::Runtime &runtime,
                 const infinilm::weights::Weight &registered,
                 const infinilm::runtime::Tensor &global,
                 std::size_t rank,bool candidate) {
 using namespace infinilm;
 runtime.assertOwnerThread();
 const auto target=registered.tensor();
 if(runtime.device().type()!=runtime::Device::Type::kHygon||runtime.isGraphRecording()
    ||runtime.graphCaptureState()!=execution::graph::GraphManager::CaptureState::kInactive
    ||rank>=32||!target||!global||target->device()!=runtime.device()
    ||global->device().type()!=runtime::Device::Type::kCpu
    ||target->dtype()!=runtime::DataType::kUInt8||global->dtype()!=runtime::DataType::kUInt8
    ||target->ndim()!=2||global->ndim()!=2||target->size(0)!=3584||global->size(0)!=3584
    ||!target->is_contiguous()||!global->is_contiguous()
    ||(target->size(1)!=48&&target->size(1)!=3)
    ||global->size(1)!=target->size(1)*32)
  throw std::invalid_argument("singlecaller exact TP32 W2/scale geometry");
 const auto width=target->size(1);const auto address=target->data();
 // Transaction boundary: single-weight consumer only, no outstanding LoadBatch.
 runtime.syncStream();
 std::shared_ptr<std::vector<std::byte>> storage;
 runtime::Tensor selected,packed;
 try {
  if(!candidate){
   // Public copy of Weight preserves every original private partition field.
   auto original=registered;original.load(runtime,global);
  }else{
   selected=global->narrow({{1,rank*width,width}});
   storage=std::make_shared<std::vector<std::byte>>(3584*width);
   const runtime::Shape shape{3584,width};
   const std::vector<std::ptrdiff_t> ds{std::ptrdiff_t(width),1};
   const auto ss=selected->strides();
   runtime::detail::CpuCopyStatsR2515 stats;
   if(!private_r436::try_row_copy(storage->data(),selected->data(),shape,ds,ss,1,stats)
      ||stats.blocks!=3584||stats.block_bytes!=3584*width)
    throw std::logic_error("singlecaller actual selection did not choose original row helper");
   packed=runtime::Tensor::from_blob(storage->data(),shape,runtime::DataType::kUInt8,
                                   runtime::Device{runtime::Device::Type::kCpu,0},storage);
   // Ephemeral LOCAL source policy, SAME registered GPU Tensor. It is never
   // bind_weight'ed and never stored into the model's registry/state_dict.
   weights::Weight local_alias(target,0,0,1);
   if(local_alias.tensor()->data()!=address)throw std::logic_error("backing identity");
   local_alias.load(runtime,packed); // Original public load includes syncStream.
  }
  if(registered.tensor()->data()!=address)throw std::logic_error("registry backing changed");
 }catch(...){
  const auto failure=std::current_exception();
  try{runtime.syncStream();}catch(...){std::terminate();}
  std::rethrow_exception(failure);
 }
 // selected/global, target, packed and storage survive the successful load's
 // completion boundary. There is no source reuse/registry lease cache.
}
}
