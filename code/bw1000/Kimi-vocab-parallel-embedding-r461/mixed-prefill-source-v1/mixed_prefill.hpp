#pragma once
// SOURCE-ONLY, UNCOMPILED, UNWIRED. No existing symbol is interposed/replaced.
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <base/gemm.h>
#include <base/cast.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <cstddef>
namespace kimi_mixed_prefill_r461 {
inline infinilm::runtime::Tensor project(infinilm::runtime::Runtime &r,
 const infinilm::runtime::Tensor &input,const infinilm::runtime::Tensor &weight,
 const infinilm::runtime::Tensor &output,float alpha) {
 using namespace infinilm;using DT=runtime::DataType;
 r.assertOwnerThread();
 if(r.device().type()!=runtime::Device::Type::kHygon
    ||r.graphCaptureState()!=execution::graph::GraphManager::CaptureState::kInactive
    ||!input||!weight||!output||input->ndim()!=2||weight->ndim()!=2||output->ndim()!=2
    ||input->dtype()!=DT::kBFloat16||weight->dtype()!=DT::kBFloat16||output->dtype()!=DT::kBFloat16
    ||input->size(0)<512||input->size(1)!=weight->size(0)
    ||output->size(0)!=input->size(0)||output->size(1)!=weight->size(1)
    ||input->device()!=r.device()||weight->device()!=r.device()||output->device()!=r.device())
  throw std::invalid_argument("mixed prefill exact eager BF16 operands/F32 sum geometry");
 runtime::Tensor x_source,w_source,w,sum32;
 try {
  x_source=input->contiguous(r);
  const bool transposed=weight->stride(0)==1&&weight->stride(1)==std::ptrdiff_t(weight->size(0));
  w_source=transposed?weight->permute({1,0})->contiguous(r):weight->contiguous(r);
  w=transposed?w_source->permute({1,0}):w_source;
  sum32=runtime::Tensor::empty(r,output->shape(),DT::kFloat32);
  // Current Hygon BlasGemm passes independent A/B/Y types; its BlasUtils uses
  // CUDA_R_16BF for these operands, CUDA_R_32F for Y, CUBLAS_COMPUTE_32F.
  // Do NOT select FullFp32 config: that metadata policy requires F32 operands.
  execution::submit<infini::ops::Gemm>(r,execution::RecordingPolicy::DeviceOnly,
   infini::ops::Gemm::DefaultConfig(r.device().type()),x_source,w,
   std::optional<runtime::Tensor>{},std::optional<float>{alpha},std::optional<float>{0},
   std::optional<int>{},std::optional<int>{},sum32);
  execution::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),sum32,output);
  // Original r2475 return/stream lifetime: same F32 sum retained by caller.
  // No permanent FP32 weights, no extra success fence, no SDK/class ABI change.
  return sum32;
 }catch(...){
  const auto failure=std::current_exception();
  try{r.syncStream();}catch(...){std::terminate();}
  std::rethrow_exception(failure);
 }
}
}
