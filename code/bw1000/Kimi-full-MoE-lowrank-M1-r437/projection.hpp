#pragma once
#include "runtime/runtime.hpp"
#include "layers/linear/linear.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" int kimi_lowrank_gemv_r432(int,void*,const void*,const void*,size_t,size_t,size_t,void*);
namespace kimi_r437 {
namespace rt=infinilm::runtime;
inline thread_local bool selected=false; // Default OFF; set by this exact owner only.
inline thread_local std::array<uint64_t,2> hits{};
inline rt::Tensor project(rt::Runtime&r,const std::shared_ptr<infinilm::nn::Linear>&layer,rt::Tensor&input,int which){
 if(!selected)return layer->forward(r,input);
 r.assertOwnerThread();
 rt::Tensor weight,output;
 // Construct before every new throwing contract check, after owned buffers:
 // the router may already be queued by the full original forward.
 struct FailureDrain{rt::Runtime&r;int count=std::uncaught_exceptions();~FailureDrain()noexcept{if(std::uncaught_exceptions()>count)try{r.syncStream();}catch(...){std::terminate();}}}drain{r};
 const size_t n=which==0?3584:7168,k=which==0?7168:3584;
 if(which<0||which>1||r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!layer||!input||input->device()!=r.device()||input->dtype()!=rt::DataType::kBFloat16||input->shape()!=rt::Shape{1,k}||input->strides()!=rt::Strides{static_cast<rt::Stride>(k),1})throw std::invalid_argument("437 only original exact C1 BF16 contiguous lowrank projection");
 int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=r.device().index())throw std::invalid_argument("437 current device must be owning Runtime device");
 // Parameter and input Tensors stay strongly owned by the MoE/module and caller;
 // no private storage pointers or interposer. A parameter copy also covers launch.
 weight=layer->weight();
 if(!weight||weight->device()!=r.device()||weight->dtype()!=rt::DataType::kBFloat16||weight->shape()!=rt::Shape{n,k}||weight->strides()!=rt::Strides{static_cast<rt::Stride>(k),1})throw std::invalid_argument("437 original resident BF16 weight shape/stride");
 output=rt::Tensor::empty(r,{1,n},rt::DataType::kBFloat16);
 // Declared after borrowed/owned Tensors: exception drain precedes their release.
 // Successful submission adds no sync; the original full-forward drain completes it.
 const int rc=kimi_lowrank_gemv_r432(which,output->data(),input->data(),weight->data(),n,k,k,r.stream());
 if(rc)throw std::runtime_error("437 reused432 GEMV launch rc="+std::to_string(rc));
 ++hits.at(which);return output;
}
}
