#pragma once
#include "contract.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <exception>
namespace kimi_split_v128 {
inline infinilm::runtime::Tensor allocate(infinilm::runtime::Runtime&r){
 using namespace infinilm::runtime;r.assertOwnerThread();
 if(r.device().type()!=Device::Type::kHygon||r.isGraphRecording())throw std::invalid_argument("splitV private eager Hygon only");
 // ONE exact owning Tensor; K/V are views of final [192|128] feature axis.
 auto storage=Tensor::empty_parameter(r,{pages,page,heads,stride},DataType::kBFloat16);
 try{r.setDeviceMemoryAsync(storage->data(),0,storage->nbytes(),r.stream());}
 catch(...){auto e=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(e);}
 return storage;
}
}
