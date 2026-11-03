#pragma once
#include "fp32_leaves_r446.h"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <infini/rt/device_guard.h>
#include <thread>
#include <stdexcept>
#include <cstdint>
#include <limits>

namespace infinilm::models::glm_moe_dsa::value_r446 {
namespace rt=runtime;
struct Owned final {
 rt::Runtime&r;void*stream;std::thread::id owner;rt::Tensor flag;
 uint32_t zero=0,host_status=0;size_t casts=0,aggregations=0,projections=0;bool finished=false;
 void check()const{
  r.assertOwnerThread();if(owner!=std::this_thread::get_id()||r.stream()!=stream||r.device().type()!=rt::Device::Type::kHygon||r.isGraphRecording())throw std::invalid_argument("r446 owner/stream/device/eager");
 }
 explicit Owned(rt::Runtime&runtime):r(runtime),stream(r.stream()),owner(std::this_thread::get_id()){
  try{
   check();infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
   flag=rt::Tensor::empty(r,{4},rt::DataType::kUInt8);
   if(reinterpret_cast<uintptr_t>(flag->data())%4)throw std::logic_error("r446 aligned flag");
   r.memcpyH2D(const_cast<std::byte*>(flag->data()),&zero,sizeof(zero),false);
  }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
 }
 Owned(const Owned&)=delete;
 ~Owned(){if(!finished)try{check();r.syncStream();}catch(...){std::terminate();}}
 void tensor(const rt::Tensor&t,rt::DataType dtype,rt::Shape shape)const{
  if(!t||t->device()!=r.device()||t->dtype()!=dtype||t->shape()!=shape||!t->is_contiguous())throw std::invalid_argument("r446 exact owned tensor view");
 }
 static void disjoint(const rt::Tensor&a,const rt::Tensor&b){
  auto x=reinterpret_cast<uintptr_t>(a->data()),y=reinterpret_cast<uintptr_t>(b->data());
  if(a->nbytes()>std::numeric_limits<uintptr_t>::max()-x||b->nbytes()>std::numeric_limits<uintptr_t>::max()-y||!(x+a->nbytes()<=y||y+b->nbytes()<=x))throw std::invalid_argument("r446 aliased operands");
 }
 void cast(const rt::Tensor&x,const rt::Tensor&y){
  check();if(finished||!x||!y||x->shape()!=y->shape()||(x->numel()!=16384&&x->numel()!=2048))throw std::invalid_argument("r446 cast geometry/state");
  const int direction=x->dtype()==rt::DataType::kBFloat16&&y->dtype()==rt::DataType::kFloat32?0:x->dtype()==rt::DataType::kFloat32&&y->dtype()==rt::DataType::kBFloat16?1:-1;
  if(direction<0)throw std::invalid_argument("r446 cast dtype");tensor(x,x->dtype(),x->shape());tensor(y,y->dtype(),y->shape());disjoint(x,y);
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
  int rc=glm_value_cast_r446(r.device().index(),direction,x->numel(),x->data(),const_cast<std::byte*>(y->data()),reinterpret_cast<uint32_t*>(const_cast<std::byte*>(flag->data())),stream);
  if(rc)throw std::runtime_error("r446 Cast launch "+std::to_string(rc));++casts;
 }
 void gemv(int kind,const rt::Tensor&x,const rt::Tensor&w,const rt::Tensor&y){
  check();if(finished||(kind!=0&&kind!=1))throw std::invalid_argument("r446 GEMV state/kind");
  size_t k=kind==0?2048:512,n=kind==0?512:256;
  tensor(x,rt::DataType::kFloat32,{1,k});tensor(w,rt::DataType::kFloat32,{k,n});tensor(y,rt::DataType::kFloat32,{1,n});disjoint(x,y);disjoint(w,y);
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
  int rc=glm_value_gemv_r446(r.device().index(),kind,reinterpret_cast<const float*>(x->data()),reinterpret_cast<const float*>(w->data()),reinterpret_cast<float*>(const_cast<std::byte*>(y->data())),reinterpret_cast<uint32_t*>(const_cast<std::byte*>(flag->data())),stream);
  if(rc)throw std::runtime_error("r446 F32GEMV launch "+std::to_string(rc));if(kind==0)++aggregations;else ++projections;
 }
 void finish(){
  check();if(finished||casts!=3||aggregations!=8||projections!=8)throw std::logic_error("r446 complete3Cast16GEMV contract");
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
  // The pinned Runtime memcpyD2H is completion-returning (its original method
  // synchronizes the stream). This one read closes all preceding leaf use.
  r.memcpyD2H(&host_status,flag->data(),sizeof(host_status));
  if(host_status)throw std::runtime_error("r446 nonfinite leaf status "+std::to_string(host_status));finished=true;
 }
};
}
