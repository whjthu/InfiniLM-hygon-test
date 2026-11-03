#pragma once
#include "dispatch.hpp"
#include "runtime/runtime.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
extern "C" int deep_activation_quant_launch_r438(const void*,void*,void*,void*,size_t,void*);
namespace deep438 {
namespace rt=infinilm::runtime;
struct RoleProof{rt::Tensor gu,activation,packed,scale,candidate_activation,candidate_packed,candidate_scale;};
struct Diagnostic{std::array<RoleProof,2>roles;};
inline Diagnostic*diagnostic(){return static_cast<Diagnostic*>(state.diagnostic);}
inline void capture_original(Role role,const rt::Tensor&gu,const rt::Tensor&a,const rt::Tensor&p,const rt::Tensor&s){
 if(auto*d=diagnostic()){auto&v=d->roles.at(role);v.gu=gu;v.activation=a;v.packed=p;v.scale=s;}
}
inline bool overlaps(const rt::Tensor&a,const rt::Tensor&b){
 auto x=reinterpret_cast<uintptr_t>(a->data()),y=reinterpret_cast<uintptr_t>(b->data());return x<y+b->nbytes()&&y<x+a->nbytes();
}
inline void*write_data(const rt::Tensor&t){return const_cast<void*>(static_cast<const void*>(t->data()));}
inline bool eligible(rt::Runtime&r,const rt::Tensor&gu){
 return !r.isGraphRecording()&&r.device().type()==rt::Device::Type::kHygon&&gu&&gu->device()==r.device()&&gu->dtype()==rt::DataType::kBFloat16&&gu->ndim()==2&&gu->size(1)==512&&gu->is_contiguous();
}
inline void apply(rt::Runtime&r,const rt::Tensor&gu,const rt::Tensor&packed,const rt::Tensor&scale,const rt::Tensor&activation={}){
 r.assertOwnerThread();if(!eligible(r,gu)||(gu->size(0)!=1&&gu->size(0)!=6)||!packed||!scale||packed->device()!=r.device()||scale->device()!=r.device()||packed->dtype()!=rt::DataType::kInt8||scale->dtype()!=rt::DataType::kFloat32||packed->shape()!=rt::Shape{gu->size(0),256}||scale->shape()!=rt::Shape{gu->size(0),1}||!packed->is_contiguous()||!scale->is_contiguous())throw std::invalid_argument("r438 dense BF16->I8/F32 shape/dtype/owner differs");
 if(overlaps(gu,packed)||overlaps(gu,scale)||overlaps(packed,scale))throw std::invalid_argument("r438 operand alias");
 if(reinterpret_cast<uintptr_t>(gu->data())%2||reinterpret_cast<uintptr_t>(scale->data())%4)throw std::invalid_argument("r438 operand alignment differs");
 int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=r.device().index())throw std::invalid_argument("r438 current device differs");
 if(activation&&(activation->device()!=r.device()||activation->dtype()!=rt::DataType::kBFloat16||activation->shape()!=rt::Shape{gu->size(0),256}||!activation->is_contiguous()||overlaps(activation,gu)||overlaps(activation,packed)||overlaps(activation,scale)))throw std::invalid_argument("r438 diagnostic shape/alias differs");
 int code=deep_activation_quant_launch_r438(gu->data(),write_data(packed),write_data(scale),activation?write_data(activation):nullptr,gu->size(0),r.stream());
 if(code)throw std::runtime_error("r438 activation quant launch failed:"+std::to_string(code));
}
inline void apply_recorded(rt::Runtime&r,Role role,const rt::Tensor&gu,const rt::Tensor&p,const rt::Tensor&s){
 rt::Tensor a;
 if(auto*d=diagnostic()){
  auto&v=d->roles.at(role);v.gu=gu;v.candidate_packed=p;v.candidate_scale=s;
  a=rt::Tensor::empty(r,{gu->size(0),256},rt::DataType::kBFloat16);v.candidate_activation=a;
 }
 apply(r,gu,p,s,a);
}
}
