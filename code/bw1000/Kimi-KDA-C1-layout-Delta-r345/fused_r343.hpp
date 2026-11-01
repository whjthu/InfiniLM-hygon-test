#pragma once
#include "runtime/runtime.hpp"
#include <unordered_map>
#include <array>
#include <nlohmann/json.hpp>
#include "layout_r345.hpp"
extern "C" int search_run(int,void**,void*);
namespace fused_r343 {
namespace rt=infinilm::runtime;
struct S{bool on=false;int read=-1,write=-1;size_t T=0;std::array<const void*,3>meta{};uint64_t chosen=0,fallback=0;std::string diag;};
inline thread_local std::unordered_map<rt::Runtime*,S> entries;
inline void mode(rt::Runtime&r,bool on){r.assertOwnerThread();entries[&r].on=on;}
inline void bind(rt::Runtime&r,size_t T,int read,int write,std::array<rt::Tensor,3>m){auto&s=entries[&r];s.T=T;s.read=read;s.write=write;for(size_t j=0;j<3;++j)s.meta[j]=m[j]->data();}
inline bool run(rt::Runtime&r,std::array<rt::Tensor,12>x,float scale,float lower,bool norm){
 r.assertOwnerThread();auto it=entries.find(&r);if(it==entries.end()||!it->second.on)return false;auto&s=it->second;
 bool yes=!r.isGraphRecording()&&s.T==1&&s.read==1&&s.write==2&&norm&&lower==-5.f&&std::abs(scale-0.08838834764831844f)<1e-8f;
 static const std::vector<rt::Shape>expected{{1,1,6,128},{3,6,128,128},{1,1,6,128},{1,1,6,128},{1,1,6,128},{1,1,6,128},{1,1,6},{6},{6,128},{2},{1},{1}};
 nlohmann::json metadata;
 for(size_t j=0;j<12;++j){if(!x[j])throw std::invalid_argument("r345 null operand");bool linear=layout_r345::accept(x[j]->shape(),x[j]->strides(),expected[j]);yes &= x[j]->device()==r.device()&&linear;
  if(s.diag.empty())metadata.push_back({{"operand",j},{"shape",x[j]->shape()},{"strides",x[j]->strides()},{"dtype",int(x[j]->dtype())},{"strict_contiguous",x[j]->is_contiguous()},{"fixedC1_linear",linear}});
 }
 yes &= x[0]->shape()==rt::Shape{1,1,6,128}&&x[1]->shape()==rt::Shape{3,6,128,128};
 for(size_t j:{size_t(0),size_t(1),size_t(2),size_t(3),size_t(4),size_t(5),size_t(6)})yes &= x[j]->dtype()==rt::DataType::kBFloat16;
 yes &= x[7]->dtype()==rt::DataType::kFloat32&&x[8]->dtype()==rt::DataType::kFloat32;
 for(size_t j=0;j<3;++j)yes &= x[9+j]->data()==s.meta[j]&&x[9+j]->dtype()==rt::DataType::kInt32&&x[9+j]->numel()==(j?1:2);
 if(s.diag.empty())s.diag=nlohmann::json{{"eligible",yes},{"T",s.T},{"read",s.read},{"write",s.write},{"scale",scale},{"lower",lower},{"norm",norm},{"metadata",metadata}}.dump();
 if(!yes){++s.fallback;return false;}
 void*p[14]={x[1]->data(),x[2]->data(),x[3]->data(),x[4]->data(),x[5]->data(),x[6]->data(),x[7]->data(),x[8]->data(),x[0]->data(),nullptr,x[9]->data(),x[10]->data(),x[11]->data(),nullptr};
 if(search_run(2,p,r.stream()))throw std::runtime_error("r343 fused KDA launch failed");++s.chosen;return true;
}
inline void clear(rt::Runtime&r){entries.erase(&r);}
}
