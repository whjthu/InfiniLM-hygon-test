#pragma once
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace infinilm::models::glm_moe_dsa::value_r433 {
namespace drt=runtime;
// A bounded first-warm snapshot, not a cache or an execution path. Values are
// original query-produced tensors; nothing is recomputed for the snapshot.
struct Diagnostic {
 int detail{},mode{};size_t ticket{};bool quiesced=false,query_completed=false;
 std::string phase="armed",query_error;
 std::vector<std::pair<std::string,drt::Tensor>>tensors;
 Diagnostic(int d,int m,size_t n):detail(d),mode(m),ticket(n){}
 void add(const std::string&name,const drt::Tensor&t){
  if(!t||tensors.size()>=24)throw std::invalid_argument("r433 bounded diagnostic tensor");
  for(const auto&v:tensors)if(v.first==name)throw std::invalid_argument("r433 duplicate diagnostic tensor");
  if(t->dtype()!=drt::DataType::kUInt8&&t->dtype()!=drt::DataType::kBFloat16&&t->dtype()!=drt::DataType::kFloat32)throw std::invalid_argument("r433 diagnostic dtype");
  size_t bytes=t->nbytes();for(const auto&v:tensors)bytes+=v.second->nbytes();if(bytes>32*1024*1024)throw std::invalid_argument("r433 diagnostic32MiB bound");
  tensors.emplace_back(name,t);
 }
 nlohmann::json descriptor()const{
  if(!quiesced||tensors.empty())throw std::logic_error("r433 no quiesced diagnostic tensors");
  nlohmann::json slots=nlohmann::json::array();size_t total=0;
  for(size_t i=0;i<tensors.size();++i){const auto&[name,t]=tensors[i];const char*d=t->dtype()==drt::DataType::kUInt8?"U8":t->dtype()==drt::DataType::kBFloat16?"BF16":"F32";total+=t->nbytes();slots.push_back({{"key",i},{"name",name},{"shape",t->shape()},{"dtype",d},{"nbytes",t->nbytes()}});}
  return{{"ticket",ticket},{"mode",mode},{"detail",detail},{"quiesced",quiesced},{"query_CAPI_completed",query_completed},{"phase",phase},{"query_error",query_error},{"total_bytes",total},{"slots",slots},{"scope","actual firstwarm query tensors;D2HoutsideCAPI;noformal snapshot"}};
 }
};
}
