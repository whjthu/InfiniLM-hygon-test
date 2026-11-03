#pragma once
#include "runtime/runtime.hpp"
#include "layers/linear/linear.hpp"
#include "row_contract.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
extern "C" int kimi_lowrank_gemv_r432(int,void*,const void*,const void*,size_t,size_t,size_t,void*);
namespace kimi_r437 {
namespace rt=infinilm::runtime;using J=nlohmann::json;
struct Predicates {
 std::array<std::pair<const char*,bool>,24> entries{};size_t count=0;int current_device=-1,device_read_rc=-999;
 void add(const char*name,bool value){if(count>=entries.size())throw std::logic_error("437 predicate capacity");entries[count++]={name,value};}
 bool get(const char*name)const{for(size_t i=0;i<count;++i)if(std::string_view(entries[i].first)==name)return entries[i].second;throw std::logic_error("437 missing predicate");}
 bool all()const{for(size_t i=0;i<count;++i)if(!entries[i].second)return false;return true;}
 J json()const{J q=J::object();for(size_t i=0;i<count;++i)q[entries[i].first]=entries[i].second;return q;}
};
inline thread_local bool selected=false;
inline thread_local std::array<uint64_t,2> hits{};
struct DiagnosticContext{rt::Tensor canary_storage;std::string stem;size_t arm=0;int local_rank=-1,global_rank=-1,node=-1;};
inline thread_local DiagnosticContext diagnostic;
struct DiagnosticScope {
 DiagnosticScope(const rt::Tensor&storage,const std::string&stem,int rank,int global,int node){if(rank<0||rank>=8||node<0||node>1||global!=node*8+rank)throw std::invalid_argument("437 exact actual rank metadata");if(diagnostic.canary_storage)throw std::logic_error("437 diagnostic scope already active");diagnostic.canary_storage=storage;diagnostic.stem=stem;diagnostic.local_rank=rank;diagnostic.global_rank=global;diagnostic.node=node;}
 ~DiagnosticScope(){diagnostic.canary_storage.reset();diagnostic.stem.clear();diagnostic.arm=0;diagnostic.local_rank=diagnostic.global_rank=diagnostic.node=-1;}
};
inline J describe(const rt::Tensor&t,const char*role){
 J v={{"role",role},{"present",bool(t)},{"underlying_Runtime_pool_offset","UNKNOWN_not_exposed_publicly"}};
 if(!t)return v;
 const auto address=reinterpret_cast<uintptr_t>(t->data());
 v.update({{"data_address",address},{"dtype",rt::toString(t->dtype())},{"shape",t->shape()},{"strides",t->strides()},{"numel",t->numel()},{"nbytes_logical",t->nbytes()},{"device",t->device().ToString()},{"is_contiguous_strict",t->is_contiguous()}});
 if(diagnostic.canary_storage){
  const auto origin=reinterpret_cast<uintptr_t>(diagnostic.canary_storage->data());const auto extent=diagnostic.canary_storage->nbytes();
  v["strong_canary_storage_data_address"]=origin;v["strong_canary_storage_bytes"]=extent;
  if(address>=origin&&address-origin<extent)v["observed_byte_displacement_from_canary_storage"]=address-origin;
  else{v["observed_byte_displacement_from_canary_storage"]=nullptr;v["canary_storage_relation"]="different_address_range";}
 }
 if(std::string(role)=="fresh_output")v["logical_offset0_source_guarantee"]="Tensor::empty fresh nonview; not a dynamic pool-offset observation";
 if(std::string(role)=="full_weight")v["logical_origin"]="current complete weight Tensor::data(); no private allocation-base access";
 return v;
}
inline void persist(rt::Runtime&r,int which,const rt::Tensor&input,const rt::Tensor&weight,const rt::Tensor&output,const Predicates&pred,const char*phase,const std::string&error=""){
 if(diagnostic.stem.empty())throw std::logic_error("437 diagnostic scope not configured by actual owner");
 J q={{"which",which},{"arm",diagnostic.arm},{"local_rank",diagnostic.local_rank},{"global_tp_rank",diagnostic.global_rank},{"node_rank",diagnostic.node},{"phase",phase},{"error",error},{"Runtime_device",r.device().ToString()},{"input",describe(input,"input")},{"weight",describe(weight,"full_weight")},{"output",describe(output,"fresh_output")},{"guard_predicates",pred.json()},{"current_device_read_rc",pred.device_read_rc},{"current_cuda_device",pred.current_device},{"down_up_hits",hits},{"source_scope","real Tensor public metadata only; pool offset UNKNOWN"}};
 std::ofstream f(diagnostic.stem+"-projection-descriptors.jsonl",std::ios::app);f<<q.dump()<<'\n';f.flush();if(!f)throw std::runtime_error("437 descriptor persistence failed");
}
inline rt::Tensor project(rt::Runtime&r,const std::shared_ptr<infinilm::nn::Linear>&layer,rt::Tensor&input,int which){
 if(!selected)return layer->forward(r,input);
 rt::Tensor weight,output;
 struct FailureDrain{rt::Runtime&r;int count=std::uncaught_exceptions();~FailureDrain()noexcept{if(std::uncaught_exceptions()>count)try{r.syncStream();}catch(...){std::terminate();}}}drain{r};
 Predicates pred;bool failure_saved=false;
 try{
  try{r.assertOwnerThread();pred.add("owner_thread",true);}catch(...){pred.add("owner_thread",false);throw;}
  const bool valid_which=which==0||which==1;const size_t n=which==0?3584:7168,k=which==0?7168:3584;
  pred.add("which0or1",valid_which);pred.add("not_graph",!r.isGraphRecording());pred.add("Runtime_Hygon",r.device().type()==rt::Device::Type::kHygon);pred.add("actual_local_rank_device",r.device().index()==diagnostic.local_rank);pred.add("layer_present",bool(layer));pred.add("input_present",bool(input));
  if(layer)weight=layer->weight();
  const bool rank2=input&&input->shape().size()==2&&input->strides().size()==2;
  pred.add("input_exact_2D_1K",rank2&&input->shape()==rt::Shape{1,k});
  pred.add("input_BF16",input&&input->dtype()==rt::DataType::kBFloat16);
  pred.add("input_same_device",input&&input->device()==r.device());
  pred.add("input_positive_unused_rowpitch_and_inner1",rank2&&kimi_r437_row::input_row(input->size(0),input->size(1),input->stride(0),input->stride(1),k));
  pred.add("input_logical_span",input&&input->numel()==k&&input->nbytes()==kimi_r437_row::bytes(k));
  pred.add("canary_storage_lease_present",bool(diagnostic.canary_storage));
  pred.add("down_actual_data_span_within_strong_storage",which!=0||(input&&diagnostic.canary_storage&&kimi_r437_row::within(reinterpret_cast<uintptr_t>(input->data()),reinterpret_cast<uintptr_t>(diagnostic.canary_storage->data()),diagnostic.canary_storage->nbytes(),kimi_r437_row::bytes(k))));
  pred.add("weight_present",bool(weight));pred.add("weight_BF16",weight&&weight->dtype()==rt::DataType::kBFloat16);pred.add("weight_same_device",weight&&weight->device()==r.device());
  pred.add("weight_exact_NK_row_contiguous",weight&&weight->shape().size()==2&&weight->strides().size()==2&&kimi_r437_row::weight(weight->size(0),weight->size(1),weight->stride(0),weight->stride(1),n,k));
  int device=-1;const int device_rc=r.device().type()==rt::Device::Type::kHygon?cudaGetDevice(&device):int(cudaErrorInvalidDevice);pred.current_device=device;pred.device_read_rc=device_rc;pred.add("current_cuda_device_owned",device_rc==cudaSuccess&&device==r.device().index());
  const bool valid=pred.all();
  if(!valid){persist(r,which,input,weight,output,pred,"guard_rejected","exact physical single-row contract");failure_saved=true;throw std::invalid_argument("437 exact physical single-row projection guard rejected; descriptors persisted");}
  output=rt::Tensor::empty(r,{1,n},rt::DataType::kBFloat16);
  pred.add("output_exact_BF16_owned_1N",output&&output->dtype()==rt::DataType::kBFloat16&&output->device()==r.device()&&output->shape()==rt::Shape{1,n}&&output->strides()==rt::Strides{static_cast<rt::Stride>(n),1});
  pred.add("output_not_alias_input_weight",output&&output->data()!=input->data()&&output->data()!=weight->data());
  if(diagnostic.arm==1||!pred.get("output_exact_BF16_owned_1N")||!pred.get("output_not_alias_input_weight"))persist(r,which,input,weight,output,pred,"allocated_before_launch");
  if(!pred.get("output_exact_BF16_owned_1N")||!pred.get("output_not_alias_input_weight")){failure_saved=true;throw std::invalid_argument("437 output contract failed; descriptors persisted");}
  // Original fixed432 kernel reads x[j], not rowpitch; weight rows retain strict k stride.
  const int rc=kimi_lowrank_gemv_r432(which,output->data(),input->data(),weight->data(),n,k,k,r.stream());
  if(rc){persist(r,which,input,weight,output,pred,"kernel_launch_error",std::to_string(rc));failure_saved=true;throw std::runtime_error("437 reused432 GEMV launch error");}
  ++hits.at(which);return output;
 }catch(const std::exception&e){
  if(!failure_saved){try{persist(r,which,input,weight,output,pred,"exception",e.what());}catch(...){/* preserve primary; original owner failstop cleanup remains required */}}
  throw;
 }
}
}
