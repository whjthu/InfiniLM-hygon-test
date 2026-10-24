#include "deepseek_v4_moe_routed_r20.hpp"
#include "deepseek_routed_mlp_batch_r20.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <chrono>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <infini/rt/device_guard.h>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {void trace(const char *stage){if(std::getenv("DEEPSEEK_MOE_TRACE")){std::fprintf(stderr,"DEEPSEEK_MOE_NATIVE_STAGE %s\n",stage);std::fflush(stderr);}}}

namespace {
thread_local std::array<uint64_t,5>pack_counts{};
rt::Tensor packed_prefill_r8750(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&indices,const rt::Tensor&route_weights,const std::vector<std::shared_ptr<DeepseekV4Int8MLPBankR20>>&modules,bool packed){
 const size_t tokens=x->size(0),topk_=6,experts_=256,hidden_=4096;const auto device_=r.device();
 if(tokens<=64||tokens>4096||indices->shape()!=rt::Shape{tokens,6}||indices->dtype()!=rt::DataType::kInt64||route_weights->shape()!=rt::Shape{tokens,6}||route_weights->dtype()!=rt::DataType::kFloat32||modules.size()!=256)throw std::invalid_argument("prefill>64 original geometry");
 auto alloc=[&](rt::Shape shape,rt::DataType dtype){return rt::Tensor::empty(r,shape,dtype);};
 trace("read_route_metadata");std::vector<int64_t> routes(tokens*topk_);r.syncStream();r.memcpyD2H(routes.data(),indices->data(),routes.size()*sizeof(int64_t));
 std::map<size_t,std::vector<size_t>> grouped;for(size_t i=0;i<routes.size();++i){if(routes[i]<0||size_t(routes[i])>=experts_)throw std::invalid_argument("DeepSeek selected expert outside pool");grouped[size_t(routes[i])].push_back(i);}
 struct Lease{rt::Runtime&r;std::vector<int64_t> hrows,hroutes;rt::Tensor rows,routes;bool complete=false;~Lease(){if(!complete)try{r.syncStream();}catch(...){std::terminate();}}} lease{r};
 auto contributions=alloc({tokens*topk_,hidden_},rt::DataType::kFloat32);
 size_t offset=0;uint64_t upload_calls=0,upload_syncs=0;
 if(packed){
  lease.hrows.reserve(routes.size());lease.hroutes.reserve(routes.size());
  for(const auto &[expert,positions]:grouped)for(auto position:positions){lease.hrows.push_back(position/topk_);lease.hroutes.push_back(position);}
  lease.rows=alloc({routes.size()},rt::DataType::kInt64);lease.routes=alloc({routes.size()},rt::DataType::kInt64);
  auto hr=rt::Tensor::from_blob(lease.hrows.data(),{routes.size()},rt::DataType::kInt64,{rt::Device::Type::kCpu,0});auto hp=rt::Tensor::from_blob(lease.hroutes.data(),{routes.size()},rt::DataType::kInt64,{rt::Device::Type::kCpu,0});
  lease.rows->copy_from(r,hr);lease.routes->copy_from(r,hp);r.syncStream();upload_calls=2;upload_syncs=1;
 }

 for(const auto &[expert,positions]:grouped){
  size_t rows=positions.size();std::vector<int64_t> row_ids(rows),route_ids(rows);if(!packed)for(size_t j=0;j<rows;++j){row_ids[j]=positions[j]/topk_;route_ids[j]=positions[j];}
  auto upload=[&](std::vector<int64_t>&v){auto host=rt::Tensor::from_blob(v.data(),{v.size()},rt::DataType::kInt64,{rt::Device::Type::kCpu,0});auto out=alloc({v.size()},rt::DataType::kInt64);out->copy_from(r,host);r.syncStream();++upload_calls;++upload_syncs;return out;};
  trace("expert_input_gather");auto row_index=packed?lease.rows->narrow({{0,offset,rows}}):upload(row_ids),route_index=packed?lease.routes->narrow({{0,offset,rows}}):upload(route_ids);offset+=rows;auto gathered=alloc({rows,hidden_},rt::DataType::kBFloat16),w=alloc({rows,1},rt::DataType::kFloat32);
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),x,row_index,int64_t{0},gathered);
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),route_weights->view({tokens*topk_,1}),route_index,int64_t{0},w);
  trace("expert_int8_mlp");auto local=modules.at(expert)->forward_local(r,gathered);auto f=alloc({rows,hidden_},rt::DataType::kFloat32),weighted=alloc({rows,hidden_},rt::DataType::kFloat32);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),local,f);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),f,w,weighted);
  for(size_t j=0;j<rows;++j)contributions->narrow({{0,positions[j],1}})->copy_from(r,weighted->narrow({{0,j,1}}));r.syncStream();
 }
 trace("sum_routed");auto routed=alloc({tokens,hidden_},rt::DataType::kFloat32);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),contributions->view({tokens,topk_,hidden_}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},routed);
  // Keep every fallback contribution alive until its Sum has completed.
  r.syncStream();lease.complete=true;pack_counts={grouped.size(),routes.size(),upload_calls,upload_syncs,2*routes.size()*sizeof(int64_t)};return routed;
}
}
DeepseekV4MoERoutedR20::DeepseekV4MoERoutedR20(rt::Runtime &r,size_t hidden,size_t intermediate,size_t shared_intermediate,size_t experts,size_t topk,size_t vocab,bool hash,double factor,const engine::distributed::RankInfo &rank)
 :device_(r.device()),hidden_(hidden),experts_(experts),topk_(topk),vocab_(vocab),hash_(hash),tp_size_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||hidden!=4096||experts!=256||topk!=6||!vocab||!std::isfinite(factor)||factor<=0||(tp_size_!=1&&tp_size_!=8)||(tp_size_>1&&!comm_))throw std::invalid_argument("DeepSeek eager MoE requires explicit Hygon TP1/TP8 ownership and real model geometry");
 try{
  gate_=rt::Tensor::empty_parameter(r,{experts,hidden},rt::DataType::kBFloat16);bind_weight("gate.weight",weights::Weight(gate_));
  if(hash){table_=rt::Tensor::empty_parameter(r,{vocab,topk},rt::DataType::kInt64);bind_weight("gate.tid2eid",weights::Weight(table_));}
  else{bias_=rt::Tensor::empty_parameter(r,{experts},rt::DataType::kFloat32);bind_weight("gate.bias",weights::Weight(bias_));}
  if(tp_size_!=8||intermediate!=2048)throw std::invalid_argument("DeepSeek routed bank candidate requires actual TP8 experts");
  w13_bank_=rt::Tensor::empty_parameter(r,{256,512,4096},rt::DataType::kInt8);
  s13_bank_=rt::Tensor::empty_parameter(r,{256,512,1},rt::DataType::kFloat32);
  w2_bank_=rt::Tensor::empty_parameter(r,{256,4096,256},rt::DataType::kInt8);
  s2_bank_=rt::Tensor::empty_parameter(r,{256,4096,1},rt::DataType::kFloat32);
  for(size_t i=0;i<experts;++i){
   auto slice=[&](const rt::Tensor&t,rt::Shape shape){return t->narrow({{0,i,1}})->view(shape);};
   auto child=std::make_shared<DeepseekV4Int8MLPBankR20>(r,hidden,intermediate,rt::DataType::kBFloat16,rank,
      slice(w13_bank_,{512,4096}),slice(s13_bank_,{512,1}),slice(w2_bank_,{4096,256}),slice(s2_bank_,{4096,1}));
   experts_modules_.push_back(bind_child("experts."+std::to_string(i),child));
  }
  shared_=bind_child("shared_experts",std::make_shared<DeepseekV4Int8MLPBankR20>(r,hidden,shared_intermediate,rt::DataType::kBFloat16,rank));
  float eps=1e-20f,scale=static_cast<float>(factor);
  epsilon_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);factor_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);
  auto he=rt::Tensor::from_blob(&eps,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}),hf=rt::Tensor::from_blob(&scale,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0});
  epsilon_->copy_from(r,he);factor_->copy_from(r,hf);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void DeepseekV4MoERoutedR20::process_weights_after_loading(rt::Runtime &r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek eager MoE postload state differs");r.syncStream();ready_=true;}
DeepseekV4MoERoutedR20::Output DeepseekV4MoERoutedR20::forward(rt::Runtime &r,const rt::Tensor &x,const rt::Tensor &token_ids)const{
 r.assertOwnerThread();if(!ready_||r.isGraphRecording()||r.device()!=device_||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=hidden_)throw std::invalid_argument("DeepSeek eager MoE requires loaded packed BF16 input");
 size_t tokens=x->size(0);auto alloc=[&](rt::Shape shape,rt::DataType type){return rt::Tensor::empty(r,shape,type);};
 auto logits=alloc({tokens,experts_},rt::DataType::kFloat32),soft=alloc({tokens,experts_},rt::DataType::kFloat32),scores=alloc({tokens,experts_},rt::DataType::kFloat32);
 trace("gate_projection");ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),x,gate_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);
 ex::submit<infini::ops::Softplus>(r,infini::ops::Softplus::DefaultConfig(device_.type()),logits,double{1},double{20},soft);
 ex::submit<infini::ops::Sqrt>(r,infini::ops::Sqrt::DefaultConfig(device_.type()),soft,scores);
 auto indices=alloc({tokens,topk_},rt::DataType::kInt64),selected_weights=alloc({tokens,topk_},rt::DataType::kFloat32);
 trace("select_routes");if(hash_){
  if(!token_ids||token_ids->device()!=device_||token_ids->dtype()!=rt::DataType::kInt64||token_ids->shape()!=rt::Shape{tokens})throw std::invalid_argument("DeepSeek hash routing requires packed token IDs");
  std::vector<int64_t> ids(tokens);r.syncStream();r.memcpyD2H(ids.data(),token_ids->data(),tokens*sizeof(int64_t));for(auto id:ids)if(id<0||size_t(id)>=vocab_)throw std::invalid_argument("DeepSeek token ID outside hash table");
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),table_,token_ids,int64_t{0},indices);
 }else{
  auto selection=alloc({tokens,experts_},rt::DataType::kFloat32),values=alloc({tokens,topk_},rt::DataType::kFloat32);
  ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),scores,bias_,selection);
  ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(device_.type()),selection,int64_t(topk_),int64_t{1},true,true,values,indices);
 }
 ex::submit<infini::ops::Gather>(r,infini::ops::Gather::DefaultConfig(device_.type()),scores,indices,int64_t{1},false,selected_weights);
 auto sum=alloc({tokens,1},rt::DataType::kFloat32),denominator=alloc({tokens,1},rt::DataType::kFloat32),normalized=alloc({tokens,topk_},rt::DataType::kFloat32),route_weights=alloc({tokens,topk_},rt::DataType::kFloat32);
 ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),selected_weights,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},sum);
 ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),sum,epsilon_,denominator);
 ex::submit<infini::ops::Div>(r,infini::ops::Div::DefaultConfig(device_.type()),selected_weights,denominator,normalized);
 ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),normalized,factor_,route_weights);
 rt::Tensor routed;
 if(tokens<=64){
  trace("gpu_batched_int8_routes");
  routed=routed_r20::forward(r,x,indices,route_weights,w13_bank_,s13_bank_,w2_bank_,s2_bank_);
 }else{
 routed=packed_prefill_r8750(r,x,indices,route_weights,experts_modules_,[](){const char*p=std::getenv("INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750");if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;throw std::invalid_argument("pack policy must0/1");}());
 }
 trace("shared_int8_mlp");auto shared=shared_->forward_local(r,x),shared_float=alloc({tokens,hidden_},rt::DataType::kFloat32),combined=alloc({tokens,hidden_},rt::DataType::kFloat32);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),shared,shared_float);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),routed,shared_float,combined);
 trace("tp_reduction");if(tp_size_>1)engine::distributed::allreduce_(r,combined,combined,infinicclSum,comm_);
 auto output=alloc({tokens,hidden_},rt::DataType::kBFloat16);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),combined,output);return {output,route_weights,indices};
}
}

namespace infinilm::models::deepseek_v4 {
namespace {
thread_local std::string pack_error,pack_report;
struct PackContext{std::shared_ptr<rt::Runtime>r;int device;std::vector<rt::Tensor>banks;std::vector<std::shared_ptr<DeepseekV4Int8MLPBankR20>>modules;};
void pcuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void powner(PackContext*c){if(!c||!c->r)throw std::invalid_argument("context");c->r->assertOwnerThread();int dev=-1;pcuda(cudaGetDevice(&dev));if(dev!=c->device||c->r->isGraphRecording())throw std::invalid_argument("owner/device/eager");}
rt::Tensor borrow(PackContext*c,void*p,rt::Shape shape,rt::DataType dtype){if(!p)throw std::invalid_argument("null");cudaPointerAttributes a{};pcuda(cudaPointerGetAttributes(&a,p));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("GPU device pointer");return rt::Tensor::from_blob(p,shape,dtype,c->r->device());}
}
extern "C" const char*deep_prefill_pack_error_r8750(){return pack_error.c_str();}
extern "C" const char*deep_prefill_pack_report_r8750(){return pack_report.c_str();}
extern "C" void*deep_prefill_pack_create_r8750(int device,void*const*p){try{
 if(!p)throw std::invalid_argument("bank pointers");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<PackContext>();c->device=device;c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
 c->banks={borrow(c.get(),p[0],{256,512,4096},rt::DataType::kInt8),borrow(c.get(),p[1],{256,512,1},rt::DataType::kFloat32),borrow(c.get(),p[2],{256,4096,256},rt::DataType::kInt8),borrow(c.get(),p[3],{256,4096,1},rt::DataType::kFloat32)};
 engine::distributed::RankInfo rank{};rank.device=c->r->device();rank.tp_size=8;rank.tp_rank=0;
 for(size_t expert=0;expert<256;++expert){auto slice=[&](size_t i,rt::Shape shape){return c->banks[i]->narrow({{0,expert,1}})->view(shape);};auto m=std::make_shared<DeepseekV4Int8MLPBankR20>(*c->r,4096,2048,rt::DataType::kBFloat16,rank,slice(0,{512,4096}),slice(1,{512,1}),slice(2,{4096,256}),slice(3,{4096,1}));m->process_weights_after_loading(*c->r,false);c->modules.push_back(m);}
 return c.release();}catch(const std::exception&e){pack_error=e.what();return nullptr;}}
extern "C" int deep_prefill_pack_forward_r8750(void*raw,int mode,size_t tokens,void*const*p){auto*c=static_cast<PackContext*>(raw);try{
 powner(c);if(!p||mode<0||mode>1||tokens<=64||tokens>4096)throw std::invalid_argument("prefillmode/token contract");auto&r=*c->r;auto x=borrow(c,p[0],{tokens,4096},rt::DataType::kBFloat16),ids=borrow(c,p[1],{tokens,6},rt::DataType::kInt64),weights=borrow(c,p[2],{tokens,6},rt::DataType::kFloat32),out=borrow(c,p[3],{tokens,4096},rt::DataType::kFloat32);
 r.syncStream();auto begin=std::chrono::steady_clock::now();auto value=packed_prefill_r8750(r,x,ids,weights,c->modules,mode==1);out->copy_from(r,value);r.syncStream();auto end=std::chrono::steady_clock::now();
 pack_report=nlohmann::json({{"mode",mode},{"tokens",tokens},{"native_routed_copy_drain_seconds",std::chrono::duration<double>(end-begin).count()},{"unique_experts",pack_counts[0]},{"routes",pack_counts[1]},{"metadata_H2D_calls",pack_counts[2]},{"metadata_H2D_syncs",pack_counts[3]},{"metadata_H2D_bytes",pack_counts[4]}}).dump();return 0;
 }catch(const std::exception&e){pack_error=e.what();if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int deep_prefill_pack_destroy_r8750(void*raw,int*expired){auto*c=static_cast<PackContext*>(raw);try{powner(c);if(!expired)throw std::invalid_argument("expired");std::weak_ptr<rt::Runtime>w=c->r;c->r->syncStream();c->modules.clear();c->banks.clear();c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){pack_error=e.what();return 1;}}
}
